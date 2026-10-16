use super::*;
use crate::rank::{
    AlgorithmPolicy, CollectiveTuning, TcpRankSession, TcpRendezvousServer, TracingRankTransport,
    TransportTraceEvent, TransportTraceOperation, TransportTraceSink, UniqueId, work::WorkError,
};
use ruda_tensor::ops::FloatTensorOps;
use ruda_tensor::{FloatDType, TensorData, read_sync};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[cfg(not(feature = "test-cuda"))]
type Compute = ruda_tensor_host::Host;
#[cfg(feature = "test-cuda")]
type Compute = ruda_tensor_device::cuda::Cuda<f32>;
type Matrix = <Compute as ruda_tensor::BackendTypes>::FloatTensorPrimitive;

#[derive(Debug, Default)]
struct Receives {
    count: Mutex<usize>,
    changed: Condvar,
}

impl TransportTraceSink for Receives {
    fn record(&self, event: &TransportTraceEvent) {
        if event.operation == TransportTraceOperation::Receive && event.success {
            *self.count.lock().unwrap() += 1;
            self.changed.notify_all();
        }
    }
}

fn world(
    ring: Vec<u32>,
    expect_abort: bool,
    test: impl Fn(u32, RankCommunicator<TensorDevice<Compute>>, Arc<Receives>) + Send + Sync + 'static,
) {
    let size = ring.len() as u32;
    let id = UniqueId::from_bytes([0xf1; 16]);
    let server = TcpRendezvousServer::bind("127.0.0.1:0", id, size as usize).unwrap();
    let address = server.local_addr().unwrap();
    let coordinator = std::thread::spawn(move || server.run());
    let test = Arc::new(test);
    let workers = (0..size)
        .map(|rank| {
            let test = test.clone();
            let ring = ring.clone();
            std::thread::spawn(move || {
                let session =
                    TcpRankSession::connect(address, id, rank, size, Duration::from_secs(5))
                        .unwrap();
                let receives = Arc::new(Receives::default());
                let transport = TracingRankTransport::new(session, receives.clone());
                let tuning = CollectiveTuning::new(AlgorithmPolicy::Ring, 0, ring).unwrap();
                let communicator = RankCommunicator::from_transport_with_tuning(
                    || Ok::<_, TensorDeviceError>(TensorDevice::<Compute>::new(Default::default())),
                    transport,
                    tuning,
                    "ag-matmul-test",
                )
                .unwrap();
                test(rank, communicator, receives);
            })
        })
        .collect::<Vec<_>>();
    let results = workers
        .into_iter()
        .map(|worker| worker.join())
        .collect::<Vec<_>>();
    let server_result = coordinator.join().unwrap();
    for result in results {
        result.unwrap();
    }
    if !expect_abort {
        server_result.unwrap();
    }
}

fn matrix(values: Vec<f32>, rows: usize, cols: usize) -> Matrix {
    Compute::float_from_data(TensorData::new(values, [rows, cols]), &Default::default())
}

fn values(tensor: Matrix) -> Vec<f32> {
    read_sync(Compute::float_into_data(Compute::float_cast(
        tensor,
        FloatDType::F32,
    )))
    .unwrap()
    .into_vec::<f32>()
    .unwrap()
}

#[test]
fn matmul_matches_scalar_oracle_with_rank_local_weights_and_noncontiguous_inputs() {
    for ring in [vec![0], vec![0, 2, 1], vec![2, 0, 3, 1]] {
        world(ring, false, |rank, comm, _| {
            for (m, k, n) in [(1, 1, 1), (3, 5, 7), (5, 3, 2)] {
                let input_values: Vec<_> = (0..m * k)
                    .map(|i| (i as i32 % 7 - 3) as f32 + rank as f32)
                    .collect();
                let weights: Vec<_> = (0..k * n)
                    .map(|i| (i % 5) as f32 - 2.0 + rank as f32)
                    .collect();
                let input_transposed: Vec<_> = (0..k)
                    .flat_map(|col| {
                        let input_values = &input_values;
                        (0..m).map(move |row| input_values[row * k + col])
                    })
                    .collect();
                let weight_transposed: Vec<_> = (0..n)
                    .flat_map(|col| {
                        let weights = &weights;
                        (0..k).map(move |row| weights[row * n + col])
                    })
                    .collect();
                let input = Compute::float_swap_dims(matrix(input_transposed, k, m), 0, 1);
                let weight = Compute::float_swap_dims(matrix(weight_transposed, n, k), 0, 1);
                let (result, stats) = comm
                    .all_gather_matmul_float(input.clone(), weight.clone())
                    .unwrap();
                assert_eq!(
                    result.shape(),
                    Shape::new([comm.world_size() as usize * m, n])
                );
                let mut expected = Vec::new();
                for source in 0..comm.world_size() {
                    for row in 0..m {
                        for col in 0..n {
                            expected.push(
                                (0..k)
                                    .map(|inner| {
                                        let x = ((row * k + inner) as i32 % 7 - 3) as f32
                                            + source as f32;
                                        x * weights[inner * n + col]
                                    })
                                    .sum::<f32>(),
                            );
                        }
                    }
                }
                assert_eq!(values(result), expected);
                assert_eq!(values(input), input_values);
                assert_eq!(values(weight), weights);
                assert_eq!(stats.steps, comm.world_size() - 1);
                assert_eq!(
                    stats.transferred_bytes,
                    (2 * (comm.world_size() as usize - 1) * m * k * 4) as u64
                );
            }
        });
    }
}

#[test]
fn matmul_preserves_half_dtypes() {
    world(vec![0, 1], false, |rank, comm, _| {
        for dtype in [FloatDType::F16, FloatDType::BF16] {
            let input = Compute::float_cast(matrix(vec![rank as f32 + 1.0, 2.0], 1, 2), dtype);
            let weight = Compute::float_cast(matrix(vec![3.0, 4.0], 2, 1), dtype);
            let (result, stats) = comm.all_gather_matmul_float(input, weight).unwrap();
            assert_eq!(result.dtype(), DType::from(dtype));
            assert_eq!(values(result), vec![11.0, 14.0]);
            assert_eq!(stats.transferred_bytes, 8);
        }
    });
}

#[test]
fn streaming_communicates_while_consumer_is_active_and_does_not_alias_input() {
    world(vec![2, 0, 3, 1], false, |rank, comm, receives| {
        let input = comm.execution().upload(&[rank as f32]).unwrap();
        let mut order = Vec::new();
        let stats = comm
            .tensor_collective::<f32>()
            .all_gather_ring_stream(&input, |source, shard| {
                if order.is_empty() {
                    assert_eq!(source, rank);
                    let (count, timeout) = receives
                        .changed
                        .wait_timeout_while(
                            receives.count.lock().unwrap(),
                            Duration::from_secs(3),
                            |count| *count == 0,
                        )
                        .unwrap();
                    assert!(
                        !timeout.timed_out(),
                        "communication stalled behind the active consumer"
                    );
                    assert!(
                        *count > 0 && *count < 3,
                        "must consume before all-gather completes"
                    );
                }
                assert_eq!(values(shard.float_tensor().unwrap()), vec![source as f32]);
                comm.execution().write(shard, &[-1.0])?;
                order.push(source);
                Ok(())
            })
            .unwrap();
        let ring = [2, 0, 3, 1];
        let position = ring.iter().position(|r| *r == rank).unwrap();
        assert_eq!(
            order,
            (0..4)
                .map(|i| ring[(position + 4 - i) % 4])
                .collect::<Vec<_>>()
        );
        assert_eq!(values(input.float_tensor().unwrap()), vec![rank as f32]);
        assert_eq!(stats.transferred_bytes, 24);
    });
}

#[test]
fn streaming_preserves_empty_shards_and_special_value_bits() {
    world(vec![0, 1, 2], false, |rank, comm, _| {
        for payload in [
            vec![],
            vec![
                f32::from_bits(0x7fc0_1234),
                f32::INFINITY,
                -0.0,
                rank as f32,
            ],
        ] {
            let input = comm.execution().upload(&payload).unwrap();
            let mut count = 0;
            comm.tensor_collective::<f32>()
                .all_gather_ring_stream(&input, |source, shard| {
                    let actual = values(shard.float_tensor()?);
                    if payload.is_empty() {
                        assert!(actual.is_empty());
                    } else {
                        assert_eq!(
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            [
                                0x7fc0_1234,
                                f32::INFINITY.to_bits(),
                                (-0.0_f32).to_bits(),
                                (source as f32).to_bits()
                            ]
                        );
                    }
                    count += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(count, 3);
        }
    });
}

#[test]
fn invalid_matrix_shapes_and_dtypes_return_errors_without_consuming_sequence() {
    world(vec![0], false, |_, comm, _| {
        assert!(matches!(
            comm.all_gather_matmul_float(matrix(vec![1.0; 6], 2, 3), matrix(vec![1.0; 2], 2, 1)),
            Err(TensorDeviceError::InvalidBuffer(_))
        ));
        let rank_one = Compute::float_reshape(matrix(vec![1.0; 2], 1, 2), Shape::new([2]));
        assert!(matches!(
            comm.all_gather_matmul_float(rank_one, matrix(vec![1.0; 2], 2, 1)),
            Err(TensorDeviceError::InvalidBuffer(_))
        ));
        let half_weight = Compute::float_cast(matrix(vec![1.0; 2], 2, 1), FloatDType::F16);
        assert!(matches!(
            comm.all_gather_matmul_float(matrix(vec![1.0; 2], 1, 2), half_weight),
            Err(TensorDeviceError::DTypeMismatch { .. })
        ));
        assert_eq!(
            values(
                comm.all_gather_matmul_float(matrix(vec![2.0], 1, 1), matrix(vec![3.0], 1, 1))
                    .unwrap()
                    .0
            ),
            vec![6.0]
        );
    });
}

#[test]
fn callback_errors_and_panics_join_workers_and_propagate() {
    for panic in [false, true] {
        world(vec![0, 1, 2, 3], true, move |rank, comm, _| {
            let input = comm.execution().upload(&[rank as f32]).unwrap();
            let result = comm
                .tensor_collective::<f32>()
                .all_gather_ring_stream(&input, |_, _| {
                    if rank == 0 {
                        if panic {
                            panic!("intentional consumer panic");
                        }
                        return Err(TensorDeviceError::InvalidOperation(
                            "intentional consumer error",
                        ));
                    }
                    Ok(())
                });
            if rank == 0 {
                if panic {
                    assert!(matches!(
                        result,
                        Err(TensorDeviceError::Work(WorkError::WorkerPanic))
                    ));
                } else {
                    assert!(matches!(
                        result,
                        Err(TensorDeviceError::InvalidOperation(
                            "intentional consumer error"
                        ))
                    ));
                }
            }
        });
    }
}

#[test]
fn mismatched_shard_lengths_fail_collective_agreement() {
    world(vec![0, 1], true, |rank, comm, _| {
        let input = comm
            .execution()
            .upload(&vec![1.0_f32; rank as usize + 1])
            .unwrap();
        assert!(
            comm.tensor_collective::<f32>()
                .all_gather_ring_stream(&input, |_, _| {
                    panic!("consumer must not run before successful agreement")
                })
                .is_err()
        );
    });
}
