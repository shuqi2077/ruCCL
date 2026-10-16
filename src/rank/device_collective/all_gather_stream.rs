use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::sync_channel;

impl<T, D> DeviceCollective<'_, T, D>
where
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    /// Consume each rank's shard as it arrives, starting with the local rank.
    /// A separate host worker forwards shards around the primary ring while
    /// the caller computes. At most one unconsumed shard is queued. Each
    /// callback receives independent device storage, not an alias of input.
    ///
    /// All ranks must call in the same order with equal lengths and element
    /// types. Callbacks must not issue collectives on this session. Callback
    /// errors/panics abort the session just like other ring worker failures.
    /// This is host-staged overlap, not GPU peer-memory kernel fusion.
    pub fn all_gather_ring_stream(
        &self,
        input: &D::Buffer,
        mut consume: impl FnMut(u32, &D::Buffer) -> Result<(), D::Error>,
    ) -> Result<CollectiveStats, D::Error> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let length = self.execution.buffer_len(input);
            let bytes = length
                .checked_mul(D::ELEMENT_SIZE)
                .ok_or(RankError::Overflow("streaming all-gather shard bytes"))?;
            let world = self.world_size() as usize;
            let ring = self.tuning.ring_order_for_channel(0);
            let position = self.ring_position(ring)?;
            let previous = ring[(position + world - 1) % world];
            let next = ring[(position + 1) % world];
            let transferred = bytes
                .checked_mul(world - 1)
                .and_then(|value| value.checked_mul(2))
                .ok_or(RankError::Overflow(
                    "streaming all-gather transferred bytes",
                ))?;
            let local = self.execution.copy_bytes_from_device_at(input, 0, length)?;
            if local.len() != bytes {
                return Err(RankError::InvalidLength("streaming all-gather input bytes").into());
            }
            let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
            self.ring_agreement(operation, length, 0x201, 1)?;
            std::thread::scope(|scope| {
                let (sender, receiver) = sync_channel(1);
                let worker = std::thread::Builder::new()
                    .name("ruccl-ag-stream".into())
                    .spawn_scoped(scope, move || {
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            let mut payload = local;
                            sender
                                .send((self.rank(), payload.clone()))
                                .map_err(|_| WorkError::WorkerQueueClosed)?;
                            for step in 0..world - 1 {
                                payload = self.exchange_host_payload(
                                    self.element_type,
                                    payload,
                                    bytes,
                                    next,
                                    previous,
                                    0,
                                    ring_data_tag(operation, 0, 0, step),
                                )?;
                                let source = ring[(position + world - step - 1) % world];
                                sender
                                    .send((source, payload.clone()))
                                    .map_err(|_| WorkError::WorkerQueueClosed)?;
                            }
                            Ok::<_, D::Error>(())
                        }))
                        .map_err(|_| D::Error::from(WorkError::WorkerPanic))
                        .and_then(|result| result);
                        self.propagate_channel_failure("streaming all-gather", result)
                    })
                    .map_err(|error| WorkError::WorkerStart(error.to_string()))?;
                let consumed = catch_unwind(AssertUnwindSafe(|| {
                    let mut count = 0;
                    for (source, payload) in &receiver {
                        let shard = self.execution.alloc(length)?;
                        self.execution
                            .copy_bytes_to_device_at(&shard, 0, &payload)?;
                        consume(source, &shard)?;
                        count += 1;
                    }
                    Ok::<_, D::Error>(count)
                }))
                .map_err(|_| D::Error::from(WorkError::WorkerPanic))
                .and_then(|result| result);
                drop(receiver);
                let consumed = self.propagate_channel_failure("all-gather consumer", consumed);
                let communicated = worker
                    .join()
                    .map_err(|_| D::Error::from(WorkError::WorkerPanic))
                    .and_then(|result| result);
                let count = consumed?;
                communicated?;
                if count != world {
                    return Err(WorkError::WorkerQueueClosed.into());
                }
                Ok::<_, D::Error>(())
            })?;
            self.stats(
                CollectiveAlgorithm::Ring,
                (world - 1) as u32,
                transferred,
                0,
            )
        }))
        .map_err(|_| D::Error::from(WorkError::WorkerPanic))
        .and_then(|result| result);
        self.propagate_channel_failure("streaming all-gather", result)
    }
}
