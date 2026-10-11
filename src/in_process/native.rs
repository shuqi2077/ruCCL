//! Rank coordination for device-resident collective adapters; no host tensor staging.
use ruda_core::device::DeviceId;
use ruda_tensor::distributed::ReduceOperation;
use std::{
    collections::HashMap,
    error::Error,
    fmt::{Display, Formatter},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Condvar, Mutex},
};

/// Device executor used by ruCCL's in-process native communicator.
pub trait NativeCollectiveDevice: Clone + Send + Sync + 'static {
    /// Owned device tensor; its elements must remain on the device.
    type Tensor: Send + 'static;
    /// Owned tensor and producer-stream capture made on the submitting rank.
    type Prepared: Send + 'static;
    /// Device preparation or collective execution failure.
    type Error: Error + Send + Sync + 'static;
    /// Capture the tensor and its producer stream, without downloading its elements.
    fn prepare(
        &self,
        tensor: Self::Tensor,
        device: DeviceId,
    ) -> Result<Self::Prepared, Self::Error>;
    /// Execute the agreed collective in rank order and complete its device streams.
    fn all_reduce(
        &self,
        devices: &[DeviceId],
        inputs: &[Self::Prepared],
        operation: ReduceOperation,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;
}

/// A collective failure delivered to every rank participating in the same generation.
#[derive(Clone, Debug)]
pub struct NativeCollectiveError(pub String);
impl Display for NativeCollectiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl Error for NativeCollectiveError {}

struct Submission<P> {
    operation: ReduceOperation,
    prepared: Result<P, NativeCollectiveError>,
}
struct State<D: NativeCollectiveDevice> {
    inputs: Vec<Option<Submission<D::Prepared>>>,
    outputs: Vec<Option<Result<D::Tensor, NativeCollectiveError>>>,
    arrived: usize,
    departed: usize,
    ready: bool,
}

/// ruCCL-owned ordered device group and reusable collective-generation rendezvous.
pub struct NativeCommunicator<D: NativeCollectiveDevice> {
    devices: Vec<DeviceId>,
    device: D,
    state: Mutex<State<D>>,
    changed: Condvar,
}

impl<D: NativeCollectiveDevice> NativeCommunicator<D> {
    /// Create a group of distinct devices in the caller's exact rank order.
    pub fn new(devices: Vec<DeviceId>, device: D) -> Result<Self, NativeCollectiveError> {
        if devices.is_empty()
            || devices
                .iter()
                .enumerate()
                .any(|(i, id)| devices[..i].contains(id))
        {
            return Err(NativeCollectiveError(
                "native collective requires distinct devices and a nonempty world".into(),
            ));
        }
        let world = devices.len();
        Ok(Self {
            devices,
            device,
            state: Mutex::new(State {
                inputs: (0..world).map(|_| None).collect(),
                outputs: (0..world).map(|_| None).collect(),
                arrived: 0,
                departed: 0,
                ready: false,
            }),
            changed: Condvar::new(),
        })
    }

    /// Number of actual device ranks in this group.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }
    /// Actual devices in rank order.
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Submit one rank's tensor and receive that rank's completed all-reduce result.
    pub fn all_reduce(
        &self,
        rank: usize,
        tensor: D::Tensor,
        operation: ReduceOperation,
    ) -> Result<D::Tensor, NativeCollectiveError> {
        let device = *self.devices.get(rank).ok_or_else(|| {
            NativeCollectiveError("native collective rank is outside the world".into())
        })?;
        let prepared = catch_unwind(AssertUnwindSafe(|| self.device.prepare(tensor, device)))
            .map_err(|_| NativeCollectiveError("native device preparation panicked".into()))
            .and_then(|result| result.map_err(|error| NativeCollectiveError(error.to_string())));
        let mut state = self
            .state
            .lock()
            .map_err(|_| NativeCollectiveError("native collective state is poisoned".into()))?;
        while state.ready || state.inputs[rank].is_some() {
            state = self
                .changed
                .wait(state)
                .map_err(|_| NativeCollectiveError("native collective state is poisoned".into()))?;
        }
        state.inputs[rank] = Some(Submission {
            operation,
            prepared,
        });
        state.arrived += 1;
        if state.arrived == self.world_size() {
            let submissions = state
                .inputs
                .iter_mut()
                .map(|entry| entry.take().expect("all native ranks arrived"))
                .collect::<Vec<_>>();
            let agreed = submissions[0].operation;
            let result = if submissions.iter().any(|entry| entry.operation != agreed) {
                Err(NativeCollectiveError(
                    "native collective reduction modes differ between ranks".into(),
                ))
            } else {
                submissions
                    .into_iter()
                    .map(|entry| entry.prepared)
                    .collect::<Result<Vec<_>, _>>()
                    .and_then(|inputs| {
                        catch_unwind(AssertUnwindSafe(|| {
                            self.device.all_reduce(&self.devices, &inputs, agreed)
                        }))
                        .map_err(|_| {
                            NativeCollectiveError("native collective executor panicked".into())
                        })
                        .and_then(|result| {
                            result.map_err(|error| NativeCollectiveError(error.to_string()))
                        })
                    })
            };
            match result {
                Ok(outputs) if outputs.len() == self.world_size() => {
                    state.outputs = outputs.into_iter().map(|output| Some(Ok(output))).collect();
                }
                result => {
                    let error = match result {
                        Err(error) => error,
                        Ok(outputs) => NativeCollectiveError(format!(
                            "native collective returned {} outputs for {} ranks",
                            outputs.len(),
                            self.world_size()
                        )),
                    };
                    state.outputs = (0..self.world_size())
                        .map(|_| Some(Err(error.clone())))
                        .collect();
                }
            }
            state.ready = true;
            self.changed.notify_all();
        } else {
            while !state.ready {
                state = self.changed.wait(state).map_err(|_| {
                    NativeCollectiveError("native collective state is poisoned".into())
                })?;
            }
        }
        let output = state.outputs[rank]
            .take()
            .expect("one native output per rank");
        state.departed += 1;
        if state.departed == self.world_size() {
            state.arrived = 0;
            state.departed = 0;
            state.ready = false;
            self.changed.notify_all();
        }
        output
    }
}

/// ruCCL registry owning the native communicators for an execution adapter.
pub struct NativeCommunicators<D: NativeCollectiveDevice> {
    device: D,
    groups: Mutex<HashMap<Vec<DeviceId>, Arc<NativeCommunicator<D>>>>,
}
impl<D: NativeCollectiveDevice> NativeCommunicators<D> {
    /// Create the registry for a device-resident executor.
    pub fn new(device: D) -> Self {
        Self {
            device,
            groups: Mutex::new(HashMap::new()),
        }
    }
    /// Reuse or create the communicator for this exact ordered device group.
    pub fn communicator(
        &self,
        devices: Vec<DeviceId>,
    ) -> Result<Arc<NativeCommunicator<D>>, NativeCollectiveError> {
        let mut groups = self.groups.lock().map_err(|_| {
            NativeCollectiveError("native communicator registry is poisoned".into())
        })?;
        if let Some(group) = groups.get(&devices) {
            return Ok(group.clone());
        }
        let group = Arc::new(NativeCommunicator::new(
            devices.clone(),
            self.device.clone(),
        )?);
        groups.insert(devices, group.clone());
        Ok(group)
    }
}
