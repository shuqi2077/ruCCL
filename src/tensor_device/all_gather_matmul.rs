use super::{TensorDevice, TensorDeviceError, TensorElement, storage};
use crate::rank::{CollectiveStats, communicator::RankCommunicator};
use ruda_tensor::{Backend, DType, Shape, TensorMetadata, bf16, f16};

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// Compute `cat(rank_shards, 0) @ weight`, consuming each ready shard
    /// while a host worker continues the all-gather ring. Inputs are [M, K],
    /// weight is [K, N], output is [world_size * M, N] in global rank order.
    /// All ranks must provide the same shard shape and dtype in call order;
    /// each rank may use its own weight. Input and weight are preserved.
    ///
    /// Each shard's GEMM is synchronized before consuming the next shard so
    /// lazy backends execute computation during communication. Transfers are
    /// host-staged; no cross-GPU stream or fused-kernel guarantee is made.
    pub fn all_gather_matmul_float(
        &self,
        input: B::FloatTensorPrimitive,
        weight: B::FloatTensorPrimitive,
    ) -> Result<(B::FloatTensorPrimitive, CollectiveStats), TensorDeviceError> {
        match input.dtype() {
            DType::F32 => self.all_gather_matmul::<f32>(input, weight),
            DType::F16 => self.all_gather_matmul::<f16>(input, weight),
            DType::BF16 => self.all_gather_matmul::<bf16>(input, weight),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    fn all_gather_matmul<T: TensorElement>(
        &self,
        input: B::FloatTensorPrimitive,
        weight: B::FloatTensorPrimitive,
    ) -> Result<(B::FloatTensorPrimitive, CollectiveStats), TensorDeviceError> {
        let execution = self.execution();
        execution.validate_type::<T>()?;
        if weight.dtype() != T::dtype() {
            return Err(TensorDeviceError::DTypeMismatch {
                expected: T::dtype(),
                actual: weight.dtype(),
            });
        }
        if &B::float_device(&input) != execution.device()
            || &B::float_device(&weight) != execution.device()
        {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        let shape = input.shape();
        let weight_shape = weight.shape();
        if shape.num_dims() != 2 || weight_shape.num_dims() != 2 {
            return Err(TensorDeviceError::InvalidBuffer(
                "all-gather matmul requires matrices",
            ));
        }
        if shape[1] != weight_shape[0] {
            return Err(TensorDeviceError::InvalidBuffer(
                "all-gather matmul K mismatch",
            ));
        }
        let checked_product = |dimensions: &[usize]| {
            dimensions
                .iter()
                .try_fold(1_usize, |total, dim| total.checked_mul(*dim))
                .ok_or(TensorDeviceError::InvalidBuffer(
                    "all-gather matmul shape overflow",
                ))
        };
        let length = checked_product(&[shape[0], shape[1]])?;
        storage::checked_length::<T>(length)?;
        storage::checked_length::<T>(checked_product(&[
            self.world_size() as usize,
            shape[0],
            weight_shape[1],
        ])?)?;
        let buffer = execution.import_float::<T>(B::float_reshape(input, Shape::new([length])))?;
        let mut outputs = vec![None; self.world_size() as usize];
        let stats =
            self.tensor_collective::<T>()
                .all_gather_ring_stream(&buffer, |rank, shard| {
                    let shard = B::float_reshape(shard.float_tensor()?, shape.clone());
                    let output = B::float_matmul(shard, weight.clone());
                    execution.synchronize()?;
                    outputs[rank as usize] = Some(output);
                    Ok(())
                })?;
        let outputs = outputs.into_iter().collect::<Option<Vec<_>>>().ok_or(
            TensorDeviceError::InvalidBuffer("all-gather matmul missing rank"),
        )?;
        Ok((B::float_cat(outputs, 0), stats))
    }
}

#[cfg(test)]
mod tests;
