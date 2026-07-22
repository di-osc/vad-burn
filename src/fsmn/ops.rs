use anyhow::{Result, bail};
use burn::prelude::Backend as BurnBackend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Tensor, TensorData};
use burn_store::pytorch::PytorchReader;

use super::constants::{CACHE_FRAMES, CONV_KERNEL, PROJ_DIM};

pub fn fsmn_memory<B: BurnBackend>(
    input: Tensor<B, 2>,
    cache: &Tensor<B, 2>,
    kernel: Tensor<B, 3>,
    frames: usize,
) -> Result<Tensor<B, 2>> {
    if input.dims() != [frames, PROJ_DIM] {
        bail!("unexpected FSMN input shape: {:?}", input.dims());
    }
    if cache.dims() != [CACHE_FRAMES, PROJ_DIM] {
        bail!("unexpected FSMN cache shape: {:?}", cache.dims());
    }
    if kernel.dims() != [PROJ_DIM, 1, CONV_KERNEL] {
        bail!("unexpected FSMN kernel shape: {:?}", kernel.dims());
    }

    let conv_input = Tensor::cat(vec![cache.clone(), input.clone()], 0)
        .swap_dims(0, 1)
        .reshape([1, PROJ_DIM, CACHE_FRAMES + frames]);
    let memory = conv1d(
        conv_input,
        kernel,
        None,
        ConvOptions::new([1], [0], [1], PROJ_DIM),
    )
    .reshape([PROJ_DIM, frames])
    .swap_dims(0, 1);
    Ok(input + memory)
}

pub fn next_cache<B: BurnBackend>(
    input: &Tensor<B, 2>,
    cache: &Tensor<B, 2>,
) -> Result<Tensor<B, 2>> {
    let [frames, proj_dim] = input.dims();
    if proj_dim != PROJ_DIM {
        bail!("unexpected FSMN cache input shape: {:?}", input.dims());
    }
    if cache.dims() != [CACHE_FRAMES, PROJ_DIM] {
        bail!("unexpected FSMN cache shape: {:?}", cache.dims());
    }

    let history = Tensor::cat(vec![cache.clone(), input.clone()], 0);
    let total = CACHE_FRAMES + frames;
    Ok(history.slice([total - CACHE_FRAMES..total, 0..PROJ_DIM]))
}

pub fn silence_posterior<B: BurnBackend>(
    logits: Tensor<B, 2>,
    rows: usize,
) -> Result<Tensor<B, 2>> {
    let [actual_rows, _cols] = logits.dims();
    if actual_rows != rows {
        bail!("unexpected FSMN output shape: {:?}", logits.dims());
    }
    Ok(burn::tensor::activation::softmax(logits, 1).slice([0..rows, 0..1]))
}

pub fn tensor_rows<B: BurnBackend>(
    tensor: Tensor<B, 2>,
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<f32>>> {
    if tensor.dims() != [rows, cols] {
        bail!("unexpected FSMN output shape: {:?}", tensor.dims());
    }
    let data = tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("burn tensor data");
    Ok((0..rows)
        .map(|row_idx| data[row_idx * cols..(row_idx + 1) * cols].to_vec())
        .collect())
}

pub fn snapshot_shape(reader: &PytorchReader, key: &str) -> Result<Vec<usize>> {
    Ok(reader
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?
        .shape
        .as_ref()
        .to_vec())
}

pub fn load_vec(reader: &PytorchReader, key: &str) -> Result<Vec<f32>> {
    let snapshot = reader
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?;
    if snapshot.dtype != DType::F32 {
        bail!("{key} must be F32, got {:?}", snapshot.dtype);
    }
    Ok(snapshot.to_data()?.convert::<f32>().into_vec::<f32>()?)
}

pub fn load_conv_left_weight<B: BurnBackend>(
    reader: &PytorchReader,
    device: &B::Device,
    key: &str,
) -> Result<Tensor<B, 3>> {
    let data = load_vec(reader, key)?;
    if data.len() != PROJ_DIM * CONV_KERNEL {
        bail!(
            "{key} must have {} values, got {}",
            PROJ_DIM * CONV_KERNEL,
            data.len()
        );
    }
    let mut conv_data = vec![0.0; data.len()];
    for channel in 0..PROJ_DIM {
        for k in 0..CONV_KERNEL {
            conv_data[channel * CONV_KERNEL + k] = data[k * PROJ_DIM + channel];
        }
    }
    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(conv_data, [PROJ_DIM, 1, CONV_KERNEL]),
        device,
    ))
}
