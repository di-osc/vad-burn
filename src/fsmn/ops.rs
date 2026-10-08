//! FSMN 前向计算使用的基础张量算子与权重读取工具。

use anyhow::{Context, Result, bail};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};
use burn_store::pytorch::PytorchReader;
use burn_store::pytorch_reader::DType;

use super::constants::{CACHE_FRAMES, CONV_KERNEL, PROJ_DIM};

/// 计算 FSMN 的 memory 投影：缓存帧与当前帧拼接后做因果卷积，再与输入相加。
pub fn fsmn_memory(
    input: Tensor<2>,
    cache: &Tensor<2>,
    kernel: Tensor<3>,
    frames: usize,
) -> Result<Tensor<2>> {
    if input.dims() != [frames, PROJ_DIM] {
        bail!("unexpected FSMN input shape: {:?}", input.dims());
    }
    if cache.dims() != [CACHE_FRAMES, PROJ_DIM] {
        bail!("unexpected FSMN cache shape: {:?}", cache.dims());
    }
    if kernel.dims() != [PROJ_DIM, 1, CONV_KERNEL] {
        bail!("unexpected FSMN kernel shape: {:?}", kernel.dims());
    }

    // 把 [cache; input] 视作 [1, PROJ_DIM, CACHE_FRAMES + frames] 送进 conv1d。
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

/// 根据当前输入和旧缓存滚动出下一次推理要用的缓存。
pub fn next_cache(input: &Tensor<2>, cache: &Tensor<2>) -> Result<Tensor<2>> {
    let [frames, proj_dim] = input.dims();
    if proj_dim != PROJ_DIM {
        bail!("unexpected FSMN cache input shape: {:?}", input.dims());
    }
    if cache.dims() != [CACHE_FRAMES, PROJ_DIM] {
        bail!("unexpected FSMN cache shape: {:?}", cache.dims());
    }

    // 拼接后只保留最后 CACHE_FRAMES 帧作为新缓存。
    let history = Tensor::cat(vec![cache.clone(), input.clone()], 0);
    let total = CACHE_FRAMES + frames;
    Ok(history.slice([total - CACHE_FRAMES..total, 0..PROJ_DIM]))
}

/// 对 logits 做 softmax，并只保留静音列（第 0 列）作为后验概率。
pub fn silence_posterior(logits: Tensor<2>, rows: usize) -> Result<Tensor<2>> {
    let [actual_rows, _cols] = logits.dims();
    if actual_rows != rows {
        bail!("unexpected FSMN output shape: {:?}", logits.dims());
    }
    Ok(burn::tensor::activation::softmax(logits, 1).slice([0..rows, 0..1]))
}

/// 把 2D 张量按行转换成 `Vec<Vec<f32>>`，方便后续逐帧后处理。
pub fn tensor_rows(tensor: Tensor<2>, rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
    if tensor.dims() != [rows, cols] {
        bail!("unexpected FSMN output shape: {:?}", tensor.dims());
    }
    let data = tensor
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("burn tensor data");
    Ok((0..rows)
        .map(|row_idx| data[row_idx * cols..(row_idx + 1) * cols].to_vec())
        .collect())
}

/// 读取 checkpoint 中某个张量的 shape。
pub fn snapshot_shape(reader: &PytorchReader, key: &str) -> Result<Vec<usize>> {
    Ok(reader
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?
        .shape()
        .to_vec())
}

/// 读取 checkpoint 中某个 F32 张量的原始字节并解释为 `f32` 向量。
pub fn load_vec(reader: &PytorchReader, key: &str) -> Result<Vec<f32>> {
    let snapshot = reader
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?;
    if snapshot.dtype() != DType::F32 {
        bail!("{key} must be F32, got {:?}", snapshot.dtype());
    }
    // pytorch-reader 返回的字节是 native-endian 的连续 f32。
    let bytes = snapshot
        .read()
        .with_context(|| format!("failed to read tensor {key}"))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect())
}

/// 读取并转置 FSMN 的 conv_left 权重，得到 conv1d 需要的 [PROJ_DIM, 1, CONV_KERNEL] 形状。
pub fn load_conv_left_weight(
    reader: &PytorchReader,
    device: &Device,
    key: &str,
) -> Result<Tensor<3>> {
    let data = load_vec(reader, key)?;
    if data.len() != PROJ_DIM * CONV_KERNEL {
        bail!(
            "{key} must have {} values, got {}",
            PROJ_DIM * CONV_KERNEL,
            data.len()
        );
    }
    // checkpoint 存的是 [CONV_KERNEL, PROJ_DIM]，需要转成 [PROJ_DIM, 1, CONV_KERNEL]。
    let mut conv_data = vec![0.0; data.len()];
    for channel in 0..PROJ_DIM {
        for k in 0..CONV_KERNEL {
            conv_data[channel * CONV_KERNEL + k] = data[k * PROJ_DIM + channel];
        }
    }
    Ok(Tensor::<3>::from_data(
        TensorData::new(conv_data, [PROJ_DIM, 1, CONV_KERNEL]),
        device,
    ))
}
