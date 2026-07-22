use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::nn::Linear;
use burn::prelude::Backend as BurnBackend;
use burn::tensor::{Tensor, TensorData};
use burn_store::pytorch::PytorchReader;

use super::constants::{Backend, CACHE_FRAMES, FEAT_DIM, LAYERS, PROJ_DIM};
use super::model::FeatureTensor;
use super::ops::{
    fsmn_memory, load_conv_left_weight, load_vec, next_cache, silence_posterior, snapshot_shape,
    tensor_rows,
};
use super::timing::FsmnForwardTiming;

pub struct BurnFsmnWeights<B: BurnBackend = Backend> {
    device: B::Device,
    in_linear1: BurnLinear<B>,
    in_linear2: BurnLinear<B>,
    blocks: Vec<BurnFsmnBlock<B>>,
    out_linear1: BurnLinear<B>,
    out_linear2: BurnLinear<B>,
}

struct BurnFsmnBlock<B: BurnBackend = Backend> {
    linear: BurnLinear<B>,
    affine: BurnLinear<B>,
    conv_left_weight: Tensor<B, 3>,
}

struct BurnLinear<B: BurnBackend = Backend> {
    inner: Linear<B>,
}

impl<B: BurnBackend> BurnFsmnWeights<B> {
    pub fn load(model_dir: &Path, device: B::Device) -> Result<Self> {
        let model_path = model_dir.join("model.pt");
        let reader = PytorchReader::new(&model_path)
            .with_context(|| format!("failed to load {}", model_path.display()))?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for idx in 0..LAYERS {
            blocks.push(BurnFsmnBlock {
                linear: BurnLinear::load(
                    &reader,
                    &device,
                    &format!("encoder.fsmn.{idx}.linear.linear"),
                    false,
                )?,
                affine: BurnLinear::load(
                    &reader,
                    &device,
                    &format!("encoder.fsmn.{idx}.affine.linear"),
                    true,
                )?,
                conv_left_weight: load_conv_left_weight(
                    &reader,
                    &device,
                    &format!("encoder.fsmn.{idx}.fsmn_block.conv_left.weight"),
                )?,
            });
        }
        Ok(Self {
            device: device.clone(),
            in_linear1: BurnLinear::load(&reader, &device, "encoder.in_linear1.linear", true)?,
            in_linear2: BurnLinear::load(&reader, &device, "encoder.in_linear2.linear", true)?,
            blocks,
            out_linear1: BurnLinear::load(&reader, &device, "encoder.out_linear1.linear", true)?,
            out_linear2: BurnLinear::load(&reader, &device, "encoder.out_linear2.linear", true)?,
        })
    }

    pub fn zero_caches(&self) -> Vec<Tensor<B, 2>> {
        (0..LAYERS)
            .map(|_| Tensor::<B, 2>::zeros([CACHE_FRAMES, PROJ_DIM], &self.device))
            .collect()
    }

    pub fn forward_frame_scores(
        &self,
        feats: FeatureTensor<B>,
        caches: &mut [Tensor<B, 2>],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_frame_scores_inner(feats, caches, false)
    }

    pub fn forward_frame_scores_streaming(
        &self,
        feats: FeatureTensor<B>,
        caches: &mut [Tensor<B, 2>],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_frame_scores_inner(feats, caches, true)
    }

    fn forward_frame_scores_inner(
        &self,
        feats: FeatureTensor<B>,
        caches: &mut [Tensor<B, 2>],
        update_caches: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let [frames, feat_dim] = feats.dims();
        if feat_dim != FEAT_DIM {
            bail!("FSMN VAD expects feature dim {FEAT_DIM}, got {feat_dim}");
        }
        if frames == 0 {
            return Ok(Vec::new());
        }

        let mut x = self.in_linear1.forward(feats, false);
        x = self.in_linear2.forward(x, true);

        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            x = block.forward(x, frames, cache, update_caches)?;
        }

        x = self.out_linear1.forward(x, false);
        x = self.out_linear2.forward(x, false);
        tensor_rows(silence_posterior(x, frames)?, frames, 1)
    }

    pub fn forward_frame_scores_with_timing(
        &self,
        feats: FeatureTensor<B>,
        caches: &mut [Tensor<B, 2>],
        timing: &mut FsmnForwardTiming,
    ) -> Result<Vec<Vec<f32>>> {
        let [frames, feat_dim] = feats.dims();
        if feat_dim != FEAT_DIM {
            bail!("FSMN VAD expects feature dim {FEAT_DIM}, got {feat_dim}");
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let start = Instant::now();
        let mut x = feats;
        timing.input_tensor_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        x = self.in_linear1.forward(x, false);
        timing.in_linear1_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        x = self.in_linear2.forward(x, true);
        timing.in_linear2_seconds += start.elapsed().as_secs_f64();

        for (idx, (block, cache)) in self.blocks.iter().zip(caches.iter()).enumerate() {
            x = block.forward_with_timing(x, frames, cache, idx, timing, false)?;
        }

        let start = Instant::now();
        x = self.out_linear1.forward(x, false);
        timing.out_linear1_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        x = self.out_linear2.forward(x, false);
        timing.out_linear2_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        x = silence_posterior(x, frames)?;
        timing.softmax_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let rows = tensor_rows(x, frames, 1);
        timing.output_tensor_seconds += start.elapsed().as_secs_f64();
        rows
    }
}

impl<B: BurnBackend> BurnFsmnBlock<B> {
    fn forward(
        &self,
        input: Tensor<B, 2>,
        frames: usize,
        cache: &mut Tensor<B, 2>,
        update_cache: bool,
    ) -> Result<Tensor<B, 2>> {
        let projected = self.linear.forward(input, false);
        let memory = fsmn_memory(
            projected.clone(),
            cache,
            self.conv_left_weight.clone(),
            frames,
        )?;
        if update_cache {
            *cache = next_cache(&projected, cache)?;
        }
        Ok(self.affine.forward(memory, true))
    }

    fn forward_with_timing(
        &self,
        input: Tensor<B, 2>,
        frames: usize,
        cache: &Tensor<B, 2>,
        idx: usize,
        timing: &mut FsmnForwardTiming,
        update_cache: bool,
    ) -> Result<Tensor<B, 2>> {
        let start = Instant::now();
        let projected = self.linear.forward(input, false);
        timing.block_linear_seconds[idx] += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let memory = fsmn_memory(
            projected.clone(),
            cache,
            self.conv_left_weight.clone(),
            frames,
        )?;
        timing.block_memory_seconds[idx] += start.elapsed().as_secs_f64();
        let _ = update_cache;

        let start = Instant::now();
        let out = self.affine.forward(memory, true);
        timing.block_affine_seconds[idx] += start.elapsed().as_secs_f64();
        Ok(out)
    }
}

impl<B: BurnBackend> BurnLinear<B> {
    fn load(
        reader: &PytorchReader,
        device: &B::Device,
        prefix: &str,
        has_bias: bool,
    ) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let weight_shape = snapshot_shape(reader, &weight_key)?;
        if weight_shape.len() != 2 {
            bail!("{weight_key} must be 2D, got {weight_shape:?}");
        }
        let out_dim = weight_shape[0];
        let in_dim = weight_shape[1];
        let weight = load_vec(reader, &weight_key)?;
        let mut weight_t = vec![0.0; weight.len()];
        for out_idx in 0..out_dim {
            for in_idx in 0..in_dim {
                weight_t[in_idx * out_dim + out_idx] = weight[out_idx * in_dim + in_idx];
            }
        }
        let bias_vec = if has_bias {
            let bias_key = format!("{prefix}.bias");
            Some(load_vec(reader, &bias_key)?)
        } else {
            None
        };
        Ok(Self {
            inner: Linear {
                weight: Param::from_tensor(Tensor::<B, 2>::from_data(
                    TensorData::new(weight_t, [in_dim, out_dim]),
                    device,
                )),
                bias: bias_vec.map(|bias| {
                    Param::from_tensor(Tensor::<B, 1>::from_data(
                        TensorData::new(bias, [out_dim]),
                        device,
                    ))
                }),
            },
        })
    }

    fn forward(&self, input: Tensor<B, 2>, relu: bool) -> Tensor<B, 2> {
        let mut out = self.inner.forward(input);
        if relu {
            out = burn::tensor::activation::relu(out);
        }
        out
    }
}
