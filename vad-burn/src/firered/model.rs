use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::backend::flex::FlexDevice;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Tensor, TensorData};
use burn_store::pytorch::PytorchReader;
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};

use crate::{DurationMs, TimeRange, VadOptions, VadSegment, Waveform};

type Backend = burn::backend::Flex;

const SAMPLE_RATE: u32 = 16_000;
const FRAME_SHIFT_MS: u64 = 10;
const FRAME_LENGTH_MS: u64 = 25;
const INPUT_DIM: usize = 80;
const PROJ_DIM: usize = 128;
const BLOCKS: usize = 8;
const FSMN_ORDER: usize = 20;
const DEFAULT_THRESHOLD: f32 = 0.4;
const DEFAULT_MAX_SPEECH_FRAME: usize = 2000;
const STREAM_DEFAULT_SMOOTH_WINDOW_SIZE: usize = 5;
const STREAM_DEFAULT_PAD_START_FRAME: usize = 5;

pub const DEFAULT_FIRERED_MODELSCOPE_REPO_ID: &str = "xukaituo/FireRedVAD";
pub const DEFAULT_FIRERED_MODELSCOPE_REVISION: &str = "master";

#[derive(Debug, Clone, Copy, Default)]
pub struct FireRedVadTiming {
    pub frontend_seconds: f64,
    pub forward_seconds: f64,
    pub postprocess_seconds: f64,
    pub frames: usize,
}

#[derive(Debug, Clone)]
pub struct FireRedVadDetection {
    pub segments: Vec<VadSegment>,
    pub frame_scores: Vec<f32>,
    pub timing: FireRedVadTiming,
}

pub struct FireRedVadModel {
    offline_weights: Arc<FireRedVadWeights>,
    stream_weights: Arc<FireRedVadWeights>,
    model_dir: PathBuf,
    offline_model_dir: PathBuf,
    stream_model_dir: PathBuf,
}

pub struct FireRedVadStream {
    feature_stream: FireRedFeatureStream,
    weights: Arc<FireRedVadWeights>,
    options: VadOptions,
    caches: Vec<Vec<f32>>,
    postprocessor: FireRedStreamVadPostprocessor,
    frame_scores: Vec<f32>,
}

impl FireRedVadModel {
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let model_dirs = FireRedModelDirs::from_path(&model_dir)?;
        let offline_weights = Arc::new(FireRedVadWeights::load(&model_dirs.offline)?);
        let stream_weights = Arc::new(FireRedVadWeights::load(&model_dirs.stream)?);
        Ok(Self {
            offline_weights,
            stream_weights,
            model_dir,
            offline_model_dir: model_dirs.offline,
            stream_model_dir: model_dirs.stream,
        })
    }

    pub fn from_modelscope() -> Result<Self> {
        Self::from_modelscope_revision(
            DEFAULT_FIRERED_MODELSCOPE_REPO_ID,
            DEFAULT_FIRERED_MODELSCOPE_REVISION,
        )
    }

    pub fn from_modelscope_revision(repo_id: &str, revision: &str) -> Result<Self> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(Self::from_modelscope_revision_async(repo_id, revision))
    }

    pub async fn from_modelscope_async() -> Result<Self> {
        Self::from_modelscope_revision_async(
            DEFAULT_FIRERED_MODELSCOPE_REPO_ID,
            DEFAULT_FIRERED_MODELSCOPE_REVISION,
        )
        .await
    }

    pub async fn from_modelscope_revision_async(repo_id: &str, revision: &str) -> Result<Self> {
        let cache_dir = modelhub::modelscope::cache_dir();
        modelhub::modelscope::download_model_revision(repo_id, revision, &cache_dir).await?;
        Self::from_pretrained(modelscope_snapshot_dir(&cache_dir, repo_id, revision))
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn offline_model_dir(&self) -> &Path {
        &self.offline_model_dir
    }

    pub fn stream_model_dir(&self) -> &Path {
        &self.stream_model_dir
    }

    pub fn new_stream(&self, options: VadOptions) -> FireRedVadStream {
        FireRedVadStream {
            feature_stream: self.stream_weights.frontend.new_stream(),
            weights: Arc::clone(&self.stream_weights),
            caches: self.stream_weights.zero_caches(),
            postprocessor: FireRedStreamVadPostprocessor::from_options(&options),
            options,
            frame_scores: Vec::new(),
        }
    }

    pub fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        validate_waveform(waveform)?;

        let feats = self.offline_weights.frontend.extract(&waveform.samples)?;
        let [frames, feat_dim] = feats.dims();
        if feat_dim != INPUT_DIM {
            bail!("FireRedVAD expects feature dim {INPUT_DIM}, got {feat_dim}");
        }
        if frames == 0 {
            return Ok(Vec::new());
        }

        let probs = self.offline_weights.forward_probs(feats)?;
        Ok(FireRedVadPostprocessor::from_options(options).process_to_segments(&probs, waveform))
    }

    pub fn detect_with_timing(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<FireRedVadDetection> {
        validate_waveform(waveform)?;

        let mut timing = FireRedVadTiming::default();
        let frontend_start = Instant::now();
        let feats = self.offline_weights.frontend.extract(&waveform.samples)?;
        timing.frontend_seconds = frontend_start.elapsed().as_secs_f64();
        let [frames, feat_dim] = feats.dims();
        timing.frames = frames;
        if feat_dim != INPUT_DIM {
            bail!("FireRedVAD expects feature dim {INPUT_DIM}, got {feat_dim}");
        }
        if frames == 0 {
            return Ok(FireRedVadDetection {
                segments: Vec::new(),
                frame_scores: Vec::new(),
                timing,
            });
        }

        let forward_start = Instant::now();
        let probs = self.offline_weights.forward_probs(feats)?;
        timing.forward_seconds = forward_start.elapsed().as_secs_f64();

        let post_start = Instant::now();
        let segments =
            FireRedVadPostprocessor::from_options(options).process_to_segments(&probs, waveform);
        timing.postprocess_seconds = post_start.elapsed().as_secs_f64();

        Ok(FireRedVadDetection {
            segments,
            frame_scores: probs,
            timing,
        })
    }
}

impl FireRedVadStream {
    pub fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<VadSegment>> {
        let waveform = Waveform::new(samples.to_vec(), sample_rate);
        validate_waveform(&waveform)?;
        let feats = self.feature_stream.push(&waveform.samples)?;
        let [frames, feat_dim] = feats.dims();
        if feat_dim != INPUT_DIM {
            bail!("FireRedVAD expects feature dim {INPUT_DIM}, got {feat_dim}");
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let probs = self
            .weights
            .forward_probs_streaming(feats, &mut self.caches)?;
        self.frame_scores.extend_from_slice(&probs);
        Ok(self.postprocessor.process_probs(&probs))
    }

    pub fn finish(&mut self) -> Result<Vec<VadSegment>> {
        let feats = self.feature_stream.finish()?;
        let [frames, feat_dim] = feats.dims();
        if feat_dim != INPUT_DIM {
            bail!("FireRedVAD expects feature dim {INPUT_DIM}, got {feat_dim}");
        }
        let mut segments = if frames == 0 {
            Vec::new()
        } else {
            let probs = self
                .weights
                .forward_probs_streaming(feats, &mut self.caches)?;
            self.frame_scores.extend_from_slice(&probs);
            self.postprocessor.process_probs(&probs)
        };
        segments.extend(self.postprocessor.finish());
        self.reset();
        Ok(segments)
    }

    pub fn reset(&mut self) {
        self.feature_stream = self.weights.frontend.new_stream();
        self.caches = self.weights.zero_caches();
        self.postprocessor = FireRedStreamVadPostprocessor::from_options(&self.options);
        self.frame_scores.clear();
    }

    pub fn frame_scores(&self) -> &[f32] {
        &self.frame_scores
    }

    pub fn options(&self) -> &VadOptions {
        &self.options
    }
}

struct FireRedVadWeights {
    frontend: FireRedFrontend,
    fc1: BurnLinear,
    fc2: BurnLinear,
    fsmn1: FireRedFsmn,
    blocks: Vec<FireRedDfsmnBlock>,
    dnn: BurnLinear,
    out: BurnLinear,
}

struct FireRedModelDirs {
    offline: PathBuf,
    stream: PathBuf,
}

impl FireRedModelDirs {
    fn from_path(model_dir: &Path) -> Result<Self> {
        if model_dir.join("VAD").is_dir() || model_dir.join("Stream-VAD").is_dir() {
            let offline = model_dir.join("VAD");
            let stream = model_dir.join("Stream-VAD");
            validate_model_dir(&offline)?;
            validate_model_dir(&stream)?;
            return Ok(Self { offline, stream });
        }

        validate_model_dir(model_dir)?;
        Ok(Self {
            offline: model_dir.to_path_buf(),
            stream: model_dir.to_path_buf(),
        })
    }
}

struct FireRedDfsmnBlock {
    fc1: BurnLinear,
    fc2: BurnLinear,
    fsmn: FireRedFsmn,
}

struct FireRedFsmn {
    lookback_tensor: Tensor<Backend, 3>,
    lookahead_tensor: Option<Tensor<Backend, 3>>,
    lookback_weight: Vec<f32>,
}

struct BurnLinear {
    weight_t: Tensor<Backend, 2>,
    bias: Option<Tensor<Backend, 1>>,
    in_dim: usize,
    out_dim: usize,
}

impl FireRedVadWeights {
    fn load(model_dir: &Path) -> Result<Self> {
        let device = FlexDevice;
        let frontend = FireRedFrontend::new(&model_dir.join("cmvn.ark"))?;
        let reader =
            PytorchReader::with_top_level_key(model_dir.join("model.pth.tar"), "model_state_dict")
                .with_context(|| {
                    format!(
                        "failed to load FireRedVAD model from {}",
                        model_dir.display()
                    )
                })?;
        let prefix = "dfsmn";
        let fc1 = BurnLinear::load(&reader, &device, &format!("{prefix}.fc1.0"), true)?;
        let fc2 = BurnLinear::load(&reader, &device, &format!("{prefix}.fc2.0"), true)?;
        let fsmn1 = FireRedFsmn::load(&reader, &device, &format!("{prefix}.fsmn1"))?;
        let mut blocks = Vec::with_capacity(BLOCKS - 1);
        for idx in 0..BLOCKS - 1 {
            blocks.push(FireRedDfsmnBlock {
                fc1: BurnLinear::load(
                    &reader,
                    &device,
                    &format!("{prefix}.fsmns.{idx}.fc1.0"),
                    true,
                )?,
                fc2: BurnLinear::load(
                    &reader,
                    &device,
                    &format!("{prefix}.fsmns.{idx}.fc2"),
                    false,
                )?,
                fsmn: FireRedFsmn::load(&reader, &device, &format!("{prefix}.fsmns.{idx}.fsmn"))?,
            });
        }
        Ok(Self {
            frontend,
            fc1,
            fc2,
            fsmn1,
            blocks,
            dnn: BurnLinear::load(&reader, &device, &format!("{prefix}.dnns.0"), true)?,
            out: BurnLinear::load(&reader, &device, "out", true)?,
        })
    }

    fn forward_probs(&self, feats: Tensor<Backend, 2>) -> Result<Vec<f32>> {
        let [frames, dim] = feats.dims();
        if dim != INPUT_DIM {
            bail!("FireRedVAD expects feature dim {INPUT_DIM}, got {dim}");
        }
        let mut x = self.fc1.forward(feats, true)?;
        x = self.fc2.forward(x, true)?;
        x = self.fsmn1.forward(x)?;
        for block in &self.blocks {
            x = block.forward(x)?;
        }
        x = self.dnn.forward(x, true)?;
        x = self.out.forward(x, false)?;
        let probs = burn::tensor::activation::sigmoid(x);
        let data = probs
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("FireRedVAD output tensor data");
        if data.len() != frames {
            bail!(
                "FireRedVAD expected {frames} probabilities, got {}",
                data.len()
            );
        }
        Ok(data)
    }

    fn forward_probs_streaming(
        &self,
        feats: Tensor<Backend, 2>,
        caches: &mut [Vec<f32>],
    ) -> Result<Vec<f32>> {
        if caches.len() != self.fsmn_count() {
            bail!(
                "FireRedVAD expected {} FSMN caches, got {}",
                self.fsmn_count(),
                caches.len()
            );
        }
        let [frames, dim] = feats.dims();
        if dim != INPUT_DIM {
            bail!("FireRedVAD expects feature dim {INPUT_DIM}, got {dim}");
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let mut x = self.fc1.forward(feats, true)?;
        x = self.fc2.forward(x, true)?;
        x = self.fsmn1.forward_streaming(x, &mut caches[0])?;
        for (idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_streaming(x, &mut caches[idx + 1])?;
        }
        x = self.dnn.forward(x, true)?;
        x = self.out.forward(x, false)?;
        let probs = burn::tensor::activation::sigmoid(x);
        let data = probs
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("FireRedVAD streaming output tensor data");
        if data.len() != frames {
            bail!(
                "FireRedVAD expected {frames} streaming probabilities, got {}",
                data.len()
            );
        }
        Ok(data)
    }

    fn zero_caches(&self) -> Vec<Vec<f32>> {
        (0..self.fsmn_count())
            .map(|_| vec![0.0; PROJ_DIM * (FSMN_ORDER - 1)])
            .collect()
    }

    fn fsmn_count(&self) -> usize {
        1 + self.blocks.len()
    }
}

impl FireRedDfsmnBlock {
    fn forward(&self, input: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>> {
        let residual = input.clone();
        let mut x = self.fc1.forward(input, true)?;
        x = self.fc2.forward(x, false)?;
        Ok(self.fsmn.forward(x)? + residual)
    }

    fn forward_streaming(
        &self,
        input: Tensor<Backend, 2>,
        cache: &mut Vec<f32>,
    ) -> Result<Tensor<Backend, 2>> {
        let residual = input.clone();
        let mut x = self.fc1.forward(input, true)?;
        x = self.fc2.forward(x, false)?;
        Ok(self.fsmn.forward_streaming(x, cache)? + residual)
    }
}

impl FireRedFsmn {
    fn load(reader: &PytorchReader, device: &FlexDevice, prefix: &str) -> Result<Self> {
        let lookback_key = format!("{prefix}.lookback_filter.weight");
        let lookahead_key = format!("{prefix}.lookahead_filter.weight");
        let (lookback_tensor, lookback_weight) = load_fsmn_weight(reader, device, &lookback_key)?;
        let lookahead_tensor = load_optional_fsmn_weight(reader, device, &lookahead_key)?;
        Ok(Self {
            lookback_tensor,
            lookahead_tensor,
            lookback_weight,
        })
    }

    fn forward(&self, input: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>> {
        let [frames, proj_dim] = input.dims();
        if proj_dim != PROJ_DIM {
            bail!("unexpected FireRed FSMN input shape: {:?}", input.dims());
        }
        let x = input.clone().swap_dims(0, 1).reshape([1, PROJ_DIM, frames]);
        let lookback = conv1d(
            x.clone(),
            self.lookback_tensor.clone(),
            None,
            ConvOptions::new([1], [FSMN_ORDER - 1], [1], PROJ_DIM),
        );
        let lookback = lookback
            .slice([0..1, 0..PROJ_DIM, 0..frames])
            .reshape([PROJ_DIM, frames])
            .swap_dims(0, 1);
        let mut memory = input + lookback;
        if frames > 1
            && let Some(lookahead_tensor) = &self.lookahead_tensor
        {
            let lookahead = conv1d(
                x,
                lookahead_tensor.clone(),
                None,
                ConvOptions::new([1], [FSMN_ORDER], [1], PROJ_DIM),
            )
            .slice([0..1, 0..PROJ_DIM, FSMN_ORDER + 1..FSMN_ORDER + 1 + frames])
            .reshape([PROJ_DIM, frames])
            .swap_dims(0, 1);
            memory = memory + lookahead;
        }
        Ok(memory)
    }

    fn forward_streaming(
        &self,
        input: Tensor<Backend, 2>,
        cache: &mut Vec<f32>,
    ) -> Result<Tensor<Backend, 2>> {
        let [frames, proj_dim] = input.dims();
        if proj_dim != PROJ_DIM {
            bail!(
                "unexpected FireRed streaming FSMN input shape: {:?}",
                input.dims()
            );
        }
        let cache_frames = FSMN_ORDER - 1;
        if cache.len() != cache_frames * PROJ_DIM {
            bail!(
                "unexpected FireRed streaming cache length: {}, expected {}",
                cache.len(),
                cache_frames * PROJ_DIM
            );
        }
        let input_data = input.into_data().convert::<f32>().into_vec::<f32>()?;
        let mut output = input_data.clone();
        apply_fsmn_streaming_lookback(
            &input_data,
            &mut output,
            frames,
            cache,
            &self.lookback_weight,
        );
        update_fsmn_cache(&input_data, frames, cache);
        Ok(Tensor::<Backend, 2>::from_data(
            TensorData::new(output, [frames, PROJ_DIM]),
            &FlexDevice,
        ))
    }
}

fn apply_fsmn_streaming_lookback(
    input: &[f32],
    output: &mut [f32],
    frames: usize,
    cache: &[f32],
    weight: &[f32],
) {
    let cache_frames = FSMN_ORDER - 1;
    for frame in 0..frames {
        let extended_frame = cache_frames + frame;
        let output_base = frame * PROJ_DIM;
        for kernel in 0..FSMN_ORDER {
            let source_frame = extended_frame + kernel - (FSMN_ORDER - 1);
            let weight_base = kernel * PROJ_DIM;
            if source_frame < cache_frames {
                let cache_base = source_frame * PROJ_DIM;
                for channel in 0..PROJ_DIM {
                    output[output_base + channel] +=
                        weight[weight_base + channel] * cache[cache_base + channel];
                }
            } else {
                let input_base = (source_frame - cache_frames) * PROJ_DIM;
                for channel in 0..PROJ_DIM {
                    output[output_base + channel] +=
                        weight[weight_base + channel] * input[input_base + channel];
                }
            }
        }
    }
}

fn update_fsmn_cache(input: &[f32], frames: usize, cache: &mut Vec<f32>) {
    let cache_frames = FSMN_ORDER - 1;
    if frames >= cache_frames {
        let start = (frames - cache_frames) * PROJ_DIM;
        cache.copy_from_slice(&input[start..start + cache_frames * PROJ_DIM]);
        return;
    }

    let keep_old_frames = cache_frames - frames;
    let mut next = Vec::with_capacity(cache_frames * PROJ_DIM);
    next.extend_from_slice(&cache[frames * PROJ_DIM..]);
    next.extend_from_slice(input);
    debug_assert_eq!(next.len(), keep_old_frames * PROJ_DIM + input.len());
    *cache = next;
}

impl BurnLinear {
    fn load(
        reader: &PytorchReader,
        device: &FlexDevice,
        prefix: &str,
        has_bias: bool,
    ) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let weight_shape = tensor_shape(reader, &weight_key)?;
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
        let bias = if has_bias {
            let bias = load_vec(reader, &format!("{prefix}.bias"))?;
            Some(Tensor::<Backend, 1>::from_data(
                TensorData::new(bias, [out_dim]),
                device,
            ))
        } else {
            None
        };
        Ok(Self {
            weight_t: Tensor::<Backend, 2>::from_data(
                TensorData::new(weight_t, [in_dim, out_dim]),
                device,
            ),
            bias,
            in_dim,
            out_dim,
        })
    }

    fn forward(&self, input: Tensor<Backend, 2>, relu: bool) -> Result<Tensor<Backend, 2>> {
        let [rows, in_dim] = input.dims();
        if in_dim != self.in_dim {
            bail!(
                "FireRedVAD linear expects input dim {}, got {in_dim}",
                self.in_dim
            );
        }
        let mut out = input.matmul(self.weight_t.clone());
        if let Some(bias) = &self.bias {
            out = out + bias.clone().unsqueeze_dim::<2>(0);
        }
        let [actual_rows, out_dim] = out.dims();
        if actual_rows != rows || out_dim != self.out_dim {
            bail!(
                "unexpected FireRedVAD linear output shape: {:?}",
                out.dims()
            );
        }
        if relu {
            Ok(burn::tensor::activation::relu(out))
        } else {
            Ok(out)
        }
    }
}

#[derive(Clone)]
struct FireRedFrontend {
    means: Vec<f32>,
    inverse_std: Vec<f32>,
    device: FlexDevice,
}

struct FireRedFeatureStream {
    frontend: FireRedFrontend,
    fbank: OnlineFbank,
    emitted_frames: usize,
}

impl FireRedFrontend {
    fn new(cmvn_path: &Path) -> Result<Self> {
        let (means, inverse_std) = read_kaldi_binary_cmvn(cmvn_path)?;
        if means.len() != INPUT_DIM || inverse_std.len() != INPUT_DIM {
            bail!(
                "FireRedVAD CMVN expects {INPUT_DIM} dims, got means={} vars={}",
                means.len(),
                inverse_std.len()
            );
        }
        Ok(Self {
            means,
            inverse_std,
            device: FlexDevice,
        })
    }

    fn extract(&self, samples: &[f32]) -> Result<Tensor<Backend, 2>> {
        let waveform = samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        let mut fbank = OnlineFbank::new(Self::fbank_options());
        fbank.accept_waveform(SAMPLE_RATE as f32, &waveform);
        let frames = fbank.num_ready_frames() as usize;
        self.collect_frames(&fbank, 0, frames)
    }

    fn new_stream(&self) -> FireRedFeatureStream {
        FireRedFeatureStream {
            frontend: self.clone(),
            fbank: OnlineFbank::new(Self::fbank_options()),
            emitted_frames: 0,
        }
    }

    fn collect_frames(
        &self,
        fbank: &OnlineFbank,
        start_frame: usize,
        end_frame: usize,
    ) -> Result<Tensor<Backend, 2>> {
        if end_frame <= start_frame {
            return Ok(Tensor::<Backend, 2>::zeros([0, INPUT_DIM], &self.device));
        }
        let mut out = Vec::with_capacity((end_frame - start_frame) * INPUT_DIM);
        for frame_idx in start_frame..end_frame {
            let frame = fbank
                .get_frame(frame_idx as i32)
                .ok_or_else(|| anyhow::anyhow!("missing FireRed fbank frame {frame_idx}"))?;
            for (dim, value) in frame.iter().enumerate() {
                out.push((*value - self.means[dim]) * self.inverse_std[dim]);
            }
        }
        Ok(Tensor::<Backend, 2>::from_data(
            TensorData::new(out, [end_frame - start_frame, INPUT_DIM]),
            &self.device,
        ))
    }

    fn fbank_options() -> FbankOptions {
        FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: SAMPLE_RATE as f32,
                dither: 0.0,
                frame_shift_ms: FRAME_SHIFT_MS as f32,
                frame_length_ms: FRAME_LENGTH_MS as f32,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: INPUT_DIM as i32,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        }
    }
}

impl FireRedFeatureStream {
    fn push(&mut self, samples: &[f32]) -> Result<Tensor<Backend, 2>> {
        let waveform = samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        self.fbank.accept_waveform(SAMPLE_RATE as f32, &waveform);
        self.collect_ready_frames()
    }

    fn finish(&mut self) -> Result<Tensor<Backend, 2>> {
        self.fbank.input_finished();
        self.collect_ready_frames()
    }

    fn collect_ready_frames(&mut self) -> Result<Tensor<Backend, 2>> {
        let frames = self.fbank.num_ready_frames() as usize;
        let feats = self
            .frontend
            .collect_frames(&self.fbank, self.emitted_frames, frames)?;
        self.emitted_frames = frames;
        Ok(feats)
    }
}

#[derive(Debug, Clone)]
struct FireRedVadPostprocessor {
    smooth_window_size: usize,
    prob_threshold: f32,
    min_speech_ms: u64,
    min_speech_frame: usize,
    max_speech_frame: usize,
    min_silence_frame: usize,
    merge_silence_frame: usize,
    extend_speech_frame: usize,
}

impl FireRedVadPostprocessor {
    fn from_options(options: &VadOptions) -> Self {
        Self {
            smooth_window_size: 5,
            prob_threshold: if options.threshold > 0.0 {
                options.threshold
            } else {
                DEFAULT_THRESHOLD
            },
            min_speech_ms: options.min_speech_ms,
            min_speech_frame: ms_to_frame_count(options.min_speech_ms),
            max_speech_frame: if options.max_segment_ms > 0 {
                ms_to_frame_count(options.max_segment_ms)
            } else {
                DEFAULT_MAX_SPEECH_FRAME
            },
            min_silence_frame: ms_to_frame_count(options.min_silence_ms),
            merge_silence_frame: 0,
            extend_speech_frame: 0,
        }
    }

    fn process_to_segments(&self, probs: &[f32], waveform: &Waveform) -> Vec<VadSegment> {
        if probs.is_empty() {
            return Vec::new();
        }
        let smoothed = self.smooth_prob(probs);
        let binary = smoothed
            .iter()
            .map(|prob| usize::from(*prob >= self.prob_threshold))
            .collect::<Vec<_>>();
        let decisions = self.smooth_preds_with_state_machine(&binary);
        let decisions = self.fix_smooth_window_start(&decisions);
        let decisions = self.merge_short_silence_segments(&decisions);
        let decisions = self.extend_speech_segments(&decisions);
        let decisions = self.split_long_speech_segments(&decisions, probs);
        self.decision_to_segments(&decisions, waveform.duration_seconds())
    }

    fn smooth_prob(&self, probs: &[f32]) -> Vec<f32> {
        if self.smooth_window_size <= 1 {
            return probs.to_vec();
        }
        let mut out = vec![0.0; probs.len()];
        let mut sum = 0.0;
        for idx in 0..probs.len() {
            sum += probs[idx];
            if idx >= self.smooth_window_size {
                sum -= probs[idx - self.smooth_window_size];
            }
            out[idx] = sum / (idx + 1).min(self.smooth_window_size) as f32;
        }
        out
    }

    fn smooth_preds_with_state_machine(&self, binary: &[usize]) -> Vec<usize> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum State {
            Silence,
            PossibleSpeech,
            Speech,
            PossibleSilence,
        }

        let mut decisions = vec![0; binary.len()];
        let mut state = State::Silence;
        let mut speech_start = None;
        let mut silence_start = None;
        for (frame, is_speech) in binary.iter().copied().enumerate() {
            match state {
                State::Silence if is_speech == 1 => {
                    state = State::PossibleSpeech;
                    speech_start = Some(frame);
                }
                State::PossibleSpeech if is_speech == 1 => {
                    if frame - speech_start.unwrap_or(frame) >= self.min_speech_frame {
                        let start = speech_start.unwrap_or(frame);
                        state = State::Speech;
                        decisions[start..frame].fill(1);
                    }
                }
                State::PossibleSpeech => {
                    state = State::Silence;
                    speech_start = None;
                }
                State::Speech if is_speech == 0 => {
                    state = State::PossibleSilence;
                    silence_start = Some(frame);
                }
                State::PossibleSilence if is_speech == 0 => {
                    if frame - silence_start.unwrap_or(frame) >= self.min_silence_frame {
                        state = State::Silence;
                        speech_start = None;
                    }
                }
                State::PossibleSilence => {
                    state = State::Speech;
                    silence_start = None;
                }
                _ => {}
            }
            decisions[frame] = usize::from(matches!(state, State::Speech | State::PossibleSilence));
        }
        decisions
    }

    fn fix_smooth_window_start(&self, decisions: &[usize]) -> Vec<usize> {
        let mut out = decisions.to_vec();
        for frame in 1..decisions.len() {
            if decisions[frame - 1] == 0 && decisions[frame] == 1 {
                out[frame.saturating_sub(self.smooth_window_size)..frame].fill(1);
            }
        }
        out
    }

    fn merge_short_silence_segments(&self, decisions: &[usize]) -> Vec<usize> {
        if self.merge_silence_frame == 0 {
            return decisions.to_vec();
        }
        let mut out = decisions.to_vec();
        let mut silence_start = None;
        for frame in 1..decisions.len() {
            if decisions[frame - 1] == 1 && decisions[frame] == 0 && silence_start.is_none() {
                silence_start = Some(frame);
            } else if decisions[frame - 1] == 0
                && decisions[frame] == 1
                && let Some(start) = silence_start.take()
            {
                let silence_frames = frame - start;
                if silence_frames < self.merge_silence_frame {
                    out[start..frame].fill(1);
                }
            }
        }
        out
    }

    fn extend_speech_segments(&self, decisions: &[usize]) -> Vec<usize> {
        if self.extend_speech_frame == 0 {
            return decisions.to_vec();
        }
        let mut out = decisions.to_vec();
        for (frame, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 {
                let start = frame.saturating_sub(self.extend_speech_frame);
                let end = (frame + self.extend_speech_frame + 1).min(decisions.len());
                out[start..end].fill(1);
            }
        }
        out
    }

    fn split_long_speech_segments(&self, decisions: &[usize], probs: &[f32]) -> Vec<usize> {
        let mut out = decisions.to_vec();
        let mut speech_start = None;
        for (frame, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 && speech_start.is_none() {
                speech_start = Some(frame);
            } else if decision == 0
                && let Some(start) = speech_start.take()
            {
                self.split_long_speech_segment(&mut out, probs, start, frame);
            }
        }
        if let Some(start) = speech_start {
            self.split_long_speech_segment(&mut out, probs, start, decisions.len());
        }
        out
    }

    fn split_long_speech_segment(
        &self,
        decisions: &mut [usize],
        probs: &[f32],
        start: usize,
        end: usize,
    ) {
        let end = end.min(probs.len()).min(decisions.len());
        if start < end && end.saturating_sub(start) > self.max_speech_frame {
            for split in self.find_split_points(&probs[start..end]) {
                decisions[start + split] = 0;
            }
        }
    }

    fn find_split_points(&self, probs: &[f32]) -> Vec<usize> {
        let mut splits = Vec::new();
        let mut start = 0usize;
        while probs.len().saturating_sub(start) > self.max_speech_frame {
            let window_start = start + self.max_speech_frame / 2;
            let window_end = start + self.max_speech_frame;
            let min_idx = probs[window_start..window_end]
                .iter()
                .enumerate()
                .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                .map(|(idx, _)| window_start + idx)
                .unwrap_or(window_start);
            splits.push(min_idx);
            start = min_idx + 1;
        }
        splits
    }

    fn decision_to_segments(&self, decisions: &[usize], wav_dur_seconds: f64) -> Vec<VadSegment> {
        self.decision_to_seconds(decisions, wav_dur_seconds)
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(
                    DurationMs((start * 1000.0).round() as u64),
                    DurationMs((end * 1000.0).round() as u64),
                ),
                probability: self.prob_threshold,
            })
            .filter(|segment| {
                segment.range.end.0.saturating_sub(segment.range.start.0) >= self.min_speech_ms
            })
            .collect()
    }

    fn decision_to_seconds(&self, decisions: &[usize], wav_dur_seconds: f64) -> Vec<(f64, f64)> {
        let mut segments = Vec::new();
        let mut speech_start = None;
        for (frame, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 && speech_start.is_none() {
                speech_start = Some(frame);
            } else if decision == 0 && speech_start.is_some() {
                let start = speech_start.take().expect("speech start");
                segments.push((start as f64 * 0.010, frame as f64 * 0.010));
            }
        }
        if let Some(start) = speech_start {
            let end = (decisions.len() as f64 * 0.010 + 0.025).min(wav_dur_seconds);
            segments.push((start as f64 * 0.010, end));
        }
        segments
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FireRedStreamVadState {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

#[derive(Debug, Clone)]
struct FireRedStreamVadPostprocessor {
    smooth_window_size: usize,
    speech_threshold: f32,
    pad_start_frame: usize,
    min_speech_frame: usize,
    max_speech_frame: usize,
    min_silence_frame: usize,
    frame_cnt: usize,
    smooth_window: VecDeque<f32>,
    smooth_window_sum: f32,
    state: FireRedStreamVadState,
    speech_cnt: usize,
    silence_cnt: usize,
    hit_max_speech: bool,
    last_speech_start_frame: Option<usize>,
    last_speech_end_frame: Option<usize>,
}

impl FireRedStreamVadPostprocessor {
    fn from_options(options: &VadOptions) -> Self {
        let smooth_window_size = STREAM_DEFAULT_SMOOTH_WINDOW_SIZE;
        Self {
            smooth_window_size,
            speech_threshold: if options.threshold > 0.0 {
                options.threshold
            } else {
                0.5
            },
            pad_start_frame: STREAM_DEFAULT_PAD_START_FRAME,
            min_speech_frame: ms_to_frame_count(options.min_speech_ms),
            max_speech_frame: if options.max_segment_ms > 0 {
                ms_to_frame_count(options.max_segment_ms)
            } else {
                DEFAULT_MAX_SPEECH_FRAME
            },
            min_silence_frame: ms_to_frame_count(options.min_silence_ms),
            frame_cnt: 0,
            smooth_window: VecDeque::new(),
            smooth_window_sum: 0.0,
            state: FireRedStreamVadState::Silence,
            speech_cnt: 0,
            silence_cnt: 0,
            hit_max_speech: false,
            last_speech_start_frame: None,
            last_speech_end_frame: None,
        }
    }

    fn process_probs(&mut self, probs: &[f32]) -> Vec<VadSegment> {
        probs
            .iter()
            .filter_map(|prob| self.process_one_frame(*prob))
            .collect()
    }

    fn process_one_frame(&mut self, raw_prob: f32) -> Option<VadSegment> {
        self.frame_cnt += 1;
        let smoothed_prob = self.smooth_prob(raw_prob.clamp(0.0, 1.0));
        let is_speech = smoothed_prob >= self.speech_threshold;
        self.state_transition(is_speech)
    }

    fn finish(&mut self) -> Vec<VadSegment> {
        self.last_speech_start_frame
            .take()
            .and_then(|start| self.segment_from_frames(start, self.frame_cnt))
            .into_iter()
            .collect()
    }

    fn smooth_prob(&mut self, prob: f32) -> f32 {
        if self.smooth_window_size <= 1 {
            return prob;
        }
        self.smooth_window.push_back(prob);
        self.smooth_window_sum += prob;
        if self.smooth_window.len() > self.smooth_window_size
            && let Some(left) = self.smooth_window.pop_front()
        {
            self.smooth_window_sum -= left;
        }
        self.smooth_window_sum / self.smooth_window.len() as f32
    }

    fn state_transition(&mut self, is_speech: bool) -> Option<VadSegment> {
        if self.hit_max_speech {
            self.last_speech_start_frame = Some(self.frame_cnt);
            self.hit_max_speech = false;
        }

        match self.state {
            FireRedStreamVadState::Silence => {
                if is_speech {
                    self.state = FireRedStreamVadState::PossibleSpeech;
                    self.speech_cnt += 1;
                } else {
                    self.silence_cnt += 1;
                    self.speech_cnt = 0;
                }
            }
            FireRedStreamVadState::PossibleSpeech => {
                if is_speech {
                    self.speech_cnt += 1;
                    if self.speech_cnt >= self.min_speech_frame {
                        self.state = FireRedStreamVadState::Speech;
                        let previous_end = self.last_speech_end_frame.unwrap_or(0);
                        let start = self
                            .frame_cnt
                            .saturating_sub(self.speech_cnt)
                            .saturating_add(1)
                            .saturating_sub(self.pad_start_frame)
                            .max(1)
                            .max(previous_end.saturating_add(1));
                        self.last_speech_start_frame = Some(start);
                        self.silence_cnt = 0;
                    }
                } else {
                    self.state = FireRedStreamVadState::Silence;
                    self.silence_cnt = 1;
                    self.speech_cnt = 0;
                }
            }
            FireRedStreamVadState::Speech => {
                self.speech_cnt += 1;
                if is_speech {
                    self.silence_cnt = 0;
                    if self.speech_cnt >= self.max_speech_frame {
                        return self.close_current_segment(true);
                    }
                } else {
                    self.state = FireRedStreamVadState::PossibleSilence;
                    self.silence_cnt += 1;
                }
            }
            FireRedStreamVadState::PossibleSilence => {
                self.speech_cnt += 1;
                if is_speech {
                    self.state = FireRedStreamVadState::Speech;
                    self.silence_cnt = 0;
                    if self.speech_cnt >= self.max_speech_frame {
                        return self.close_current_segment(true);
                    }
                } else {
                    self.silence_cnt += 1;
                    if self.silence_cnt >= self.min_silence_frame {
                        self.state = FireRedStreamVadState::Silence;
                        let segment = self.close_current_segment(false);
                        self.speech_cnt = 0;
                        return segment;
                    }
                }
            }
        }
        None
    }

    fn close_current_segment(&mut self, hit_max_speech: bool) -> Option<VadSegment> {
        self.hit_max_speech = hit_max_speech;
        self.speech_cnt = 0;
        let start = self.last_speech_start_frame.take()?;
        let end = self.frame_cnt;
        self.last_speech_end_frame = Some(end);
        self.segment_from_frames(start, end)
    }

    fn segment_from_frames(&self, start_frame: usize, end_frame: usize) -> Option<VadSegment> {
        let start_ms = start_frame.saturating_sub(1) as u64 * FRAME_SHIFT_MS;
        let end_ms = end_frame.saturating_sub(1) as u64 * FRAME_SHIFT_MS;
        if end_ms.saturating_sub(start_ms) < self.min_speech_frame as u64 * FRAME_SHIFT_MS {
            return None;
        }
        Some(VadSegment {
            range: TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
            probability: self.speech_threshold,
        })
    }
}

fn ms_to_frame_count(ms: u64) -> usize {
    ms.div_ceil(FRAME_SHIFT_MS).max(1) as usize
}

fn validate_waveform(waveform: &Waveform) -> Result<()> {
    if waveform.sample_rate != SAMPLE_RATE {
        bail!(
            "FireRedVAD expects 16kHz mono audio, got sample_rate={}",
            waveform.sample_rate
        );
    }
    if waveform.channels != 1 {
        bail!(
            "FireRedVAD expects mono audio, got channels={}",
            waveform.channels
        );
    }
    Ok(())
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    if !model_dir.is_dir() {
        bail!(
            "FireRedVAD model path is not a directory: {}",
            model_dir.display()
        );
    }
    for name in ["cmvn.ark", "model.pth.tar"] {
        let path = model_dir.join(name);
        let meta = std::fs::metadata(&path)
            .with_context(|| format!("failed to stat {}", path.display()))?;
        if !meta.is_file() || meta.len() == 0 {
            bail!("FireRedVAD model file missing or empty: {}", path.display());
        }
    }
    Ok(())
}

fn modelscope_snapshot_dir(cache_dir: &Path, repo_id: &str, revision: &str) -> PathBuf {
    cache_dir
        .join("models")
        .join(repo_id.replace('/', "--"))
        .join("snapshots")
        .join(revision)
}

fn tensor_shape(reader: &PytorchReader, key: &str) -> Result<Vec<usize>> {
    Ok(reader
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?
        .shape
        .as_ref()
        .to_vec())
}

fn load_vec(reader: &PytorchReader, key: &str) -> Result<Vec<f32>> {
    let snapshot = reader
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?;
    if snapshot.dtype != DType::F32 {
        bail!("{key} must be F32, got {:?}", snapshot.dtype);
    }
    Ok(snapshot.to_data()?.convert::<f32>().into_vec::<f32>()?)
}

fn load_fsmn_weight(
    reader: &PytorchReader,
    device: &FlexDevice,
    key: &str,
) -> Result<(Tensor<Backend, 3>, Vec<f32>)> {
    let shape = tensor_shape(reader, key)?;
    if shape != [PROJ_DIM, 1, FSMN_ORDER] {
        bail!("{key} must have shape [{PROJ_DIM}, 1, {FSMN_ORDER}], got {shape:?}");
    }
    let raw = load_vec(reader, key)?;
    let mut transposed = vec![0.0; raw.len()];
    for channel in 0..PROJ_DIM {
        for kernel in 0..FSMN_ORDER {
            transposed[kernel * PROJ_DIM + channel] = raw[channel * FSMN_ORDER + kernel];
        }
    }
    Ok((
        Tensor::<Backend, 3>::from_data(TensorData::new(raw, [PROJ_DIM, 1, FSMN_ORDER]), device),
        transposed,
    ))
}

fn load_optional_fsmn_weight(
    reader: &PytorchReader,
    device: &FlexDevice,
    key: &str,
) -> Result<Option<Tensor<Backend, 3>>> {
    if reader.get(key).is_none() {
        return Ok(None);
    }
    let (tensor, _) = load_fsmn_weight(reader, device, key)?;
    Ok(Some(tensor))
}

fn read_kaldi_binary_cmvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read CMVN {}", path.display()))?;
    if bytes.len() < 16 || bytes[0] != 0 || &bytes[1..4] != b"BDM" {
        bail!("unsupported FireRedVAD CMVN format: {}", path.display());
    }
    let mut offset = 4usize;
    if bytes.get(offset) == Some(&b' ') {
        offset += 1;
    }
    let rows = read_kaldi_i32(&bytes, &mut offset)? as usize;
    let cols = read_kaldi_i32(&bytes, &mut offset)? as usize;
    if rows != 2 || cols != INPUT_DIM + 1 {
        bail!("unexpected FireRedVAD CMVN shape {rows}x{cols}");
    }
    let need = rows * cols * 8;
    if bytes.len() < offset + need {
        bail!("truncated FireRedVAD CMVN matrix: {}", path.display());
    }
    let mut values = Vec::with_capacity(rows * cols);
    for chunk in bytes[offset..offset + need].chunks_exact(8) {
        values.push(f64::from_le_bytes(chunk.try_into().expect("8 bytes")));
    }
    let count = values[INPUT_DIM].max(1.0);
    let mut means = Vec::with_capacity(INPUT_DIM);
    let mut inverse_std = Vec::with_capacity(INPUT_DIM);
    for dim in 0..INPUT_DIM {
        let mean = values[dim] / count;
        let variance = (values[cols + dim] / count - mean * mean).max(1e-20);
        means.push(mean as f32);
        inverse_std.push((1.0 / variance.sqrt()) as f32);
    }
    Ok((means, inverse_std))
}

fn read_kaldi_i32(bytes: &[u8], offset: &mut usize) -> Result<i32> {
    if bytes.get(*offset) != Some(&4) || *offset + 5 > bytes.len() {
        bail!("invalid Kaldi binary int marker");
    }
    *offset += 1;
    let value = i32::from_le_bytes(bytes[*offset..*offset + 4].try_into().expect("4 bytes"));
    *offset += 4;
    Ok(value)
}
