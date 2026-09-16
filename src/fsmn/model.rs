use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use burn::prelude::Backend as BurnBackend;
use burn::tensor::Tensor;

use super::constants::{Backend, FEAT_DIM, SAMPLE_RATE};
use super::frontend::{FsmnVadFeatureStream, FsmnVadFrontend};
use super::post::{FsmnVadPostProcessor, FsmnVadStreamingPostProcessor};
use super::timing::{FsmnForwardTiming, FsmnVadTiming};
use super::weights::BurnFsmnWeights;
use crate::{
    Audio, AudioChannel, AudioChunk, AudioStream, TimeSpan, VadOptions, Waveform, annotate_audio,
    prepare_stream_16k,
};

pub type FeatureTensor<B = Backend> = Tensor<B, 2>;

pub const DEFAULT_MODELSCOPE_REPO_ID: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";
pub const DEFAULT_MODELSCOPE_REVISION: &str = "master";

#[derive(Debug, Clone)]
pub struct FsmnVadDetection {
    pub segments: Vec<TimeSpan>,
    pub frame_scores: Vec<Vec<f32>>,
    pub timing: FsmnVadTiming,
}

pub struct FsmnVadModel<B: BurnBackend = Backend> {
    frontend: FsmnVadFrontend<B>,
    post_processor: FsmnVadPostProcessor,
    weights: Arc<BurnFsmnWeights<B>>,
    model_dir: PathBuf,
}

/// 有状态的流式 FSMN VAD 会话，持有特征缓存和在线切段状态。
pub struct FsmnVadSession<B: BurnBackend = Backend> {
    frontend: FsmnVadFrontend<B>,
    weights: Arc<BurnFsmnWeights<B>>,
    options: VadOptions,
    channels: HashMap<AudioChannel, FsmnVadChannel<B>>,
    resamplers: HashMap<AudioChannel, asr_data::StreamingResampler>,
}

/// 单个声道的流式 FSMN 推理状态。
struct FsmnVadChannel<B: BurnBackend> {
    feature_stream: FsmnVadFeatureStream<B>,
    caches: Vec<Tensor<B, 2>>,
    post_processor: FsmnVadStreamingPostProcessor,
    samples: Vec<f32>,
    pending_samples: Vec<f32>,
    pending_frame_scores: Vec<Vec<f32>>,
    frame_scores: Vec<Vec<f32>>,
}

impl FsmnVadModel {
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_pretrained_on_device(model_dir, Default::default())
    }
}

impl<B: BurnBackend> FsmnVadModel<B> {
    pub fn from_pretrained_on_device(
        model_dir: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let frontend = FsmnVadFrontend::new_on_device(&model_dir, device.clone())?;
        let weights = BurnFsmnWeights::load(&model_dir, device)?;
        Ok(Self {
            frontend,
            post_processor: FsmnVadPostProcessor,
            weights: Arc::new(weights),
            model_dir,
        })
    }
}

impl FsmnVadModel {
    pub fn from_modelscope() -> Result<Self> {
        Self::from_modelscope_revision(DEFAULT_MODELSCOPE_REPO_ID, DEFAULT_MODELSCOPE_REVISION)
    }

    pub fn from_modelscope_revision(repo_id: &str, revision: &str) -> Result<Self> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(Self::from_modelscope_revision_async(repo_id, revision))
    }

    pub async fn from_modelscope_async() -> Result<Self> {
        Self::from_modelscope_revision_async(
            DEFAULT_MODELSCOPE_REPO_ID,
            DEFAULT_MODELSCOPE_REVISION,
        )
        .await
    }

    pub async fn from_modelscope_revision_async(repo_id: &str, revision: &str) -> Result<Self> {
        let cache_dir = modelhub::modelscope::cache_dir();
        modelhub::modelscope::download_model_revision(repo_id, revision, &cache_dir).await?;
        Self::from_pretrained(modelscope_snapshot_dir(&cache_dir, repo_id, revision))
    }
}

impl<B: BurnBackend> FsmnVadModel<B> {
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn forward_frame_scores(&self, feats: FeatureTensor<B>) -> Result<Vec<Vec<f32>>> {
        let mut caches = self.weights.zero_caches();
        self.weights.forward_frame_scores(feats, &mut caches)
    }

    pub fn forward_frame_scores_with_timing(
        &self,
        feats: FeatureTensor<B>,
    ) -> Result<(Vec<Vec<f32>>, FsmnForwardTiming)> {
        let mut caches = self.weights.zero_caches();
        let mut timing = FsmnForwardTiming::default();
        let scores =
            self.weights
                .forward_frame_scores_with_timing(feats, &mut caches, &mut timing)?;
        Ok((scores, timing))
    }

    /// 创建有状态的流式推理会话。
    ///
    /// 每个音频流对应一个会话；不要在同一会话上并发调用 `push` / `finish`。
    pub fn new_session(&self, options: VadOptions) -> FsmnVadSession<B> {
        FsmnVadSession {
            frontend: self.frontend.clone(),
            weights: Arc::clone(&self.weights),
            options,
            channels: HashMap::new(),
            resamplers: HashMap::new(),
        }
    }

    pub fn detect_with_timing(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<FsmnVadDetection> {
        validate_waveform(waveform)?;

        let mut timing = FsmnVadTiming::default();
        let frontend_start = Instant::now();
        let feats = self
            .frontend
            .extract_features_from_normalized_f32(&waveform.samples)?;
        timing.frontend_seconds = frontend_start.elapsed().as_secs_f64();

        let forward_start = Instant::now();
        let (frame_scores, forward_ops) = self.forward_frame_scores_with_timing(feats)?;
        timing.forward_seconds = forward_start.elapsed().as_secs_f64();
        timing.forward_ops = forward_ops;

        let segment_start = Instant::now();
        let segments = self.post_processor.segments_from_frame_scores(
            waveform,
            &frame_scores,
            options,
            |waveform, segment, options, min_silence_ms| {
                self.detect_segment_with_min_silence(waveform, segment, options, min_silence_ms)
            },
        )?;
        timing.segmenter_seconds = segment_start.elapsed().as_secs_f64();

        Ok(FsmnVadDetection {
            segments,
            frame_scores,
            timing,
        })
    }

    /// 按声道检测并把 `speech` 活动写进 [`Audio`] 的 prediction timeline。
    ///
    /// # Errors
    ///
    /// 解码、重采样、推理或写入标注失败时返回错误。
    pub fn annotate(&self, audio: &mut Audio, options: &VadOptions) -> Result<()> {
        annotate_audio(audio, |waveform| self.detect(waveform, options))
    }

    pub fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<TimeSpan>> {
        validate_waveform(waveform)?;

        let feats = self
            .frontend
            .extract_features_from_normalized_f32(&waveform.samples)?;
        let frame_scores = self.forward_frame_scores(feats)?;
        self.post_processor.segments_from_frame_scores(
            waveform,
            &frame_scores,
            options,
            |waveform, segment, options, min_silence_ms| {
                self.detect_segment_with_min_silence(waveform, segment, options, min_silence_ms)
            },
        )
    }

    fn detect_segment_with_min_silence(
        &self,
        waveform: &Waveform,
        segment: &TimeSpan,
        options: &VadOptions,
        min_silence_ms: u64,
    ) -> Result<Vec<TimeSpan>> {
        let local_waveform =
            waveform.slice_ms(segment.range.start_ms as u64, segment.range.end_ms as u64);
        let mut refined_options = options.clone();
        refined_options.min_silence_ms = min_silence_ms;
        refined_options.max_segment_ms = 0;
        Ok(self
            .detect(&local_waveform, &refined_options)?
            .into_iter()
            .map(|mut local| {
                let offset = segment.range.start_ms;
                local.range.start_ms = local.range.start_ms.saturating_add(offset);
                local.range.end_ms = local
                    .range
                    .end_ms
                    .saturating_add(offset)
                    .min(segment.range.end_ms);
                local
            })
            .filter(|local| local.range.end_ms > local.range.start_ms)
            .collect())
    }
}

fn modelscope_snapshot_dir(cache_dir: &Path, repo_id: &str, revision: &str) -> PathBuf {
    cache_dir
        .join("models")
        .join(repo_id.replace('/', "--"))
        .join("snapshots")
        .join(revision)
}

impl<B: BurnBackend> FsmnVadSession<B> {
    /// 标注已经从 `stream` 拉下来的一块音频，并把新产生的 activity 写进 timeline。
    ///
    /// 每个声道使用独立推理状态。源采样率不是 16 kHz 时在内部做有状态重采样。
    /// 最后一块会 flush 该声道。返回本块新写出的 span，便于立刻使用中间结果。
    ///
    /// # Errors
    ///
    /// 重采样、推理或写入标注失败时返回错误。
    pub fn annotate(
        &mut self,
        stream: &mut AudioStream,
        chunk: &AudioChunk,
    ) -> Result<Vec<TimeSpan>> {
        stream.annotate_activity_chunk(chunk, |channel, waveform, is_final| {
            self.annotate_waveform(channel, waveform, is_final)
        })
    }

    /// 处理一块指定声道的流式波形，不写入 timeline。
    ///
    /// `is_final` 时冲刷该声道会话。采样率由模型在内部重采样到 16 kHz。
    ///
    /// # Errors
    ///
    /// 采样率为 0、重采样或推理失败时返回错误。
    pub fn annotate_waveform(
        &mut self,
        channel: AudioChannel,
        waveform: &Waveform,
        is_final: bool,
    ) -> Result<Vec<TimeSpan>> {
        let prepared = prepare_stream_16k(waveform, is_final, &mut self.resamplers, channel)?;
        let input = match prepared.as_ref() {
            Some(wave) if wave.samples.is_empty() && !is_final => return Ok(Vec::new()),
            Some(wave) => wave,
            None => waveform,
        };
        let mut spans = self.push_channel(channel, &input.samples, input.sample_rate)?;
        if is_final {
            spans.extend(self.finish_channel(channel)?);
        }
        Ok(spans)
    }

    pub fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<TimeSpan>> {
        self.push_channel(AudioChannel::Mono, samples, sample_rate)
    }

    pub fn finish(&mut self) -> Result<Vec<TimeSpan>> {
        let spans = if self.channels.contains_key(&AudioChannel::Mono) {
            self.finish_channel(AudioChannel::Mono)?
        } else {
            Vec::new()
        };
        self.reset();
        Ok(spans)
    }

    pub fn frame_scores(&self) -> &[Vec<f32>] {
        self.channels
            .get(&AudioChannel::Mono)
            .map(|channel| channel.frame_scores.as_slice())
            .unwrap_or(&[])
    }

    pub fn options(&self) -> &VadOptions {
        &self.options
    }

    pub fn reset(&mut self) {
        for channel in self.channels.values_mut() {
            channel.feature_stream.reset();
        }
        self.channels.clear();
        self.resamplers.clear();
    }

    fn channel_mut(&mut self, channel: AudioChannel) -> &mut FsmnVadChannel<B> {
        if !self.channels.contains_key(&channel) {
            let inner = FsmnVadChannel::new(&self.frontend, &self.weights, &self.options);
            self.channels.insert(channel, inner);
        }
        self.channels
            .get_mut(&channel)
            .expect("channel session inserted")
    }

    fn push_channel(
        &mut self,
        channel: AudioChannel,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<TimeSpan>> {
        let weights = Arc::clone(&self.weights);
        self.channel_mut(channel)
            .push(&weights, samples, sample_rate)
    }

    fn finish_channel(&mut self, channel: AudioChannel) -> Result<Vec<TimeSpan>> {
        let weights = Arc::clone(&self.weights);
        Ok(self
            .channels
            .remove(&channel)
            .map(|mut inner| inner.finish(&weights))
            .transpose()?
            .unwrap_or_default())
    }
}

impl<B: BurnBackend> FsmnVadChannel<B> {
    fn new(
        frontend: &FsmnVadFrontend<B>,
        weights: &BurnFsmnWeights<B>,
        options: &VadOptions,
    ) -> Self {
        Self {
            feature_stream: frontend.new_stream(),
            caches: weights.zero_caches(),
            post_processor: FsmnVadStreamingPostProcessor::new(options.clone()),
            samples: Vec::new(),
            pending_samples: Vec::new(),
            pending_frame_scores: Vec::new(),
            frame_scores: Vec::new(),
        }
    }

    fn push(
        &mut self,
        weights: &BurnFsmnWeights<B>,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<TimeSpan>> {
        let frame_scores = self.next_frame_scores(weights, samples, sample_rate)?;
        let segments = if self.pending_frame_scores.is_empty() {
            Vec::new()
        } else {
            self.post_processor.detect_chunk(
                &self.pending_samples,
                &self.pending_frame_scores,
                false,
            )
        };
        self.pending_samples = samples.to_vec();
        self.pending_frame_scores = frame_scores;
        Ok(segments)
    }

    fn finish(&mut self, weights: &BurnFsmnWeights<B>) -> Result<Vec<TimeSpan>> {
        let final_frame_scores = self.next_final_frame_scores(weights)?;
        let mut segments = if self.pending_frame_scores.is_empty() {
            Vec::new()
        } else {
            self.post_processor.detect_chunk(
                &self.pending_samples,
                &self.pending_frame_scores,
                final_frame_scores.is_empty(),
            )
        };
        if !final_frame_scores.is_empty() {
            segments.extend(
                self.post_processor
                    .detect_chunk(&[], &final_frame_scores, true),
            );
        }
        Ok(segments)
    }

    fn next_frame_scores(
        &mut self,
        weights: &BurnFsmnWeights<B>,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Vec<f32>>> {
        let waveform = Waveform::new(samples.to_vec(), sample_rate);
        validate_waveform(&waveform)?;

        self.samples.extend_from_slice(samples);
        let feats = self.feature_stream.push_normalized_f32(samples)?;
        let [frames, feat_dim] = feats.dims();
        if feat_dim != FEAT_DIM {
            bail!("FSMN VAD expects feature dim {FEAT_DIM}, got {feat_dim}");
        }
        let frame_scores = if frames > 0 {
            weights.forward_frame_scores_streaming(feats, &mut self.caches)?
        } else {
            Vec::new()
        };
        self.frame_scores.extend(frame_scores.clone());
        Ok(frame_scores)
    }

    fn next_final_frame_scores(&mut self, weights: &BurnFsmnWeights<B>) -> Result<Vec<Vec<f32>>> {
        let feats = self.feature_stream.finish()?;
        let [frames, feat_dim] = feats.dims();
        if feat_dim != FEAT_DIM {
            bail!("FSMN VAD expects feature dim {FEAT_DIM}, got {feat_dim}");
        }
        let frame_scores = if frames > 0 {
            weights.forward_frame_scores_streaming(feats, &mut self.caches)?
        } else {
            Vec::new()
        };
        self.frame_scores.extend(frame_scores.clone());
        Ok(frame_scores)
    }
}

fn validate_waveform(waveform: &Waveform) -> Result<()> {
    if waveform.sample_rate != SAMPLE_RATE {
        bail!(
            "FSMN VAD expects 16kHz mono audio, got sample_rate={}",
            waveform.sample_rate
        );
    }
    if waveform.channels != 1 {
        bail!(
            "FSMN VAD expects 16kHz mono audio, got channels={}",
            waveform.channels
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[cfg(feature = "metal")]
    #[test]
    fn burn_metal_detects_fixture() -> Result<()> {
        use burn::backend::{Metal, wgpu::WgpuDevice};

        let Some(model_dir) = default_model_path() else {
            eprintln!("skipping: FSMN VAD model not found");
            return Ok(());
        };
        let audio = workspace_root().join("assets/vad_example.wav");
        if !audio.exists() {
            eprintln!("skipping: {} not found", audio.display());
            return Ok(());
        }
        let waveform = Waveform::from_path(&audio)?.slice_ms(0, 12_000);
        let options = VadOptions::default();
        let burn =
            FsmnVadModel::<Metal>::from_pretrained_on_device(model_dir, WgpuDevice::default())?;

        let detection = burn.detect_with_timing(&waveform, &options)?;
        assert!(!detection.frame_scores.is_empty());
        assert!(!detection.segments.is_empty());
        Ok(())
    }

    #[test]
    fn burn_flex_detects_fixture_and_splits_long_segments() -> Result<()> {
        let Some(model_dir) = default_model_path() else {
            eprintln!("skipping: FSMN VAD model not found");
            return Ok(());
        };
        let audio = workspace_root().join("assets/vad_example.wav");
        if !audio.exists() {
            eprintln!("skipping: {} not found", audio.display());
            return Ok(());
        }
        let waveform = Waveform::from_path(&audio)?.slice_ms(0, 12_000);
        let options = VadOptions::default();
        let burn = FsmnVadModel::from_pretrained(model_dir)?;

        let detection = burn.detect_with_timing(&waveform, &options)?;
        assert!(!detection.frame_scores.is_empty());
        assert!(!detection.segments.is_empty());
        assert!(detection.timing.frontend_seconds > 0.0);
        assert!(detection.timing.forward_seconds > 0.0);

        let mut split_options = options.clone();
        split_options.max_segment_ms = 1_000;
        let full_waveform = Waveform::from_path(&audio)?;
        let split_segments = burn.detect(&full_waveform, &split_options)?;
        assert!(!split_segments.is_empty());
        assert!(
            split_segments
                .iter()
                .all(|segment| segment.range.end_ms - segment.range.start_ms <= 1_000)
        );

        let streaming_segments = detect_streaming(&burn, &full_waveform, &options, 600)?;
        let offline_segments = burn.detect(&full_waveform, &options)?;
        assert_eq!(streaming_segments.len(), offline_segments.len());
        assert!(
            streaming_segments
                .iter()
                .zip(&offline_segments)
                .all(|(left, right)| left.content_eq(right))
        );
        Ok(())
    }

    fn detect_streaming(
        model: &FsmnVadModel,
        waveform: &Waveform,
        options: &VadOptions,
        chunk_ms: u64,
    ) -> Result<Vec<TimeSpan>> {
        let mut session = model.new_session(options.clone());
        let chunk_samples = (waveform.sample_rate as u64 * chunk_ms / 1000) as usize;
        let mut segments = Vec::new();
        for chunk in waveform.samples.chunks(chunk_samples) {
            segments.extend(session.push(chunk, waveform.sample_rate)?);
        }
        segments.extend(session.finish()?);
        Ok(segments)
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn default_model_path() -> Option<PathBuf> {
        [
            PathBuf::from(
                "/workspace/data/models/asr/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            ),
            PathBuf::from(
                "/Users/wangmengdi/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            ),
            workspace_root().join(".cache/fsmn-vad"),
        ]
        .into_iter()
        .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists())
    }
}
