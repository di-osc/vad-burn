use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use burn::tensor::Tensor;

use super::constants::{Backend, FEAT_DIM, SAMPLE_RATE};
use super::frontend::FsmnVadFrontend;
use super::post::{FsmnVadPostProcessor, FsmnVadStreamingPostProcessor};
use super::timing::{FsmnForwardTiming, FsmnVadTiming};
use super::weights::BurnFsmnWeights;
use crate::{VadOptions, VadSegment, Waveform};

pub type FeatureTensor = Tensor<Backend, 2>;

#[derive(Debug, Clone)]
pub struct FsmnVadDetection {
    pub segments: Vec<VadSegment>,
    pub frame_scores: Vec<Vec<f32>>,
    pub timing: FsmnVadTiming,
}

pub struct FsmnVadModel {
    frontend: FsmnVadFrontend,
    post_processor: FsmnVadPostProcessor,
    weights: Arc<BurnFsmnWeights>,
    model_dir: PathBuf,
}

pub struct FsmnVadStream {
    frontend: FsmnVadFrontend,
    weights: Arc<BurnFsmnWeights>,
    options: VadOptions,
    caches: Vec<Tensor<Backend, 2>>,
    post_processor: FsmnVadStreamingPostProcessor,
    samples: Vec<f32>,
    processed_frames: usize,
    frame_scores: Vec<Vec<f32>>,
}

impl FsmnVadModel {
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let frontend = FsmnVadFrontend::new(&model_dir)?;
        let weights = BurnFsmnWeights::load(&model_dir)?;
        Ok(Self {
            frontend,
            post_processor: FsmnVadPostProcessor,
            weights: Arc::new(weights),
            model_dir,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn forward_frame_scores(&self, feats: FeatureTensor) -> Result<Vec<Vec<f32>>> {
        let mut caches = self.weights.zero_caches();
        self.weights.forward_frame_scores(feats, &mut caches)
    }

    pub fn forward_frame_scores_with_timing(
        &self,
        feats: FeatureTensor,
    ) -> Result<(Vec<Vec<f32>>, FsmnForwardTiming)> {
        let mut caches = self.weights.zero_caches();
        let mut timing = FsmnForwardTiming::default();
        let scores =
            self.weights
                .forward_frame_scores_with_timing(feats, &mut caches, &mut timing)?;
        Ok((scores, timing))
    }

    pub fn new_stream(&self, options: VadOptions) -> FsmnVadStream {
        FsmnVadStream {
            frontend: self.frontend.clone(),
            weights: Arc::clone(&self.weights),
            caches: self.weights.zero_caches(),
            post_processor: FsmnVadStreamingPostProcessor::new(options.clone()),
            options,
            samples: Vec::new(),
            processed_frames: 0,
            frame_scores: Vec::new(),
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

    pub fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
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
        segment: &VadSegment,
        options: &VadOptions,
        min_silence_ms: u64,
    ) -> Result<Vec<VadSegment>> {
        let local_waveform = waveform.slice_ms(segment.range.start.0, segment.range.end.0);
        let mut refined_options = options.clone();
        refined_options.min_silence_ms = min_silence_ms;
        refined_options.max_segment_ms = 0;
        Ok(self
            .detect(&local_waveform, &refined_options)?
            .into_iter()
            .map(|mut local| {
                local.range.start.0 = local.range.start.0.saturating_add(segment.range.start.0);
                local.range.end.0 = local
                    .range
                    .end
                    .0
                    .saturating_add(segment.range.start.0)
                    .min(segment.range.end.0);
                local
            })
            .filter(|local| local.range.end.0 > local.range.start.0)
            .collect())
    }
}

impl FsmnVadStream {
    pub fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<VadSegment>> {
        self.process_chunk(samples, sample_rate, false)
    }

    pub fn finish(&mut self) -> Result<Vec<VadSegment>> {
        self.process_chunk(&[], SAMPLE_RATE, true)
    }

    fn process_chunk(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        is_final: bool,
    ) -> Result<Vec<VadSegment>> {
        let waveform = Waveform::new(samples.to_vec(), sample_rate);
        validate_waveform(&waveform)?;

        self.samples.extend_from_slice(samples);
        let feats = self
            .frontend
            .extract_features_from_normalized_f32(&self.samples)?;
        let [frames, feat_dim] = feats.dims();
        if feat_dim != FEAT_DIM {
            bail!("FSMN VAD expects feature dim {FEAT_DIM}, got {feat_dim}");
        }
        let frame_scores = if frames > self.processed_frames {
            let feats = feats.slice([self.processed_frames..frames, 0..FEAT_DIM]);
            self.processed_frames = frames;
            self.weights
                .forward_frame_scores_streaming(feats, &mut self.caches)?
        } else {
            Vec::new()
        };
        let segments = self
            .post_processor
            .detect_chunk(samples, &frame_scores, is_final);
        self.frame_scores.extend(frame_scores);

        if is_final {
            self.reset();
        }

        Ok(segments)
    }

    pub fn frame_scores(&self) -> &[Vec<f32>] {
        &self.frame_scores
    }

    pub fn options(&self) -> &VadOptions {
        &self.options
    }

    pub fn reset(&mut self) {
        self.caches = self.weights.zero_caches();
        self.post_processor = FsmnVadStreamingPostProcessor::new(self.options.clone());
        self.samples.clear();
        self.processed_frames = 0;
        self.frame_scores.clear();
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
    use anyhow::{Result, bail};

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
        let waveform = load_pcm16_wav(&audio)?.slice_ms(0, 12_000);
        let options = VadOptions::default();
        let burn = FsmnVadModel::from_pretrained(model_dir)?;

        let detection = burn.detect_with_timing(&waveform, &options)?;
        assert!(!detection.frame_scores.is_empty());
        assert!(!detection.segments.is_empty());
        assert!(detection.timing.frontend_seconds > 0.0);
        assert!(detection.timing.forward_seconds > 0.0);

        let mut split_options = options.clone();
        split_options.max_segment_ms = 1_000;
        let full_waveform = load_pcm16_wav(&audio)?;
        let split_segments = burn.detect(&full_waveform, &split_options)?;
        assert!(!split_segments.is_empty());
        assert!(
            split_segments
                .iter()
                .all(|segment| segment.range.end.0 - segment.range.start.0 <= 1_000)
        );
        Ok(())
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    fn default_model_path() -> Option<PathBuf> {
        [
            PathBuf::from(
                "/workspace/data/models/asr/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            ),
            workspace_root().join(".cache/fsmn-vad"),
        ]
        .into_iter()
        .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists())
    }

    fn load_pcm16_wav(path: &Path) -> Result<Waveform> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            bail!("expected RIFF/WAVE file");
        }
        let mut offset = 12usize;
        let mut sample_rate = None;
        let mut channels = None;
        let mut bits_per_sample = None;
        let mut data = None;
        while offset + 8 <= bytes.len() {
            let chunk_id = &bytes[offset..offset + 4];
            let chunk_size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
            offset += 8;
            if offset + chunk_size > bytes.len() {
                bail!("invalid WAV chunk size");
            }
            match chunk_id {
                b"fmt " => {
                    if chunk_size < 16 {
                        bail!("invalid fmt chunk");
                    }
                    let audio_format = u16::from_le_bytes(bytes[offset..offset + 2].try_into()?);
                    if audio_format != 1 {
                        bail!("expected PCM WAV");
                    }
                    channels = Some(u16::from_le_bytes(
                        bytes[offset + 2..offset + 4].try_into()?,
                    ));
                    sample_rate = Some(u32::from_le_bytes(
                        bytes[offset + 4..offset + 8].try_into()?,
                    ));
                    bits_per_sample = Some(u16::from_le_bytes(
                        bytes[offset + 14..offset + 16].try_into()?,
                    ));
                }
                b"data" => data = Some(bytes[offset..offset + chunk_size].to_vec()),
                _ => {}
            }
            offset += chunk_size + (chunk_size % 2);
        }
        if bits_per_sample != Some(16) {
            bail!("expected 16-bit PCM WAV");
        }
        let samples = data
            .ok_or_else(|| anyhow::anyhow!("missing data chunk"))?
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
            .collect();
        Ok(Waveform::new_with_channels(
            samples,
            sample_rate.ok_or_else(|| anyhow::anyhow!("missing sample rate"))?,
            channels.ok_or_else(|| anyhow::anyhow!("missing channels"))?,
        ))
    }
}
