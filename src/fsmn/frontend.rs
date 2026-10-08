//! FSMN VAD 特征前端：fbank -> LFR -> CMVN，支持离线与流式。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::tensor::{Device, Int, Tensor, TensorData};
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};

use super::FeatureTensor;

/// 离线特征前端，持有 CMVN 参数。
#[derive(Debug, Clone)]
pub struct FsmnVadFrontend {
    config: WavFrontendConfig,
    device: Device,
    cmvn_means: Option<FeatureTensor>,
    cmvn_vars: Option<FeatureTensor>,
}

/// 流式特征前端，按块累积 fbank 帧并增量产出 LFR 特征。
pub struct FsmnVadFeatureStream {
    config: WavFrontendConfig,
    device: Device,
    cmvn_means: Option<FeatureTensor>,
    cmvn_vars: Option<FeatureTensor>,
    fbank: OnlineFbank,
    fbank_frames: Vec<Vec<f32>>,
    emitted_lfr_frames: usize,
}

impl FsmnVadFrontend {
    /// 在指定设备上从模型目录构造前端。
    pub fn new_on_device(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        validate_model_dir(model_dir)?;
        Self::from_config(
            WavFrontendConfig {
                sample_rate: 16_000,
                lfr_m: 5,
                lfr_n: 1,
                cmvn_file: Some(model_dir.join("am.mvn")),
                ..Default::default()
            },
            device,
        )
    }

    /// 从归一化到 `[-1, 1]` 的浮点 PCM 提取 LFR + CMVN 特征。
    pub fn extract_features_from_normalized_f32(&self, samples: &[f32]) -> Result<FeatureTensor> {
        let waveform = samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        let fbank = self.compute_fbank_features(&waveform)?;
        let lfr = self.apply_lfr(fbank);
        Ok(self.apply_cmvn(lfr))
    }

    /// 基于当前前端配置创建一个流式特征会话。
    pub fn new_stream(&self) -> FsmnVadFeatureStream {
        FsmnVadFeatureStream {
            config: self.config.clone(),
            device: self.device.clone(),
            cmvn_means: self.cmvn_means.clone(),
            cmvn_vars: self.cmvn_vars.clone(),
            fbank: OnlineFbank::new(self.fbank_options()),
            fbank_frames: Vec::new(),
            emitted_lfr_frames: 0,
        }
    }

    /// 计算整段波形的 fbank 特征。
    fn compute_fbank_features(&self, waveform: &[f32]) -> Result<FeatureTensor> {
        let mut fbank = OnlineFbank::new(self.fbank_options());
        fbank.accept_waveform(self.config.sample_rate as f32, waveform);
        let frames = fbank.num_ready_frames() as usize;
        let mut out = Vec::with_capacity(frames * self.config.n_mels);
        for i in 0..frames as i32 {
            let frame = fbank
                .get_frame(i)
                .ok_or_else(|| anyhow::anyhow!("missing fbank frame {i}"))?;
            out.extend_from_slice(frame);
        }
        Ok(Tensor::<2>::from_data(
            TensorData::new(out, [frames, self.config.n_mels]),
            &self.device,
        ))
    }

    /// 当前前端的 fbank 配置。
    fn fbank_options(&self) -> FbankOptions {
        fbank_options(&self.config)
    }

    /// 对 fbank 做 LFR（低帧率）拼接。
    fn apply_lfr(&self, fbank: FeatureTensor) -> FeatureTensor {
        let [t, _] = fbank.dims();
        let n_mels = self.config.n_mels;
        let feat_dim = n_mels * self.config.lfr_m;
        if t == 0 {
            return Tensor::<2>::zeros([0, feat_dim], &self.device);
        }

        let t_lfr = t.div_ceil(self.config.lfr_n);
        // LFR 左侧需要补 (lfr_m - 1) / 2 行。
        let left_padding_rows = (self.config.lfr_m - 1) / 2;
        let padded = if left_padding_rows == 0 {
            fbank
        } else {
            let left_pad = fbank
                .clone()
                .slice([0..1, 0..n_mels])
                .repeat_dim(0, left_padding_rows);
            Tensor::cat(vec![left_pad, fbank], 0)
        };
        let padded_rows = t + left_padding_rows;

        // 每一帧拼接 lfr_m 个相邻 fbank 帧，通过 gather 索引实现。
        let mut parts = Vec::with_capacity(self.config.lfr_m);
        for m in 0..self.config.lfr_m {
            let mut indices = Vec::with_capacity(t_lfr);
            for row in 0..t_lfr {
                indices.push(((row * self.config.lfr_n + m).min(padded_rows - 1)) as i32);
            }
            let indices = Tensor::<1, Int>::from_data(
                TensorData::new(indices, [t_lfr]).convert::<i32>(),
                &self.device,
            );
            parts.push(padded.clone().select(0, indices));
        }
        Tensor::cat(parts, 1)
    }

    /// 应用 CMVN 归一化。
    fn apply_cmvn(&self, feats: FeatureTensor) -> FeatureTensor {
        apply_cmvn(feats, &self.cmvn_means, &self.cmvn_vars)
    }

    /// 加载 CMVN 参数并构造前端。
    fn from_config(config: WavFrontendConfig, device: Device) -> Result<Self> {
        let (cmvn_means, cmvn_vars) = if let Some(cmvn_path) = &config.cmvn_file {
            let (means, vars) = load_cmvn(cmvn_path)?;
            let dim = means.len();
            (
                Some(Tensor::<2>::from_data(
                    TensorData::new(means, [1, dim]),
                    &device,
                )),
                Some(Tensor::<2>::from_data(
                    TensorData::new(vars, [1, dim]),
                    &device,
                )),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            config,
            device,
            cmvn_means,
            cmvn_vars,
        })
    }
}

impl FsmnVadFeatureStream {
    /// 送入一块归一化浮点 PCM，返回本次新产生的 LFR + CMVN 特征。
    pub fn push_normalized_f32(&mut self, samples: &[f32]) -> Result<FeatureTensor> {
        let waveform = samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        self.fbank
            .accept_waveform(self.config.sample_rate as f32, &waveform);
        self.collect_ready_fbank_frames()?;
        let lfr = self.next_lfr_frames(false);
        Ok(apply_cmvn(lfr, &self.cmvn_means, &self.cmvn_vars))
    }

    /// 冲刷会话，返回剩余特征。
    pub fn finish(&mut self) -> Result<FeatureTensor> {
        self.fbank.input_finished();
        self.collect_ready_fbank_frames()?;
        let lfr = self.next_lfr_frames(true);
        Ok(apply_cmvn(lfr, &self.cmvn_means, &self.cmvn_vars))
    }

    /// 重置流式状态，便于复用到下一段音频。
    pub fn reset(&mut self) {
        self.fbank = OnlineFbank::new(fbank_options(&self.config));
        self.fbank_frames.clear();
        self.emitted_lfr_frames = 0;
    }

    /// 把 fbank 中已经就绪、但尚未收集的帧追加进缓存。
    fn collect_ready_fbank_frames(&mut self) -> Result<()> {
        let ready_frames = self.fbank.num_ready_frames() as usize;
        for frame_idx in self.fbank_frames.len()..ready_frames {
            let frame = self
                .fbank
                .get_frame(frame_idx as i32)
                .ok_or_else(|| anyhow::anyhow!("missing fbank frame {frame_idx}"))?;
            self.fbank_frames.push(frame.to_vec());
        }
        Ok(())
    }

    /// 产出尚未发射的 LFR 帧；`is_final` 时使用末尾不足窗口的残余帧。
    fn next_lfr_frames(&mut self, is_final: bool) -> FeatureTensor {
        let fbank_rows = self.fbank_frames.len();
        let n_mels = self.config.n_mels;
        let feat_dim = n_mels * self.config.lfr_m;
        if fbank_rows == 0 {
            return Tensor::<2>::zeros([0, feat_dim], &self.device);
        }

        let total_lfr_frames = if is_final {
            fbank_rows.div_ceil(self.config.lfr_n)
        } else {
            self.complete_lfr_frame_count(fbank_rows)
        };
        if total_lfr_frames <= self.emitted_lfr_frames {
            return Tensor::<2>::zeros([0, feat_dim], &self.device);
        }

        let left_padding_rows = (self.config.lfr_m - 1) / 2;
        let padded_rows = fbank_rows + left_padding_rows;
        let new_lfr_frames = total_lfr_frames - self.emitted_lfr_frames;
        let mut out = Vec::with_capacity(new_lfr_frames * feat_dim);

        for row in self.emitted_lfr_frames..total_lfr_frames {
            for m in 0..self.config.lfr_m {
                let padded_idx = (row * self.config.lfr_n + m).min(padded_rows - 1);
                let fbank_idx = padded_idx.saturating_sub(left_padding_rows);
                out.extend_from_slice(&self.fbank_frames[fbank_idx]);
            }
        }
        self.emitted_lfr_frames = total_lfr_frames;

        Tensor::<2>::from_data(
            TensorData::new(out, [new_lfr_frames, feat_dim]),
            &self.device,
        )
    }

    /// 在不看未来帧的前提下，当前能完整产出的 LFR 帧数。
    fn complete_lfr_frame_count(&self, fbank_rows: usize) -> usize {
        let left_padding_rows = (self.config.lfr_m - 1) / 2;
        if fbank_rows <= left_padding_rows {
            return 0;
        }
        ((fbank_rows - left_padding_rows - 1) / self.config.lfr_n) + 1
    }
}

/// 应用 CMVN：`(feats + means) * vars`，维度不匹配时原样返回。
fn apply_cmvn(
    feats: FeatureTensor,
    cmvn_means: &Option<FeatureTensor>,
    cmvn_vars: &Option<FeatureTensor>,
) -> FeatureTensor {
    let (Some(means), Some(vars)) = (cmvn_means, cmvn_vars) else {
        return feats;
    };
    if means.dims()[1] != feats.dims()[1] || vars.dims()[1] != feats.dims()[1] {
        return feats;
    }
    (feats + means.clone()) * vars.clone()
}

/// 根据前端配置生成 kaldi fbank 参数。
fn fbank_options(config: &WavFrontendConfig) -> FbankOptions {
    FbankOptions {
        frame_opts: FrameExtractionOptions {
            samp_freq: config.sample_rate as f32,
            window_type: c"hamming".as_ptr(),
            dither: 0.0,
            frame_shift_ms: config.frame_shift_ms,
            frame_length_ms: config.frame_length_ms,
            snip_edges: true,
            ..Default::default()
        },
        mel_opts: MelBanksOptions {
            num_bins: config.n_mels as i32,
            ..Default::default()
        },
        energy_floor: 0.0,
        ..Default::default()
    }
}

#[derive(Debug, Clone)]
struct WavFrontendConfig {
    sample_rate: i32,
    frame_length_ms: f32,
    frame_shift_ms: f32,
    n_mels: usize,
    lfr_m: usize,
    lfr_n: usize,
    cmvn_file: Option<PathBuf>,
}

impl Default for WavFrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            n_mels: 80,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_file: None,
        }
    }
}

fn load_cmvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read CMVN file {}", path.display()))?;
    let means = extract_cmvn_vector(&text, "<AddShift>")
        .with_context(|| format!("failed to parse AddShift CMVN in {}", path.display()))?;
    let vars = extract_cmvn_vector(&text, "<Rescale>")
        .with_context(|| format!("failed to parse Rescale CMVN in {}", path.display()))?;
    if means.len() != vars.len() {
        bail!(
            "CMVN file {} has mismatched AddShift/Rescale dims: {} vs {}",
            path.display(),
            means.len(),
            vars.len()
        );
    }
    Ok((means, vars))
}

fn extract_cmvn_vector(text: &str, section: &str) -> Result<Vec<f32>> {
    let section_start = text
        .find(section)
        .ok_or_else(|| anyhow::anyhow!("missing {section} section"))?;
    let after_section = &text[section_start + section.len()..];
    let learn_rate = "<LearnRateCoef>";
    let learn_start = after_section
        .find(learn_rate)
        .ok_or_else(|| anyhow::anyhow!("missing {learn_rate} after {section}"))?;
    let after_learn = &after_section[learn_start + learn_rate.len()..];
    let bracket_start = after_learn
        .find('[')
        .ok_or_else(|| anyhow::anyhow!("missing vector start after {section}"))?;
    let after_bracket = &after_learn[bracket_start + 1..];
    let bracket_end = after_bracket
        .find(']')
        .ok_or_else(|| anyhow::anyhow!("missing vector end after {section}"))?;
    let values = after_bracket[..bracket_end]
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f32>()
                .with_context(|| format!("invalid CMVN value {token:?} in {section}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        bail!("empty CMVN vector in {section}");
    }
    Ok(values)
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    if !model_dir.is_dir() {
        bail!(
            "FSMN VAD model path is not a directory: {}",
            model_dir.display()
        );
    }
    for name in ["model.pt", "am.mvn"] {
        let path = model_dir.join(name);
        let meta = std::fs::metadata(&path)
            .with_context(|| format!("failed to stat {}", path.display()))?;
        if !meta.is_file() || meta.len() == 0 {
            bail!("FSMN VAD model file missing or empty: {}", path.display());
        }
    }
    Ok(())
}
