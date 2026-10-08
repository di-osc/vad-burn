//! FSMN VAD 特征前端：fbank -> LFR -> CMVN，支持离线与流式。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::tensor::{Device, Int, Tensor, TensorData};

use super::FeatureTensor;
use crate::fbank::{FbankExtractor, FbankOptions, FbankStream, WindowType, to_kaldi_scale};

/// 离线特征前端，持有 Fbank 提取器与 CMVN 参数。
#[derive(Debug, Clone)]
pub struct FsmnVadFrontend {
    config: WavFrontendConfig,
    device: Device,
    fbank: FbankExtractor,
    cmvn_means: Option<FeatureTensor>,
    cmvn_vars: Option<FeatureTensor>,
}

/// 流式特征前端，按块累积 fbank 帧并增量产出 LFR 特征。
pub struct FsmnVadFeatureStream {
    config: WavFrontendConfig,
    device: Device,
    cmvn_means: Option<FeatureTensor>,
    cmvn_vars: Option<FeatureTensor>,
    fbank: FbankStream,
    /// 目前累积的全部 fbank 帧，形状 `[rows, n_mels]`。
    fbank_frames: FeatureTensor,
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
        let waveform = to_kaldi_scale(samples);
        let fbank = self.fbank.compute(&waveform);
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
            fbank: FbankStream::new(&self.fbank),
            fbank_frames: Tensor::<2>::zeros([0, self.config.n_mels], &self.device),
            emitted_lfr_frames: 0,
        }
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
        let fbank = FbankExtractor::new(fbank_options(&config), device.clone())?;
        Ok(Self {
            config,
            device,
            fbank,
            cmvn_means,
            cmvn_vars,
        })
    }
}

impl FsmnVadFeatureStream {
    /// 送入一块归一化浮点 PCM，返回本次新产生的 LFR + CMVN 特征。
    pub fn push_normalized_f32(&mut self, samples: &[f32]) -> Result<FeatureTensor> {
        let waveform = to_kaldi_scale(samples);
        let new_frames = self.fbank.push(&waveform);
        self.append_fbank_frames(new_frames);
        let lfr = self.next_lfr_frames(false);
        Ok(apply_cmvn(lfr, &self.cmvn_means, &self.cmvn_vars))
    }

    /// 冲刷会话，返回剩余特征。
    pub fn finish(&mut self) -> Result<FeatureTensor> {
        let new_frames = self.fbank.finish();
        self.append_fbank_frames(new_frames);
        let lfr = self.next_lfr_frames(true);
        Ok(apply_cmvn(lfr, &self.cmvn_means, &self.cmvn_vars))
    }

    /// 重置流式状态，便于复用到下一段音频。
    pub fn reset(&mut self) {
        self.fbank.reset();
        self.fbank_frames = Tensor::<2>::zeros([0, self.config.n_mels], &self.device);
        self.emitted_lfr_frames = 0;
    }

    /// 把新产出的 fbank 帧追加进累积张量。
    fn append_fbank_frames(&mut self, new_frames: FeatureTensor) {
        if new_frames.dims()[0] == 0 {
            return;
        }
        self.fbank_frames = Tensor::cat(vec![self.fbank_frames.clone(), new_frames], 0);
    }

    /// 产出尚未发射的 LFR 帧；`is_final` 时使用末尾不足窗口的残余帧。
    fn next_lfr_frames(&mut self, is_final: bool) -> FeatureTensor {
        let fbank_rows = self.fbank_frames.dims()[0];
        let n_mels = self.config.n_mels;
        let lfr_m = self.config.lfr_m;
        let lfr_n = self.config.lfr_n;
        let feat_dim = n_mels * lfr_m;
        if fbank_rows == 0 {
            return Tensor::<2>::zeros([0, feat_dim], &self.device);
        }

        let total_lfr_frames = if is_final {
            fbank_rows.div_ceil(lfr_n)
        } else {
            self.complete_lfr_frame_count(fbank_rows)
        };
        if total_lfr_frames <= self.emitted_lfr_frames {
            return Tensor::<2>::zeros([0, feat_dim], &self.device);
        }

        // 左侧补 (lfr_m - 1) / 2 行，复制第一帧即可。
        let left_padding_rows = (lfr_m - 1) / 2;
        let padded_rows = fbank_rows + left_padding_rows;
        let padded = if left_padding_rows == 0 {
            self.fbank_frames.clone()
        } else {
            let left_pad = self
                .fbank_frames
                .clone()
                .slice([0..1, 0..n_mels])
                .repeat_dim(0, left_padding_rows);
            Tensor::cat(vec![left_pad, self.fbank_frames.clone()], 0)
        };

        // 与离线路径相同的 gather 逻辑，只是只取尚未发射的行。
        let new_lfr_frames = total_lfr_frames - self.emitted_lfr_frames;
        let mut parts = Vec::with_capacity(lfr_m);
        for m in 0..lfr_m {
            let mut indices = Vec::with_capacity(new_lfr_frames);
            for row in self.emitted_lfr_frames..total_lfr_frames {
                indices.push(((row * lfr_n + m).min(padded_rows - 1)) as i32);
            }
            let indices = Tensor::<1, Int>::from_data(
                TensorData::new(indices, [new_lfr_frames]).convert::<i32>(),
                &self.device,
            );
            parts.push(padded.clone().select(0, indices));
        }
        self.emitted_lfr_frames = total_lfr_frames;

        Tensor::cat(parts, 1)
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

/// 根据前端配置生成 Kaldi 兼容的 Fbank 参数。
fn fbank_options(config: &WavFrontendConfig) -> FbankOptions {
    FbankOptions {
        sample_rate: config.sample_rate as u32,
        frame_length_ms: config.frame_length_ms,
        frame_shift_ms: config.frame_shift_ms,
        num_mel_bins: config.n_mels,
        low_freq: 20.0,
        high_freq: 0.0,
        preemph_coeff: 0.97,
        remove_dc_offset: true,
        window_type: WindowType::Hamming,
        round_to_power_of_two: true,
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
