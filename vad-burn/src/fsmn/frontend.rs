use std::ffi::CStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::backend::{Flex, flex::FlexDevice};
use burn::tensor::{Int, Tensor, TensorData};
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};

use super::BurnFeatureTensor;

#[derive(Debug, Clone)]
pub struct FsmnVadFrontend {
    config: WavFrontendConfig,
    device: FlexDevice,
    cmvn_means: Option<BurnFeatureTensor>,
    cmvn_vars: Option<BurnFeatureTensor>,
}

impl FsmnVadFrontend {
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        validate_model_dir(model_dir)?;
        Self::from_config(WavFrontendConfig {
            sample_rate: 16_000,
            lfr_m: 5,
            lfr_n: 1,
            cmvn_file: Some(model_dir.join("am.mvn")),
            ..Default::default()
        })
    }

    pub fn extract_features_from_normalized_f32(
        &self,
        samples: &[f32],
    ) -> Result<BurnFeatureTensor> {
        let waveform = samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        let fbank = self.compute_fbank_features(&waveform)?;
        let lfr = self.apply_lfr(fbank);
        Ok(self.apply_cmvn(lfr))
    }

    fn compute_fbank_features(&self, waveform: &[f32]) -> Result<BurnFeatureTensor> {
        let opt = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: self.config.sample_rate as f32,
                window_type: CStr::from_bytes_with_nul(b"hamming\0")?.as_ptr(),
                dither: 0.0,
                frame_shift_ms: self.config.frame_shift_ms,
                frame_length_ms: self.config.frame_length_ms,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: self.config.n_mels as i32,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };
        let mut fbank = OnlineFbank::new(opt);
        fbank.accept_waveform(self.config.sample_rate as f32, waveform);
        let frames = fbank.num_ready_frames() as usize;
        let mut out = Vec::with_capacity(frames * self.config.n_mels);
        for i in 0..frames as i32 {
            let frame = fbank
                .get_frame(i)
                .ok_or_else(|| anyhow::anyhow!("missing fbank frame {i}"))?;
            out.extend_from_slice(frame);
        }
        Ok(Tensor::<Flex, 2>::from_data(
            TensorData::new(out, [frames, self.config.n_mels]),
            &self.device,
        ))
    }

    fn apply_lfr(&self, fbank: BurnFeatureTensor) -> BurnFeatureTensor {
        let [t, _] = fbank.dims();
        let n_mels = self.config.n_mels;
        let feat_dim = n_mels * self.config.lfr_m;
        if t == 0 {
            return Tensor::<Flex, 2>::zeros([0, feat_dim], &self.device);
        }

        let t_lfr = t.div_ceil(self.config.lfr_n);
        let left_padding_rows = (self.config.lfr_m - 1) / 2;
        let padded = if left_padding_rows == 0 {
            fbank
        } else {
            let left_pad = fbank.clone().slice([0..1]).repeat_dim(0, left_padding_rows);
            Tensor::cat(vec![left_pad, fbank], 0)
        };
        let padded_rows = t + left_padding_rows;

        let mut parts = Vec::with_capacity(self.config.lfr_m);
        for m in 0..self.config.lfr_m {
            let mut indices = Vec::with_capacity(t_lfr);
            for row in 0..t_lfr {
                indices.push(((row * self.config.lfr_n + m).min(padded_rows - 1)) as i32);
            }
            let indices = Tensor::<Flex, 1, Int>::from_data(
                TensorData::new(indices, [t_lfr]).convert::<i32>(),
                &self.device,
            );
            parts.push(padded.clone().select(0, indices));
        }
        Tensor::cat(parts, 1)
    }

    fn apply_cmvn(&self, feats: BurnFeatureTensor) -> BurnFeatureTensor {
        let (Some(means), Some(vars)) = (&self.cmvn_means, &self.cmvn_vars) else {
            return feats;
        };
        if means.dims()[1] != feats.dims()[1] || vars.dims()[1] != feats.dims()[1] {
            return feats;
        }
        (feats + means.clone()) * vars.clone()
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

impl FsmnVadFrontend {
    fn from_config(config: WavFrontendConfig) -> Result<Self> {
        let device = FlexDevice;
        let (cmvn_means, cmvn_vars) = if let Some(cmvn_path) = &config.cmvn_file {
            let (means, vars) = load_cmvn(cmvn_path)?;
            let dim = means.len();
            (
                Some(Tensor::<Flex, 2>::from_data(
                    TensorData::new(means, [1, dim]),
                    &device,
                )),
                Some(Tensor::<Flex, 2>::from_data(
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
