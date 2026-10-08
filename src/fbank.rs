//! 基于 Burn 的 Kaldi 兼容 Fbank（log-mel filterbank）特征提取。
//!
//! 本模块复刻 [`kaldi-native-fbank`] 的默认行为，使特征与
//! `kaldi-fbank-rust-kautism`（其 Rust 封装）保持一致：
//!
//! 1. 按 `snip_edges = true` 分帧：帧 `f` 的起始样本为 `f * frame_shift`，
//!    帧数不足一帧时直接丢弃（不补边）。
//! 2. 去直流：减掉当前帧内的均值（Kaldi 默认 `remove_dc_offset = true`）。
//! 3. 预加重：`x[i] -= coeff * x[i-1]`，且 `x[0] -= coeff * x[0]`。
//!    注意顺序为先预加重、后加窗，与 Kaldi `ProcessWindow` 一致。
//! 4. 加窗，并零填充到 2 的幂长度做实数 FFT（正向、不归一化）。
//! 5. 幂率谱 `re^2 + im^2`，只保留前 `fft_size / 2` 个 bin（Kaldi 会丢弃
//!    Nyquist bin）。
//! 6. 三角 Mel 滤波器组（HTK 尺度 `1127 * ln(1 + f / 700)`）。
//! 7. 自然对数压缩，下界为 `f32::EPSILON`。
//!
//! 输入样本需与 Kaldi 一致地缩放到 int16 量级（即 `[-1, 1] * 32768`）。
//!
//! [`kaldi-native-fbank`]: https://github.com/csukuangfj/kaldi-native-fbank

use anyhow::{Result, bail};
use burn::signal::rfft;
use burn::tensor::{Device, Tensor, TensorData};

/// 窗函数类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hamming 窗：`0.54 - 0.46 * cos(2πi / (N - 1))`。
    Hamming,
    /// Povey 窗：`(0.5 - 0.5 * cos(2πi / (N - 1))) ^ 0.85`，两端收敛到 0。
    Povey,
}

impl WindowType {
    /// 生成 `frame_length` 个窗系数。
    ///
    /// 与 Kaldi 相同：角度在 `f64` 下计算，最后再收敛到 `f32`。
    fn coefficients(self, frame_length: usize) -> Vec<f32> {
        // Kaldi 使用 M_2PI / (frame_length - 1)，即 sqrt(2π) 的两倍。
        let step = std::f64::consts::TAU / (frame_length - 1) as f64;
        (0..frame_length)
            .map(|i| {
                let cosine = (step * i as f64).cos();
                match self {
                    WindowType::Hamming => (0.54 - 0.46 * cosine) as f32,
                    WindowType::Povey => (0.5 - 0.5 * cosine).powf(0.85) as f32,
                }
            })
            .collect()
    }
}

/// Fbank 提取参数，语义对齐 Kaldi `FbankOptions` 中被本库使用的字段。
///
/// 未实现的 Kaldi 选项：VTLN 弯曲、HTK 兼容模式、dither（调用方统一使用 0）、
/// librosa 风格 Mel 滤波器组。
#[derive(Debug, Clone)]
pub struct FbankOptions {
    /// 采样率（Hz）。
    pub sample_rate: u32,
    /// 帧长（ms）。
    pub frame_length_ms: f32,
    /// 帧移（ms）。
    pub frame_shift_ms: f32,
    /// Mel 滤波器个数，同时也是输出特征维度。
    pub num_mel_bins: usize,
    /// Mel 滤波器组低截止频率（Hz）。
    pub low_freq: f32,
    /// Mel 滤波器组高截止频率（Hz）；`<= 0` 时表示相对 Nyquist 的偏移，
    /// 与 Kaldi 的 `high_freq = nyquist + high_freq` 一致。
    pub high_freq: f32,
    /// 预加重系数，`0.0` 表示跳过预加重。
    pub preemph_coeff: f32,
    /// 是否在加窗之前去除帧内直流分量。
    pub remove_dc_offset: bool,
    /// 窗函数类型。
    pub window_type: WindowType,
    /// 是否把帧长向上取整到最近的 2 的幂作为 FFT 长度。
    pub round_to_power_of_two: bool,
}

/// Kaldi 兼容的 Fbank 提取器。
///
/// 构造时会预先算好窗系数与 Mel 滤波器组并搬到 `device` 上，因此可以
/// 在线程间廉价克隆、被多个流复用。
#[derive(Clone, Debug)]
pub struct FbankExtractor {
    options: FbankOptions,
    device: Device,
    /// 每帧原始采样点数（未补零）。
    frame_length: usize,
    /// 相邻帧之间的采样点数。
    frame_shift: usize,
    /// 实际送入 FFT 的长度（2 的幂）。
    fft_size: usize,
    /// 参与 Mel 累加的 FFT bin 数，等于 `fft_size / 2`。
    num_fft_bins: usize,
    /// 窗系数，形状 `[frame_length]`。
    window: Tensor<1>,
    /// Mel 滤波器组，形状 `[num_fft_bins, num_mel_bins]`。
    mel_matrix: Tensor<2>,
}

impl FbankExtractor {
    /// 在 `device` 上按 `options` 构造提取器。
    ///
    /// # Errors
    ///
    /// 当帧长/帧移算出来为 0、Mel 滤波器少于 3 个、频率范围非法，
    /// 或 FFT 长度不是 2 的幂（Burn 的 `rfft` 目前只支持 2 的幂）时返回错误。
    pub fn new(options: FbankOptions, device: Device) -> Result<Self> {
        let sample_rate = options.sample_rate as f32;
        // 与 Kaldi 一致：先乘 0.001 再乘毫秒数，最后截断取整。
        let frame_length = (sample_rate * 0.001 * options.frame_length_ms) as usize;
        let frame_shift = (sample_rate * 0.001 * options.frame_shift_ms) as usize;
        if frame_length == 0 || frame_shift == 0 {
            bail!("非法的帧长/帧移: frame_length={frame_length}, frame_shift={frame_shift}");
        }
        if frame_length < 2 {
            bail!("帧长至少需要 2 个采样点，当前为 {frame_length}");
        }
        if options.num_mel_bins < 3 {
            bail!("Mel 滤波器至少需要 3 个，当前为 {}", options.num_mel_bins);
        }

        let fft_size = if options.round_to_power_of_two {
            frame_length.next_power_of_two()
        } else {
            frame_length
        };
        if !fft_size.is_power_of_two() {
            bail!("Burn 的 rfft 仅支持 2 的幂长度，当前 FFT 长度为 {fft_size}");
        }

        let num_fft_bins = fft_size / 2;
        let nyquist = 0.5 * sample_rate;
        let high_freq = if options.high_freq > 0.0 {
            options.high_freq
        } else {
            nyquist + options.high_freq
        };
        if options.low_freq < 0.0
            || options.low_freq >= nyquist
            || high_freq <= 0.0
            || high_freq > nyquist
            || high_freq <= options.low_freq
        {
            bail!(
                "非法的 Mel 频率范围: low_freq={}, high_freq={}, nyquist={nyquist}",
                options.low_freq,
                high_freq
            );
        }

        let window = Tensor::<1>::from_data(
            TensorData::new(
                options.window_type.coefficients(frame_length),
                [frame_length],
            ),
            &device,
        );
        let mel_matrix = Tensor::<2>::from_data(
            TensorData::new(
                mel_filterbank(&options, fft_size, num_fft_bins, high_freq),
                [num_fft_bins, options.num_mel_bins],
            ),
            &device,
        );

        Ok(Self {
            options,
            device,
            frame_length,
            frame_shift,
            fft_size,
            num_fft_bins,
            window,
            mel_matrix,
        })
    }

    /// 输出特征维度，即 Mel 滤波器个数。
    pub fn num_mel_bins(&self) -> usize {
        self.options.num_mel_bins
    }

    /// 特征所在的设备。
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 按 `snip_edges = true` 计算 `num_samples` 个采样点能产出的完整帧数。
    pub fn num_frames(&self, num_samples: usize) -> usize {
        if num_samples < self.frame_length {
            0
        } else {
            1 + (num_samples - self.frame_length) / self.frame_shift
        }
    }

    /// 对整段波形提取 Fbank 特征，返回 `[frames, num_mel_bins]`。
    ///
    /// `samples` 需已缩放到 int16 量级；长度不足一帧时返回 0 行张量。
    pub fn compute(&self, samples: &[f32]) -> Tensor<2> {
        self.compute_frames(samples, self.num_frames(samples.len()))
    }

    /// 对已经确认能切出 `num_frames` 帧的波形提取特征。
    ///
    /// 调用方需保证 `samples` 长度足以容纳 `num_frames` 帧，否则会触发下溢 panic。
    fn compute_frames(&self, samples: &[f32], num_frames: usize) -> Tensor<2> {
        let num_mel_bins = self.options.num_mel_bins;
        if num_frames == 0 {
            return Tensor::<2>::zeros([0, num_mel_bins], &self.device);
        }

        // [num_samples] -> [num_frames, frame_length]，窗口不重叠时会有数据复制。
        let input = Tensor::<1>::from_data(
            TensorData::new(samples.to_vec(), [samples.len()]),
            &self.device,
        );
        let frames = input.unfold::<2, _>(0, self.frame_length, self.frame_shift);

        // 去直流：减掉帧内均值，`[num_frames, 1]` 广播到整帧。
        let frames = if self.options.remove_dc_offset {
            let mean = frames.clone().mean_dim(1);
            frames - mean
        } else {
            frames
        };

        // 预加重：`out[i] = x[i] - coeff * x[i-1]`，其中 `x[-1]` 取 `x[0]`，
        // 这样第 0 个点自然得到 `(1 - coeff) * x[0]`，与 Kaldi 的手写特例一致。
        let frames = if self.options.preemph_coeff != 0.0 {
            let shifted = Tensor::cat(
                vec![
                    frames.clone().slice([0..num_frames, 0..1]),
                    frames
                        .clone()
                        .slice([0..num_frames, 0..self.frame_length - 1]),
                ],
                1,
            );
            frames - shifted.mul_scalar(self.options.preemph_coeff)
        } else {
            frames
        };

        // 加窗后再交给 rfft：长度不足时会自动在尾部零填充到 fft_size。
        let frames = frames * self.window.clone().unsqueeze::<2>();
        let (real, imag) = rfft(frames, 1, Some(self.fft_size));

        // 幂率谱，并丢弃 Nyquist bin 以对齐 Kaldi 的 `num_fft_bins`。
        let power = real.clone() * real + imag.clone() * imag;
        let power = power.slice([0..num_frames, 0..self.num_fft_bins]);

        // 三角滤波器组累加后取对数。
        power
            .matmul(self.mel_matrix.clone())
            .clamp_min(f32::EPSILON)
            .log()
    }
}

/// 流式 Fbank 提取器：按块追加波形，增量产出新出现的完整帧。
///
/// 内部只保留后续帧仍会用到的尾部样本，内存占用与帧长/帧移同阶。
pub struct FbankStream {
    extractor: FbankExtractor,
    /// 已接收但尚未被丢弃的样本，`buffer[0]` 对应全局索引 `buffer_offset`。
    buffer: Vec<f32>,
    /// `buffer[0]` 在整个波形中的全局样本索引。
    buffer_offset: usize,
    /// 已经产出的帧数。
    emitted_frames: usize,
}

impl FbankStream {
    /// 基于某个提取器创建流式会话，提取器会被廉价克隆复用。
    pub fn new(extractor: &FbankExtractor) -> Self {
        Self {
            extractor: extractor.clone(),
            buffer: Vec::new(),
            buffer_offset: 0,
            emitted_frames: 0,
        }
    }

    /// 重置状态，便于把同一个流复用到下一段音频。
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.buffer_offset = 0;
        self.emitted_frames = 0;
    }

    /// 追加一块波形，返回本次新产出的 `[frames, num_mel_bins]` 特征。
    ///
    /// 产出帧数可能为 0（当累积样本还凑不满一帧时）。
    pub fn push(&mut self, samples: &[f32]) -> Tensor<2> {
        self.buffer.extend_from_slice(samples);
        self.drain_ready_frames()
    }

    /// 冲刷当前块，产出所有已经能完整形成的帧。
    ///
    /// 在 `snip_edges = true` 下截断策略不依赖输入是否结束，因此本方法
    /// 与 [`FbankStream::push`] 的语义一致，仅用于让调用方表达“输入结束”。
    pub fn finish(&mut self) -> Tensor<2> {
        self.drain_ready_frames()
    }

    /// 计算并返回尚未产出的完整帧，同时丢弃不再需要的样本。
    fn drain_ready_frames(&mut self) -> Tensor<2> {
        let total_samples = self.buffer_offset + self.buffer.len();
        let total_frames = self.extractor.num_frames(total_samples);
        let new_frames = total_frames.saturating_sub(self.emitted_frames);
        if new_frames == 0 {
            return Tensor::<2>::zeros([0, self.extractor.num_mel_bins()], self.extractor.device());
        }

        let frame_shift = self.extractor.frame_shift;
        let frame_length = self.extractor.frame_length;
        // 上一轮已经把 `emitted_frames * frame_shift` 之前的样本丢掉了，
        // 所以待处理区域的起点就在缓冲区开头。
        let start = self.emitted_frames * frame_shift - self.buffer_offset;
        // 恰好够切出 `new_frames` 帧，让 unfold 的窗口数与预期一致。
        let needed = (new_frames - 1) * frame_shift + frame_length;
        let features = self.extractor.compute(&self.buffer[start..start + needed]);

        // 丢弃不会再被任何未来帧使用的样本。
        let consumed = total_frames * frame_shift - self.buffer_offset;
        if consumed > 0 {
            self.buffer.drain(..consumed);
            self.buffer_offset += consumed;
        }
        self.emitted_frames = total_frames;

        features
    }
}

/// HTK Mel 尺度：`1127 * ln(1 + f / 700)`。
fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// 按 Kaldi 的三角滤波器组规则生成 `[num_fft_bins, num_mel_bins]` 行主序权重。
fn mel_filterbank(
    options: &FbankOptions,
    fft_size: usize,
    num_fft_bins: usize,
    high_freq: f32,
) -> Vec<f32> {
    let fft_bin_width = options.sample_rate as f32 / fft_size as f32;
    let mel_low = mel_scale(options.low_freq);
    let mel_high = mel_scale(high_freq);
    let num_bins = options.num_mel_bins;
    // 除以 num_bins + 1 是为了让首尾滤波器也能覆盖到频带边缘。
    let mel_delta = (mel_high - mel_low) / (num_bins + 1) as f32;

    let mut matrix = vec![0.0_f32; num_fft_bins * num_bins];
    for bin in 0..num_bins {
        let left = mel_low + bin as f32 * mel_delta;
        let center = mel_low + (bin + 1) as f32 * mel_delta;
        let right = mel_low + (bin + 2) as f32 * mel_delta;
        for i in 0..num_fft_bins {
            let mel = mel_scale(fft_bin_width * i as f32);
            // Kaldi 使用严格不等号，三角权重在边界处为 0。
            if mel > left && mel < right {
                let weight = if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                };
                matrix[i * num_bins + bin] = weight;
            }
        }
    }
    matrix
}

/// 把归一化到 `[-1, 1]` 的浮点 PCM 缩放到 Kaldi 期望的 int16 量级。
pub fn to_kaldi_scale(samples: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use kaldi_fbank_rust_kautism::{
        FbankOptions as KaldiFbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
    };

    use super::*;

    /// 生成一段确定性的伪随机波形，量级与 Kaldi 期望的 int16 缩放一致。
    fn test_waveform(len: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..len)
            .map(|i| {
                // 线性同余发生器，保证任何平台都能复现同一段波形。
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
                let tone = (i as f32 * 0.05).sin() * 0.4;
                0.5 * 32768.0 * (tone + noise)
            })
            .collect()
    }

    /// 用 Kaldi 参考实现提取整段波形的 fbank 帧。
    fn kaldi_reference(waveform: &[f32], window: &CStr, num_bins: i32) -> Vec<Vec<f32>> {
        let options = KaldiFbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: 16_000.0,
                window_type: window.as_ptr(),
                dither: 0.0,
                frame_shift_ms: 10.0,
                frame_length_ms: 25.0,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };
        let mut fbank = OnlineFbank::new(options);
        fbank.accept_waveform(16_000.0, waveform);
        fbank.input_finished();
        (0..fbank.num_ready_frames() as usize)
            .map(|i| {
                fbank
                    .get_frame(i as i32)
                    .expect("kaldi 参考帧应当存在")
                    .to_vec()
            })
            .collect()
    }

    /// 构造与本库前端一致的 Fbank 参数。
    fn options(window_type: WindowType, num_mel_bins: usize) -> FbankOptions {
        FbankOptions {
            sample_rate: 16_000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            num_mel_bins,
            low_freq: 20.0,
            high_freq: 0.0,
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type,
            round_to_power_of_two: true,
        }
    }

    /// 把张量摊平成 CPU `Vec<f32>`。
    fn flatten(tensor: Tensor<2>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("特征张量应可读取为 f32")
    }

    /// 逐元素比较并返回（最大绝对误差, RMS 误差）。
    fn diff_stats(got: &[f32], expected: &[f32]) -> (f32, f64) {
        assert_eq!(got.len(), expected.len(), "特征元素总数不一致");
        let mut max_abs = 0.0_f32;
        let mut sum_sq = 0.0_f64;
        for (lhs, rhs) in got.iter().zip(expected) {
            let diff = (lhs - rhs).abs();
            max_abs = max_abs.max(diff);
            sum_sq += (diff as f64) * (diff as f64);
        }
        (max_abs, (sum_sq / got.len() as f64).sqrt())
    }

    /// 与 Kaldi 参考实现在给定窗函数下逐帧对拍。
    fn assert_matches_kaldi(window_type: WindowType, kaldi_window: &CStr, num_mel_bins: usize) {
        let waveform = test_waveform(40_000);
        let reference = kaldi_reference(&waveform, kaldi_window, num_mel_bins as i32);

        let extractor =
            FbankExtractor::new(options(window_type, num_mel_bins), Device::flex()).unwrap();
        let features = extractor.compute(&waveform);
        let [frames, bins] = features.dims();
        assert_eq!(frames, reference.len(), "帧数应与 Kaldi 一致");
        assert_eq!(bins, num_mel_bins);

        let expected = reference.concat();
        let (max_abs, rms) = diff_stats(&flatten(features), &expected);
        eprintln!("{window_type:?}: frames={frames} max_abs={max_abs:.3e} rms={rms:.3e}");
        assert!(
            max_abs < 1e-3,
            "{window_type:?} 最大绝对误差过大: {max_abs}"
        );
    }

    #[test]
    fn matches_kaldi_hamming_window() {
        assert_matches_kaldi(WindowType::Hamming, c"hamming", 80);
    }

    #[test]
    fn matches_kaldi_povey_window() {
        assert_matches_kaldi(WindowType::Povey, c"povey", 80);
    }

    #[test]
    fn frame_count_matches_kaldi_rule() {
        let extractor = FbankExtractor::new(options(WindowType::Hamming, 80), Device::flex())
            .expect("提取器构造应当成功");
        // snip_edges = true 时帧数为 1 + (n - 400) / 160，不足一帧则为 0。
        for samples in [0usize, 1, 399, 400, 401, 559, 560, 561, 40_000] {
            let expected = if samples < 400 {
                0
            } else {
                1 + (samples - 400) / 160
            };
            assert_eq!(extractor.num_frames(samples), expected, "n={samples}");
        }
    }

    #[test]
    fn streaming_matches_offline() {
        let waveform = test_waveform(40_000);
        let extractor = FbankExtractor::new(options(WindowType::Hamming, 80), Device::flex())
            .expect("提取器构造应当成功");
        let offline = flatten(extractor.compute(&waveform));

        // 用不规则分块推流，覆盖“块边界落在帧中间”的情况。
        let mut stream = FbankStream::new(&extractor);
        let mut streamed = Vec::new();
        for chunk in waveform.chunks(1234) {
            let frames = stream.push(chunk);
            if frames.dims()[0] > 0 {
                streamed.extend(flatten(frames));
            }
        }
        let tail = stream.finish();
        if tail.dims()[0] > 0 {
            streamed.extend(flatten(tail));
        }

        assert_eq!(streamed.len(), offline.len(), "流式与离线帧数应一致");
        for (lhs, rhs) in streamed.iter().zip(&offline) {
            assert!(
                (lhs - rhs).abs() < 1e-4,
                "流式结果应与离线逐位一致: {lhs} vs {rhs}"
            );
        }
    }
}
