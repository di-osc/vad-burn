//! VAD 推理超参。波形和时间轴类型来自 asr-data。

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use anyhow::{Result, bail};
use asr_data::{
    Audio, AudioActivity, AudioChannel, StreamingResampler, TimeRange, TimeSpan, Waveform,
};

/// FSMN VAD 写入 timeline 时使用的 prediction source。
pub const FSMN_VAD_SOURCE: &str = "fsmn-vad";
/// FireRed VAD 写入 timeline 时使用的 prediction source。
pub const FIRERED_VAD_SOURCE: &str = "firered-vad";
/// 检测前端要求的采样率。
pub const VAD_SAMPLE_RATE: u32 = 16_000;

/// 语音活动检测的切段参数。
#[derive(Debug, Clone)]
pub struct VadOptions {
    /// 判定为语音的概率阈值。
    pub threshold: f32,
    /// 短于该时长的语音段会被丢弃，单位毫秒。
    pub min_speech_ms: u64,
    /// 结束一段语音所需的静音时长，单位毫秒。
    pub min_silence_ms: u64,
    /// 单段语音的最大时长，单位毫秒；`0` 表示不切分。
    pub max_segment_ms: u64,
    /// 预留的端点填充，单位毫秒；当前仅为兼容字段。
    pub pad_ms: u64,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            min_speech_ms: 250,
            min_silence_ms: 500,
            max_segment_ms: 30_000,
            pad_ms: 0,
        }
    }
}

/// 构造一条 `speech` 活动预测 span。
pub fn speech_span(
    start_ms: usize,
    end_ms: usize,
    confidence: f32,
    source: impl Into<String>,
) -> TimeSpan {
    AudioActivity::new()
        .with_event("speech")
        .with_confidence(confidence)
        .into_span(TimeRange::new(start_ms, end_ms), source)
}

/// 取出 activity span 上的置信度；非 activity 时为 `0.0`。
pub fn span_confidence(span: &TimeSpan) -> f32 {
    span.annotation.confidence().unwrap_or(0.0)
}

/// 把波形重采样到 16 kHz。已经是目标采样率时克隆。
///
/// # Errors
///
/// 采样率为 0 或重采样失败时返回错误。
pub fn prepare_16k(waveform: &Waveform) -> Result<Waveform> {
    if waveform.sample_rate == 0 {
        bail!("sample rate must be greater than zero");
    }
    if waveform.sample_rate == VAD_SAMPLE_RATE {
        return Ok(waveform.clone());
    }
    waveform.resample(VAD_SAMPLE_RATE)
}

/// 按声道检测并把结果写进 [`Audio`] 的 prediction timeline。
///
/// # Errors
///
/// 解码、重采样、检测或写入标注失败时返回错误。
pub fn annotate_audio<F>(audio: &mut Audio, mut detect: F) -> Result<()>
where
    F: FnMut(&Waveform) -> Result<Vec<TimeSpan>>,
{
    audio.annotate_activity(|_channel, waveform| {
        let prepared = prepare_16k(waveform)?;
        detect(&prepared)
    })
}

/// 把一块流式波形准备成模型需要的 16 kHz。
///
/// 已经是 16 kHz 时返回 `None`，调用方继续使用原始波形。
/// `is_final` 时冲刷该声道的有状态重采样器。
///
/// # Errors
///
/// 采样率为 0 或重采样失败时返回错误。
pub fn prepare_stream_16k(
    waveform: &Waveform,
    is_final: bool,
    resamplers: &mut HashMap<AudioChannel, StreamingResampler>,
    channel: AudioChannel,
) -> Result<Option<Waveform>> {
    if waveform.sample_rate == 0 {
        bail!("sample rate must be greater than zero");
    }
    if waveform.sample_rate == VAD_SAMPLE_RATE {
        return Ok(None);
    }
    let resampler = match resamplers.entry(channel) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(StreamingResampler::new(
            waveform.sample_rate,
            VAD_SAMPLE_RATE,
            waveform.channels,
        )?),
    };
    let samples = resampler.process(&waveform.samples, waveform.frame_count(), is_final)?;
    Ok(Some(Waveform::new(samples, VAD_SAMPLE_RATE)))
}

/// 解析 timeline 键为声道标识。
///
/// # Errors
///
/// 名称既不是 `mono` / `left` / `right`，也不能解析为声道下标时返回错误。
pub fn parse_audio_channel(name: &str) -> Result<AudioChannel> {
    match name {
        "mono" => Ok(AudioChannel::Mono),
        "left" => Ok(AudioChannel::Left),
        "right" => Ok(AudioChannel::Right),
        other => match other.parse::<u16>() {
            Ok(index) => Ok(AudioChannel::from_index(index)),
            Err(_) => {
                bail!("unsupported audio channel {name:?}; expected mono, left, right, or an index")
            }
        },
    }
}
