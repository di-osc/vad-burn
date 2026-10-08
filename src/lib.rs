mod fbank;
pub mod firered;
pub mod fsmn;
#[cfg(test)]
mod golden;
#[cfg(feature = "python")]
mod py;
mod types;

pub use asr_data::{
    Audio, AudioActivity, AudioChannel, AudioChunk, AudioStream, TimeRange, TimeSpan, Waveform,
};
// 重新导出推理设备类型，方便调用方为 `from_pretrained_on_device` 选择后端。
pub use burn::tensor::{Device, DeviceKind};
pub use firered::{
    DEFAULT_FIRERED_MODELSCOPE_REPO_ID, DEFAULT_FIRERED_MODELSCOPE_REVISION, FireRedVadDetection,
    FireRedVadModel, FireRedVadSession, FireRedVadTiming,
};
pub use fsmn::{
    DEFAULT_MODELSCOPE_REPO_ID, DEFAULT_MODELSCOPE_REVISION, FeatureTensor, FsmnForwardTiming,
    FsmnVadDetection, FsmnVadModel, FsmnVadSession, FsmnVadTiming,
};
pub use types::{
    FIRERED_VAD_SOURCE, FSMN_VAD_SOURCE, VAD_SAMPLE_RATE, VadOptions, annotate_audio,
    parse_audio_channel, prepare_16k, prepare_stream_16k, span_confidence, speech_span,
};
