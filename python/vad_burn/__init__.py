from asr_data import Audio, AudioChunk
from asr_data.annotation import AudioActivity

from .vad_burn import (
    FireRedVadDetection,
    FireRedVadModel,
    FireRedVadSession,
    FireRedVadTiming,
    FsmnVadModel,
    FsmnVadSession,
    VadDetection,
    VadOptions,
    VadSegment,
    VadTiming,
)

__all__ = [
    "FireRedVadDetection",
    "FireRedVadModel",
    "FireRedVadSession",
    "FireRedVadTiming",
    "FsmnVadModel",
    "FsmnVadSession",
    "VadDetection",
    "VadOptions",
    "VadSegment",
    "VadTiming",
]

# 模型前端要求的采样率；重采样发生在 annotate 内部，而不是创建流时。
_VAD_SAMPLE_RATE = 16000


def _write_spans(timeline, spans) -> None:
    """把 VAD 段写成 timeline 上的 prediction activity。"""
    for span in spans:
        timeline.annotate_span(
            span.start_ms,
            span.end_ms,
            AudioActivity(event=span.event, confidence=span.confidence),
            is_reference=False,
            source=span.source,
        )


def _annotate(self, audio: Audio, options: VadOptions | None = None) -> Audio:
    """按声道检测语音活动，并写入对应 timeline 的 prediction。

    每个声道单独推理，不会把多声道平均成单声道。采样率不是 16 kHz 时会先重采样。

    Args:
        audio: asr-data 的 Audio 文档。
        options: 可选切段参数；缺省使用模型默认值。

    Returns:
        写入 activity 标注后的同一份 `audio`。
    """
    for timeline in audio.timelines.values():
        wave = timeline.as_waveform()
        if wave.sample_rate != _VAD_SAMPLE_RATE:
            wave = wave.resample(_VAD_SAMPLE_RATE)
        _write_spans(
            timeline, self.detect(list(wave.samples), wave.sample_rate, options)
        )
    return audio


def _annotate_chunk(self, chunk: AudioChunk) -> list:
    """标注一块 AudioChunk，并把新产生的 activity 写进父流 timeline。

    每个声道使用独立推理状态。源采样率由模型在内部重采样到 16 kHz。
    最后一块会 flush 该声道。返回本块新写出的 span，便于立刻使用中间结果。

    Args:
        chunk: asr-data 的 AudioChunk；必须来自正在迭代的 AudioStream。

    Returns:
        本块新写入的 activity span 列表。
    """
    spans_all = []
    for name, timeline in chunk.timelines.items():
        wave = chunk.as_waveform(name)
        spans = self.annotate_waveform(
            list(wave.samples), int(wave.sample_rate), name, chunk.is_final
        )
        _write_spans(timeline, spans)
        spans_all.extend(spans)
    return spans_all


FsmnVadModel.annotate = _annotate
FireRedVadModel.annotate = _annotate
FsmnVadSession.annotate = _annotate_chunk
FireRedVadSession.annotate = _annotate_chunk
