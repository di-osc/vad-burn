"""Type stubs for vad_burn.

Audio inputs are mono, normalized float PCM samples at 16 kHz unless a method
explicitly says otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional


class VadOptions:
    """Segmentation options used by both offline and streaming VAD."""

    threshold: float
    """Speech threshold used by the FSMN VAD post-processor."""

    min_speech_ms: int
    """Drop speech segments shorter than this duration, in milliseconds."""

    min_silence_ms: int
    """Silence duration required to close a speech segment, in milliseconds."""

    max_segment_ms: int
    """Maximum speech segment duration in milliseconds. Use 0 to disable splitting."""

    pad_ms: int
    """Reserved padding value in milliseconds; currently kept for API compatibility."""

    def __init__(
        self,
        threshold: float = 0.6,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        max_segment_ms: int = 30000,
        pad_ms: int = 0,
    ) -> None: ...


class VadSegment:
    """Detected speech segment."""

    start_ms: int
    """Segment start time in milliseconds."""

    end_ms: int
    """Segment end time in milliseconds."""

    probability: float
    """Segment score reported by the post-processor."""


class VadTiming:
    """Timing breakdown for a timed VAD run."""

    frontend_seconds: float
    """Feature extraction time in seconds."""

    forward_seconds: float
    """FSMN forward pass time in seconds."""

    segmenter_seconds: float
    """Segmentation post-processing time in seconds."""


class VadDetection:
    """Timed VAD detection result."""

    segments: list[VadSegment]
    """Detected speech segments."""

    frame_scores: list[list[float]]
    """Per-frame posterior scores produced by the model."""

    timing: VadTiming
    """Timing breakdown."""


class FsmnVadStream:
    """Stateful streaming FSMN VAD session."""

    def push(self, samples: Sequence[float], sample_rate: int) -> list[VadSegment]:
        """Push one mono 16 kHz audio chunk and return newly finalized segments.

        Samples must be normalized float PCM values, usually in the range
        [-1.0, 1.0]. Call finish() once after the final chunk to flush any
        pending speech segment.
        """
        ...

    def finish(self) -> list[VadSegment]:
        """Flush the final pending chunk, return remaining segments, and reset state."""
        ...

    def reset(self) -> None:
        """Clear stream state and cached frames."""
        ...


class FsmnVadModel:
    """FSMN VAD model."""

    def __init__(self, model_dir: str) -> None:
        """Load a local model directory containing model.pt and am.mvn."""
        ...

    @staticmethod
    def from_pretrained(model_dir: str) -> FsmnVadModel:
        """Load a local model directory containing model.pt and am.mvn."""
        ...

    @staticmethod
    def from_modelscope(
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> FsmnVadModel:
        """Download through ModelScope cache and load the FSMN VAD model.

        Defaults to repo_id "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        and revision "master". Existing cached files are reused automatically.
        """
        ...

    def detect(
        self,
        samples: Sequence[float],
        sample_rate: int,
        options: Optional[VadOptions] = None,
    ) -> list[VadSegment]:
        """Run offline VAD on mono normalized float PCM samples.

        sample_rate must be 16000.
        """
        ...

    def detect_with_timing(
        self,
        samples: Sequence[float],
        sample_rate: int,
        options: Optional[VadOptions] = None,
    ) -> VadDetection:
        """Run offline VAD and return segments, model frame scores, and timing."""
        ...

    def new_stream(self, options: Optional[VadOptions] = None) -> FsmnVadStream:
        """Create a stateful streaming VAD session from this loaded model."""
        ...
