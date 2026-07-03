from __future__ import annotations

from collections.abc import Sequence
from typing import Optional


class VadOptions:
    """VAD segmentation options."""

    threshold: float
    """Speech/noise posterior threshold."""

    min_speech_ms: int
    """Drop detected speech shorter than this duration."""

    min_silence_ms: int
    """Silence duration used to close a speech segment."""

    max_segment_ms: int
    """Maximum segment duration. Use 0 to disable long-segment splitting."""

    pad_ms: int
    """Reserved segment padding in milliseconds."""

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
    """Segment confidence or threshold value."""


class VadTiming:
    """Timing breakdown for a timed VAD run."""

    frontend_seconds: float
    """Feature extraction time."""

    forward_seconds: float
    """FSMN forward time."""

    segmenter_seconds: float
    """Post-processing time."""


class VadDetection:
    """Timed VAD detection result."""

    segments: list[VadSegment]
    """Detected speech segments."""

    frame_scores: list[list[float]]
    """Per-frame silence posterior scores."""

    timing: VadTiming
    """Timing breakdown."""


class FsmnVadStream:
    """Stateful streaming FSMN VAD session."""

    def push(self, samples: Sequence[float], sample_rate: int) -> list[VadSegment]:
        """Push one audio chunk and return newly finalized speech segments."""
        ...

    def finish(self) -> list[VadSegment]:
        """Finalize the stream and return remaining speech segments."""
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
        """Download from ModelScope cache and load the FSMN VAD model.

        Defaults to repo_id "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        and revision "master".
        """
        ...

    def detect(
        self,
        samples: Sequence[float],
        sample_rate: int,
        options: Optional[VadOptions] = None,
    ) -> list[VadSegment]:
        """Run offline VAD on normalized float PCM samples."""
        ...

    def detect_with_timing(
        self,
        samples: Sequence[float],
        sample_rate: int,
        options: Optional[VadOptions] = None,
    ) -> VadDetection:
        """Run offline VAD and return segments, frame scores, and timing."""
        ...

    def new_stream(self, options: Optional[VadOptions] = None) -> FsmnVadStream:
        """Create a stateful streaming VAD session."""
        ...
