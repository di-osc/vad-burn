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

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class VadSegment:
    """Detected speech segment."""

    start_ms: int
    """Segment start time in milliseconds."""

    end_ms: int
    """Segment end time in milliseconds."""

    probability: float
    """Segment score reported by the post-processor."""

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class VadTiming:
    """Timing breakdown for a timed VAD run."""

    frontend_seconds: float
    """Feature extraction time in seconds."""

    forward_seconds: float
    """FSMN forward pass time in seconds."""

    segmenter_seconds: float
    """Segmentation post-processing time in seconds."""

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class VadDetection:
    """Timed VAD detection result."""

    segments: list[VadSegment]
    """Detected speech segments."""

    frame_scores: list[list[float]]
    """Per-frame posterior scores produced by the model."""

    timing: VadTiming
    """Timing breakdown."""

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class FireRedVadTiming:
    """Timing breakdown for a timed FireRedVAD run."""

    frontend_seconds: float
    """Feature extraction time in seconds."""

    forward_seconds: float
    """Burn forward pass time in seconds."""

    postprocess_seconds: float
    """Segmentation post-processing time in seconds."""

    frames: int
    """Number of acoustic frames processed."""

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class FireRedVadDetection:
    """Timed FireRedVAD detection result."""

    segments: list[VadSegment]
    """Detected speech segments."""

    frame_scores: list[float]
    """Per-frame speech probabilities."""

    timing: FireRedVadTiming
    """Timing breakdown."""

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class FsmnVadStream:
    """Stateful streaming FSMN VAD session.

    A stream owns mutable decoding state and should be driven sequentially by
    one audio stream. Do not call push(), finish(), or reset() concurrently on
    the same stream. For parallel streaming sessions, create one stream per
    audio stream with FsmnVadModel.new_stream().
    """

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

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class FsmnVadModel:
    """FSMN VAD model.

    The loaded model can be shared across threads for offline detect() calls
    and for creating independent streaming sessions. Each new_stream() call
    returns a separate stateful stream; the stream itself is not a shared
    concurrent object.
    """

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

        sample_rate must be 16000. This method releases the Python GIL while
        running Rust inference.
        """
        ...

    def detect_with_timing(
        self,
        samples: Sequence[float],
        sample_rate: int,
        options: Optional[VadOptions] = None,
    ) -> VadDetection:
        """Run offline VAD and return segments, model frame scores, and timing.

        This method releases the Python GIL while running Rust inference.
        """
        ...

    def new_stream(self, options: Optional[VadOptions] = None) -> FsmnVadStream:
        """Create a stateful streaming VAD session from this loaded model.

        Use one stream per audio stream when running multiple streams in
        parallel.
        """
        ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class FireRedVadStream:
    """Stateful streaming FireRedVAD session.

    A stream owns mutable FSMN caches and streaming post-processing state. Do
    not call push(), finish(), or reset() concurrently on the same stream. For
    parallel streaming sessions, create one stream per audio stream with
    FireRedVadModel.new_stream().
    """

    def push(self, samples: Sequence[float], sample_rate: int) -> list[VadSegment]:
        """Push one mono 16 kHz audio chunk and return newly finalized segments.

        Samples must be normalized float PCM values, usually in the range
        [-1.0, 1.0]. FireRedVadModel.from_modelscope() loads both official
        VAD and Stream-VAD weights; streams use the Stream-VAD weights.
        """
        ...

    def finish(self) -> list[VadSegment]:
        """Flush an open speech segment, return remaining segments, and reset state."""
        ...

    def reset(self) -> None:
        """Clear stream state, FSMN caches, and cached frame scores."""
        ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class FireRedVadModel:
    """FireRedVAD model implemented with Burn Flex.

    The loaded model can be shared across threads for concurrent offline
    detect() calls and for creating independent streaming sessions.
    from_modelscope() loads both official VAD and Stream-VAD weights. When
    loading from disk, pass the official repository root containing VAD/ and
    Stream-VAD/ to make detect() and new_stream() use their matching weights.
    """

    def __init__(self, model_dir: str) -> None:
        """Load FireRedVAD from a local model directory.

        Prefer the official repository root containing VAD/ and Stream-VAD/.
        A single directory containing model.pth.tar and cmvn.ark is accepted
        for compatibility and is used for both offline and streaming paths.
        """
        ...

    @staticmethod
    def from_pretrained(model_dir: str) -> FireRedVadModel:
        """Load FireRedVAD from a local model directory.

        Prefer the official repository root containing VAD/ and Stream-VAD/.
        A single directory containing model.pth.tar and cmvn.ark is accepted
        for compatibility and is used for both offline and streaming paths.
        """
        ...

    @staticmethod
    def from_modelscope(
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> FireRedVadModel:
        """Download through ModelScope cache and load FireRedVAD.

        Defaults to repo_id "xukaituo/FireRedVAD" and revision "master".
        Both VAD and Stream-VAD subdirectories are loaded into one model
        object. detect() uses VAD; new_stream() uses Stream-VAD.
        """
        ...

    def detect(
        self,
        samples: Sequence[float],
        sample_rate: int,
        options: Optional[VadOptions] = None,
    ) -> list[VadSegment]:
        """Run offline FireRedVAD on mono normalized float PCM samples.

        sample_rate must be 16000. This method releases the Python GIL while
        running Rust inference.
        """
        ...

    def detect_with_timing(
        self,
        samples: Sequence[float],
        sample_rate: int,
        options: Optional[VadOptions] = None,
    ) -> FireRedVadDetection:
        """Run offline FireRedVAD and return segments, frame scores, and timing.

        This method releases the Python GIL while running Rust inference.
        """
        ...

    def new_stream(self, options: Optional[VadOptions] = None) -> FireRedVadStream:
        """Create a stateful streaming FireRedVAD session from this loaded model.

        The stream uses the Stream-VAD weights when the model was loaded from
        ModelScope or from an official local repository root. Use one stream
        per audio stream when running multiple streams in parallel.
        """
        ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...
