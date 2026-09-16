"""Type stubs for vad_burn.

Audio inputs are mono, normalized float PCM samples at 16 kHz unless a method
explicitly says otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from asr_data import Audio, AudioChunk

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
    """Detected speech activity span aligned with asr-data TimeSpan fields."""

    id: str
    """Generated span id."""

    start_ms: int
    """Segment start time in milliseconds."""

    end_ms: int
    """Segment end time in milliseconds."""

    confidence: float
    """Activity confidence written onto AudioActivity."""

    event: Optional[str]
    """Activity event name, usually \"speech\"."""

    source: Optional[str]
    """Prediction source, for example \"fsmn-vad\" or \"firered-vad\"."""

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

class FsmnVadSession:
    """Stateful streaming FSMN VAD session.

    A session owns mutable decoding state and should be driven sequentially by
    one audio stream. Do not call push(), finish(), or reset() concurrently on
    the same session. For parallel streaming sessions, create one session per
    audio stream with FsmnVadModel.new_session().
    """

    def push(self, samples: Sequence[float], sample_rate: int) -> list[VadSegment]:
        """Push one mono 16 kHz audio chunk and return newly finalized segments.

        Samples must be normalized float PCM values, usually in the range
        [-1.0, 1.0]. Call finish() once after the final chunk to flush any
        pending speech segment.
        """
        ...

    def annotate(self, chunk: AudioChunk) -> list[VadSegment]:
        """Annotate one AudioChunk and write new activity onto the parent stream.

        Iterate the AudioStream yourself and call this per chunk so intermediate
        timeline predictions remain available. Sample rate is a model property
        and is resampled to 16 kHz internally. Each channel uses independent
        decoder state; the last chunk flushes via finish().
        """
        ...

    def finish(self) -> list[VadSegment]:
        """Flush the final pending chunk, return remaining segments, and reset state."""
        ...

    def reset(self) -> None:
        """Clear session state and cached frames."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class FsmnVadModel:
    """FSMN VAD model.

    The loaded model can be shared across threads for offline detect() calls
    and for creating independent streaming sessions. Each new_session() call
    returns a separate stateful session; the session itself is not a shared
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

    def annotate(
        self,
        audio: Audio,
        options: Optional[VadOptions] = None,
    ) -> Audio:
        """Detect speech per channel and write AudioActivity predictions onto audio."""
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

    def new_session(self, options: Optional[VadOptions] = None) -> FsmnVadSession:
        """Create a stateful streaming VAD session from this loaded model.

        Use one session per audio stream when running multiple streams in
        parallel.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class FireRedVadSession:
    """Stateful streaming FireRedVAD session.

    A session owns mutable FSMN caches and streaming post-processing state. Do
    not call push(), finish(), or reset() concurrently on the same session. For
    parallel streaming sessions, create one session per audio stream with
    FireRedVadModel.new_session().
    """

    def push(self, samples: Sequence[float], sample_rate: int) -> list[VadSegment]:
        """Push one mono 16 kHz audio chunk and return newly finalized segments.

        Samples must be normalized float PCM values, usually in the range
        [-1.0, 1.0]. FireRedVadModel.from_modelscope() loads both official
        VAD and Stream-VAD weights; sessions use the Stream-VAD weights.
        """
        ...

    def annotate(self, chunk: AudioChunk) -> list[VadSegment]:
        """Annotate one AudioChunk and write new activity onto the parent stream.

        Iterate the AudioStream yourself and call this per chunk so intermediate
        timeline predictions remain available. Sample rate is a model property
        and is resampled to 16 kHz internally. Each channel uses independent
        decoder state; the last chunk flushes via finish().
        """
        ...

    def finish(self) -> list[VadSegment]:
        """Flush an open speech segment, return remaining segments, and reset state."""
        ...

    def reset(self) -> None:
        """Clear session state, FSMN caches, and cached frame scores."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class FireRedVadModel:
    """FireRedVAD model implemented with Burn Flex.

    The loaded model can be shared across threads for concurrent offline
    detect() calls and for creating independent streaming sessions.
    from_modelscope() loads both official VAD and Stream-VAD weights. When
    loading from disk, pass the official repository root containing VAD/ and
    Stream-VAD/ to make detect() and new_session() use their matching weights.
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
        object. detect() uses VAD; new_session() uses Stream-VAD.
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

    def annotate(
        self,
        audio: Audio,
        options: Optional[VadOptions] = None,
    ) -> Audio:
        """Detect speech per channel and write AudioActivity predictions onto audio."""
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

    def new_session(self, options: Optional[VadOptions] = None) -> FireRedVadSession:
        """Create a stateful streaming FireRedVAD session from this loaded model.

        The session uses the Stream-VAD weights when the model was loaded from
        ModelScope or from an official local repository root. Use one session
        per audio stream when running multiple streams in parallel.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
