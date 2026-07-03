#[derive(Debug, Clone)]
pub struct VadOptions {
    pub threshold: f32,
    pub min_speech_ms: u64,
    pub min_silence_ms: u64,
    pub max_segment_ms: u64,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct DurationMs(pub u64);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TimeRange {
    pub start: DurationMs,
    pub end: DurationMs,
}

impl TimeRange {
    pub fn new(start: DurationMs, end: DurationMs) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VadSegment {
    pub range: TimeRange,
    pub probability: f32,
}
