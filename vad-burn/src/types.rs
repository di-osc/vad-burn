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

#[derive(Debug, Clone, PartialEq)]
pub struct Waveform {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Waveform {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new_with_channels(samples, sample_rate, 1)
    }

    pub fn new_with_channels(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / self.sample_rate as f64 / self.channels as f64
    }

    pub fn duration_ms(&self) -> f64 {
        self.duration_seconds() * 1000.0
    }

    pub fn slice_ms(&self, start_ms: u64, end_ms: u64) -> Self {
        if self.sample_rate == 0 || self.channels == 0 || end_ms <= start_ms {
            return Self::new_with_channels(Vec::new(), self.sample_rate, self.channels);
        }

        let channels = self.channels as usize;
        let start_frame = ms_to_frame(start_ms, self.sample_rate);
        let end_frame = ms_to_frame(end_ms, self.sample_rate);
        let start = start_frame.saturating_mul(channels).min(self.samples.len());
        let end = end_frame
            .saturating_mul(channels)
            .min(self.samples.len())
            .max(start);

        Self::new_with_channels(
            self.samples[start..end].to_vec(),
            self.sample_rate,
            self.channels,
        )
    }
}

fn ms_to_frame(ms: u64, sample_rate: u32) -> usize {
    ((ms as u128 * sample_rate as u128) / 1000) as usize
}
