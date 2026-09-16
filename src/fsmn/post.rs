use super::e2e::{E2EVadConfig, E2EVadModel};
use crate::{FSMN_VAD_SOURCE, TimeSpan, VadOptions, Waveform, span_confidence, speech_span};
use anyhow::Result;

pub const FRAME_SHIFT_SAMPLES: usize = 160;
pub const FRAME_LENGTH_SAMPLES: usize = 400;
const FEAT_CHUNK_SIZE: usize = 6000;
const REFINE_SILENCE_MS: [u64; 2] = [500, 300];

#[derive(Debug, Clone, Copy, Default)]
pub struct FsmnVadPostProcessor;

impl FsmnVadPostProcessor {
    pub fn segments_from_frame_scores<F>(
        &self,
        waveform: &Waveform,
        frame_scores: &[Vec<f32>],
        options: &VadOptions,
        mut refine_long_segment: F,
    ) -> Result<Vec<TimeSpan>>
    where
        F: FnMut(&Waveform, &TimeSpan, &VadOptions, u64) -> Result<Vec<TimeSpan>>,
    {
        let mut e2e = build_e2e_model(options);
        let max_end_sil = options.min_silence_ms as i32;
        let mut ms_segments = Vec::new();
        let total_frames = frame_scores.len();
        let mut frame_offset = 0usize;

        while frame_offset < total_frames {
            let step = FEAT_CHUNK_SIZE.min(total_frames - frame_offset);
            let is_final = frame_offset + step >= total_frames;
            let wave_start = frame_offset * FRAME_SHIFT_SAMPLES;
            let wave_end = if is_final {
                waveform.samples.len()
            } else {
                ((frame_offset + step - 1) * FRAME_SHIFT_SAMPLES + FRAME_LENGTH_SAMPLES)
                    .min(waveform.samples.len())
            };
            ms_segments.extend(e2e.detect_chunk(
                &frame_scores[frame_offset..frame_offset + step],
                &waveform.samples[wave_start..wave_end],
                is_final,
                max_end_sil,
            ));
            frame_offset += step;
        }

        let raw_segments = ms_segments
            .into_iter()
            .map(|(start, end)| {
                speech_span(
                    start as usize,
                    end as usize,
                    options.threshold,
                    FSMN_VAD_SOURCE,
                )
            })
            .filter(|segment| segment_duration_ms(segment) >= options.min_speech_ms)
            .collect::<Vec<_>>();
        self.split_segments_for_asr(waveform, &raw_segments, options, &mut refine_long_segment)
    }

    fn split_segments_for_asr<F>(
        &self,
        waveform: &Waveform,
        segments: &[TimeSpan],
        options: &VadOptions,
        refine_long_segment: &mut F,
    ) -> Result<Vec<TimeSpan>>
    where
        F: FnMut(&Waveform, &TimeSpan, &VadOptions, u64) -> Result<Vec<TimeSpan>>,
    {
        let max_segment_ms = options.max_segment_ms;
        if segments.is_empty() || max_segment_ms == 0 {
            return Ok(segments.to_vec());
        }

        let mut split = Vec::with_capacity(segments.len());
        for segment in segments {
            if segment_duration_ms(segment) <= max_segment_ms {
                split.push(segment.clone());
                continue;
            }

            let refined = self.refine_long_segment(
                waveform,
                segment,
                options,
                max_segment_ms,
                refine_long_segment,
            )?;
            split.extend(refined);
        }
        Ok(split)
    }

    fn refine_long_segment<F>(
        &self,
        waveform: &Waveform,
        segment: &TimeSpan,
        options: &VadOptions,
        max_segment_ms: u64,
        refine_long_segment: &mut F,
    ) -> Result<Vec<TimeSpan>>
    where
        F: FnMut(&Waveform, &TimeSpan, &VadOptions, u64) -> Result<Vec<TimeSpan>>,
    {
        let mut current = vec![segment.clone()];
        for silence_ms in REFINE_SILENCE_MS {
            if options.min_silence_ms <= silence_ms {
                continue;
            }

            let mut refined = Vec::new();
            let mut changed = false;
            for candidate in current {
                if segment_duration_ms(&candidate) <= max_segment_ms {
                    refined.push(candidate);
                    continue;
                }

                let local = refine_long_segment(waveform, &candidate, options, silence_ms)?;
                if local.is_empty() {
                    refined.push(candidate);
                    continue;
                }
                changed = true;
                refined.extend(local);
            }

            current = refined;
            if current
                .iter()
                .all(|candidate| segment_duration_ms(candidate) <= max_segment_ms)
            {
                return Ok(current);
            }
            if !changed {
                break;
            }
        }

        Ok(current
            .into_iter()
            .flat_map(|candidate| hard_split_segment(&candidate, max_segment_ms))
            .collect())
    }
}

pub struct FsmnVadStreamingPostProcessor {
    e2e: E2EVadModel,
    options: VadOptions,
}

impl FsmnVadStreamingPostProcessor {
    pub fn new(options: VadOptions) -> Self {
        Self {
            e2e: build_e2e_model(&options),
            options,
        }
    }

    pub fn detect_chunk(
        &mut self,
        samples: &[f32],
        frame_scores: &[Vec<f32>],
        is_final: bool,
    ) -> Vec<TimeSpan> {
        let max_end_sil = self.options.min_silence_ms as i32;
        self.e2e
            .detect_chunk(frame_scores, samples, is_final, max_end_sil)
            .into_iter()
            .map(|(start, end)| {
                speech_span(
                    start as usize,
                    end as usize,
                    self.options.threshold,
                    FSMN_VAD_SOURCE,
                )
            })
            .filter(|segment| segment_duration_ms(segment) >= self.options.min_speech_ms)
            .collect()
    }
}

fn build_e2e_model(options: &VadOptions) -> E2EVadModel {
    let config = E2EVadConfig {
        speech_noise_thres: options.threshold,
        ..Default::default()
    };
    E2EVadModel::new(config)
}

fn hard_split_segment(segment: &TimeSpan, max_segment_ms: u64) -> Vec<TimeSpan> {
    if max_segment_ms == 0 || segment_duration_ms(segment) <= max_segment_ms {
        return vec![segment.clone()];
    }

    let mut split = Vec::new();
    let mut start = segment.range.start_ms;
    let end = segment.range.end_ms;
    let confidence = span_confidence(segment);
    let max_segment_ms = max_segment_ms as usize;
    while start < end {
        let next = start.saturating_add(max_segment_ms).min(end);
        split.push(speech_span(start, next, confidence, FSMN_VAD_SOURCE));
        start = next;
    }
    split
}

fn segment_duration_ms(segment: &TimeSpan) -> u64 {
    segment.range.duration() as u64
}
