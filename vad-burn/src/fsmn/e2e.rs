//! Frame-level VAD endpoint detector ported from `funasr_onnx.utils.e2e_vad`.
//! Matches fasr / funasr_onnx offline segmentation behaviour.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadStateMachine {
    StartPointNotDetected,
    InSpeechSegment,
    EndPointDetected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameState {
    Sil = 0,
    Speech = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioChangeState {
    Speech2Speech,
    Speech2Sil,
    Sil2Sil,
    Sil2Speech,
    #[allow(dead_code)]
    NoBegin,
    #[allow(dead_code)]
    Invalid,
}

#[derive(Debug, Clone)]
pub struct E2EVadConfig {
    pub sample_rate: u32,
    pub detect_mode: i32,
    pub max_end_silence_time: i32,
    pub max_start_silence_time: i32,
    pub window_size_ms: i32,
    pub sil_to_speech_time_thres: i32,
    pub speech_to_sil_time_thres: i32,
    pub speech_2_noise_ratio: f32,
    pub do_extend: bool,
    pub lookback_time_start_point: i32,
    pub lookahead_time_end_point: i32,
    pub max_single_segment_time: i32,
    pub snr_thres: f32,
    pub noise_frame_num_used_for_snr: i32,
    pub decibel_thres: f32,
    pub speech_noise_thres: f32,
    pub fe_prior_thres: f32,
    pub silence_pdf_num: usize,
    pub sil_pdf_ids: Vec<usize>,
    pub frame_in_ms: i32,
    pub frame_length_ms: i32,
}

impl Default for E2EVadConfig {
    fn default() -> Self {
        // Defaults from funasr/fsmn-vad config.yaml `model_conf`.
        Self {
            sample_rate: 16_000,
            detect_mode: 1,
            max_end_silence_time: 800,
            max_start_silence_time: 3000,
            window_size_ms: 200,
            sil_to_speech_time_thres: 150,
            speech_to_sil_time_thres: 150,
            speech_2_noise_ratio: 1.0,
            do_extend: true,
            lookback_time_start_point: 200,
            lookahead_time_end_point: 100,
            max_single_segment_time: 60_000,
            snr_thres: -100.0,
            noise_frame_num_used_for_snr: 100,
            decibel_thres: -100.0,
            speech_noise_thres: 0.6,
            fe_prior_thres: 1e-4,
            silence_pdf_num: 1,
            sil_pdf_ids: vec![0],
            frame_in_ms: 10,
            frame_length_ms: 25,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SpeechSegmentBuf {
    start_ms: i32,
    end_ms: i32,
    contain_seg_start_point: bool,
    contain_seg_end_point: bool,
}

struct WindowDetector {
    win_size_frame: i32,
    win_sum: i32,
    win_state: Vec<i32>,
    cur_win_pos: usize,
    pre_frame_state: FrameState,
    sil_to_speech_frmcnt_thres: i32,
    speech_to_sil_frmcnt_thres: i32,
}

impl WindowDetector {
    fn new(
        window_size_ms: i32,
        sil_to_speech_time: i32,
        speech_to_sil_time: i32,
        frame_size_ms: i32,
    ) -> Self {
        let win_size_frame = window_size_ms / frame_size_ms;
        Self {
            win_size_frame,
            win_sum: 0,
            win_state: vec![0; win_size_frame as usize],
            cur_win_pos: 0,
            pre_frame_state: FrameState::Sil,
            sil_to_speech_frmcnt_thres: sil_to_speech_time / frame_size_ms,
            speech_to_sil_frmcnt_thres: speech_to_sil_time / frame_size_ms,
        }
    }

    fn reset(&mut self) {
        self.cur_win_pos = 0;
        self.win_sum = 0;
        self.win_state.fill(0);
        self.pre_frame_state = FrameState::Sil;
    }

    fn win_size(&self) -> i32 {
        self.win_size_frame
    }

    fn detect_one_frame(&mut self, frame_state: FrameState, _frame_count: i32) -> AudioChangeState {
        let cur_frame_state = match frame_state {
            FrameState::Speech => 1,
            FrameState::Sil => 0,
        };
        self.win_sum -= self.win_state[self.cur_win_pos];
        self.win_sum += cur_frame_state;
        self.win_state[self.cur_win_pos] = cur_frame_state;
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame as usize;

        if self.pre_frame_state == FrameState::Sil
            && self.win_sum >= self.sil_to_speech_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Speech;
            return AudioChangeState::Sil2Speech;
        }
        if self.pre_frame_state == FrameState::Speech
            && self.win_sum <= self.speech_to_sil_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Sil;
            return AudioChangeState::Speech2Sil;
        }
        if self.pre_frame_state == FrameState::Sil {
            AudioChangeState::Sil2Sil
        } else {
            AudioChangeState::Speech2Speech
        }
    }
}

pub struct E2EVadModel {
    opts: E2EVadConfig,
    windows_detector: WindowDetector,
    data_buf_start_frame: i32,
    frm_cnt: i32,
    latest_confirmed_speech_frame: i32,
    lastest_confirmed_silence_frame: i32,
    continous_silence_frame_count: i32,
    vad_state_machine: VadStateMachine,
    confirmed_start_frame: i32,
    confirmed_end_frame: i32,
    number_end_time_detected: i32,
    sil_frame: i32,
    noise_average_decibel: f32,
    pre_end_silence_detected: bool,
    output_data_buf: Vec<SpeechSegmentBuf>,
    output_data_buf_offset: usize,
    max_end_sil_frame_cnt_thresh: i32,
    speech_noise_thres: f32,
    nn_eval_block_size: i32,
    idx_pre_chunk: i32,
    max_time_out: bool,
    decibel: Vec<f32>,
    data_buf_size: i32,
    data_buf_all_size: i32,
}

impl E2EVadModel {
    pub fn new(opts: E2EVadConfig) -> Self {
        let speech_noise_thres = opts.speech_noise_thres;
        let max_end_sil_frame_cnt_thresh =
            opts.max_end_silence_time - opts.speech_to_sil_time_thres;
        let windows_detector = WindowDetector::new(
            opts.window_size_ms,
            opts.sil_to_speech_time_thres,
            opts.speech_to_sil_time_thres,
            opts.frame_in_ms,
        );
        let mut model = Self {
            opts,
            windows_detector,
            data_buf_start_frame: 0,
            frm_cnt: 0,
            latest_confirmed_speech_frame: 0,
            lastest_confirmed_silence_frame: -1,
            continous_silence_frame_count: 0,
            vad_state_machine: VadStateMachine::StartPointNotDetected,
            confirmed_start_frame: -1,
            confirmed_end_frame: -1,
            number_end_time_detected: 0,
            sil_frame: 0,
            noise_average_decibel: -100.0,
            pre_end_silence_detected: false,
            output_data_buf: Vec::new(),
            output_data_buf_offset: 0,
            max_end_sil_frame_cnt_thresh,
            speech_noise_thres,
            nn_eval_block_size: 0,
            idx_pre_chunk: 0,
            max_time_out: false,
            decibel: Vec::new(),
            data_buf_size: 0,
            data_buf_all_size: 0,
        };
        model.reset_detection();
        model
    }

    fn reset_detection(&mut self) {
        self.continous_silence_frame_count = 0;
        self.latest_confirmed_speech_frame = 0;
        self.lastest_confirmed_silence_frame = -1;
        self.confirmed_start_frame = -1;
        self.confirmed_end_frame = -1;
        self.vad_state_machine = VadStateMachine::StartPointNotDetected;
        self.windows_detector.reset();
        self.sil_frame = 0;
    }

    fn all_reset_detection(&mut self) {
        self.data_buf_start_frame = 0;
        self.frm_cnt = 0;
        self.output_data_buf.clear();
        self.output_data_buf_offset = 0;
        self.idx_pre_chunk = 0;
        self.max_time_out = false;
        self.decibel.clear();
        self.data_buf_size = 0;
        self.data_buf_all_size = 0;
        self.number_end_time_detected = 0;
        self.noise_average_decibel = -100.0;
        self.pre_end_silence_detected = false;
        self.max_end_sil_frame_cnt_thresh =
            self.opts.max_end_silence_time - self.opts.speech_to_sil_time_thres;
        self.reset_detection();
    }

    fn compute_decibel(&mut self, waveform: &[f32]) {
        let frame_sample_length = (self.opts.frame_length_ms * self.opts.sample_rate as i32) / 1000;
        let frame_shift_length = (self.opts.frame_in_ms * self.opts.sample_rate as i32) / 1000;
        if self.data_buf_all_size == 0 {
            self.data_buf_all_size = waveform.len() as i32;
            self.data_buf_size = self.data_buf_all_size;
        } else {
            self.data_buf_all_size += waveform.len() as i32;
        }
        let max_offset = waveform.len().saturating_sub(frame_sample_length as usize);
        let mut offset = 0usize;
        while offset <= max_offset {
            let end = offset + frame_sample_length as usize;
            let energy: f32 = waveform[offset..end].iter().map(|s| s * s).sum();
            self.decibel.push(10.0 * (energy + 1e-6).log10());
            offset += frame_shift_length as usize;
        }
    }

    fn compute_scores(&mut self, scores: &[Vec<f32>]) {
        self.nn_eval_block_size = scores.len() as i32;
        self.frm_cnt += scores.len() as i32;
    }

    fn pop_data_buf_till_frame(&mut self, frame_idx: i32) {
        while self.data_buf_start_frame < frame_idx {
            let shift = (self.opts.frame_in_ms * self.opts.sample_rate as i32) / 1000;
            if self.data_buf_size >= shift {
                self.data_buf_start_frame += 1;
                self.data_buf_size = self.data_buf_all_size
                    - self.data_buf_start_frame
                        * ((self.opts.frame_in_ms * self.opts.sample_rate as i32) / 1000);
            } else {
                break;
            }
        }
    }

    fn pop_data_to_output_buf(
        &mut self,
        start_frm: i32,
        frm_cnt: i32,
        first_frm_is_start_point: bool,
        last_frm_is_end_point: bool,
        end_point_is_sent_end: bool,
    ) {
        self.pop_data_buf_till_frame(start_frm);
        let mut expected_sample_number =
            (frm_cnt * self.opts.sample_rate as i32 * self.opts.frame_in_ms) / 1000;
        if last_frm_is_end_point {
            let extra_sample = ((self.opts.frame_length_ms * self.opts.sample_rate as i32) / 1000)
                - ((self.opts.sample_rate as i32 * self.opts.frame_in_ms) / 1000);
            expected_sample_number += extra_sample.max(0);
        }
        if end_point_is_sent_end {
            expected_sample_number = expected_sample_number.max(self.data_buf_size);
        }

        if self.output_data_buf.is_empty() || first_frm_is_start_point {
            self.output_data_buf.push(SpeechSegmentBuf {
                start_ms: start_frm * self.opts.frame_in_ms,
                end_ms: start_frm * self.opts.frame_in_ms,
                contain_seg_start_point: false,
                contain_seg_end_point: false,
            });
        }
        let cur_idx = self.output_data_buf.len() - 1;
        let data_to_pop = if end_point_is_sent_end {
            expected_sample_number.min(self.data_buf_size)
        } else {
            ((frm_cnt * self.opts.frame_in_ms * self.opts.sample_rate as i32) / 1000)
                .min(self.data_buf_size)
        };
        self.data_buf_start_frame += frm_cnt;
        let cur = &mut self.output_data_buf[cur_idx];
        cur.end_ms = (start_frm + frm_cnt) * self.opts.frame_in_ms;
        if first_frm_is_start_point {
            cur.contain_seg_start_point = true;
        }
        if last_frm_is_end_point {
            cur.contain_seg_end_point = true;
        }
        let _ = data_to_pop;
    }

    fn on_silence_detected(&mut self, valid_frame: i32) {
        self.lastest_confirmed_silence_frame = valid_frame;
        if self.vad_state_machine == VadStateMachine::StartPointNotDetected {
            self.pop_data_buf_till_frame(valid_frame);
        }
    }

    fn on_voice_detected(&mut self, valid_frame: i32) {
        self.latest_confirmed_speech_frame = valid_frame;
        self.pop_data_to_output_buf(valid_frame, 1, false, false, false);
    }

    fn on_voice_start(&mut self, start_frame: i32, fake_result: bool) {
        self.confirmed_start_frame = start_frame;
        if !fake_result && self.vad_state_machine == VadStateMachine::StartPointNotDetected {
            self.pop_data_to_output_buf(self.confirmed_start_frame, 1, true, false, false);
        }
    }

    fn on_voice_end(&mut self, end_frame: i32, fake_result: bool, is_last_frame: bool) {
        for t in (self.latest_confirmed_speech_frame + 1)..end_frame {
            self.on_voice_detected(t);
        }
        self.confirmed_end_frame = end_frame;
        if !fake_result {
            self.sil_frame = 0;
            self.pop_data_to_output_buf(self.confirmed_end_frame, 1, false, true, is_last_frame);
        }
        self.number_end_time_detected += 1;
    }

    fn maybe_on_voice_end_if_last_frame(&mut self, is_final_frame: bool, cur_frm_idx: i32) {
        if is_final_frame {
            self.on_voice_end(cur_frm_idx, false, true);
            self.vad_state_machine = VadStateMachine::EndPointDetected;
        }
    }

    fn latency_frm_num_at_start_point(&self) -> i32 {
        let mut vad_latency = self.windows_detector.win_size();
        if self.opts.do_extend {
            vad_latency += self.opts.lookback_time_start_point / self.opts.frame_in_ms;
        }
        vad_latency
    }

    fn get_frame_state(&mut self, t: i32, scores: &[Vec<f32>]) -> FrameState {
        let cur_decibel = self.decibel.get(t as usize).copied().unwrap_or(-100.0);
        let cur_snr = cur_decibel - self.noise_average_decibel;
        if cur_decibel < self.opts.decibel_thres {
            return FrameState::Sil;
        }

        let local_t = (t - self.idx_pre_chunk) as usize;
        let frame_scores = scores.get(local_t);
        if self.opts.silence_pdf_num > 0 {
            let Some(frame_scores) = frame_scores else {
                return FrameState::Sil;
            };
            let sil_sum: f32 = self
                .opts
                .sil_pdf_ids
                .iter()
                .filter_map(|&id| frame_scores.get(id).copied())
                .sum();
            let sum_score = 1.0 - sil_sum;
            let noise_prob = (sil_sum.max(1e-10)).ln() * self.opts.speech_2_noise_ratio;
            let speech_prob = (sum_score.max(1e-10)).ln();
            if speech_prob.exp() >= noise_prob.exp() + self.speech_noise_thres {
                if cur_snr >= self.opts.snr_thres && cur_decibel >= self.opts.decibel_thres {
                    FrameState::Speech
                } else {
                    FrameState::Sil
                }
            } else {
                if self.noise_average_decibel < -99.9 {
                    self.noise_average_decibel = cur_decibel;
                } else {
                    self.noise_average_decibel = (cur_decibel
                        + self.noise_average_decibel
                            * (self.opts.noise_frame_num_used_for_snr - 1) as f32)
                        / self.opts.noise_frame_num_used_for_snr as f32;
                }
                FrameState::Sil
            }
        } else {
            FrameState::Sil
        }
    }

    fn detect_one_frame(
        &mut self,
        cur_frm_state: FrameState,
        cur_frm_idx: i32,
        is_final_frame: bool,
        _scores: &[Vec<f32>],
    ) {
        let tmp_cur_frm_state = match cur_frm_state {
            FrameState::Speech if 1.0_f32.abs() > self.opts.fe_prior_thres => FrameState::Speech,
            FrameState::Speech => FrameState::Sil,
            FrameState::Sil => FrameState::Sil,
        };
        let state_change = self
            .windows_detector
            .detect_one_frame(tmp_cur_frm_state, cur_frm_idx);
        let frm_shift_in_ms = self.opts.frame_in_ms;

        match state_change {
            AudioChangeState::Sil2Speech => {
                self.continous_silence_frame_count = 0;
                self.pre_end_silence_detected = false;
                if self.vad_state_machine == VadStateMachine::StartPointNotDetected {
                    let start_frame = self
                        .data_buf_start_frame
                        .max(cur_frm_idx - self.latency_frm_num_at_start_point());
                    self.on_voice_start(start_frame, false);
                    self.vad_state_machine = VadStateMachine::InSpeechSegment;
                    for t in (start_frame + 1)..=cur_frm_idx {
                        self.on_voice_detected(t);
                    }
                } else if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    for t in (self.latest_confirmed_speech_frame + 1)..cur_frm_idx {
                        self.on_voice_detected(t);
                    }
                    if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx);
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                    }
                }
            }
            AudioChangeState::Speech2Sil => {
                self.continous_silence_frame_count = 0;
                if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx);
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                    }
                }
            }
            AudioChangeState::Speech2Speech => {
                self.continous_silence_frame_count = 0;
                if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.max_time_out = true;
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx);
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                    }
                }
            }
            AudioChangeState::Sil2Sil => {
                self.continous_silence_frame_count += 1;
                if self.vad_state_machine == VadStateMachine::StartPointNotDetected {
                    if (self.opts.detect_mode == 0
                        && self.continous_silence_frame_count * frm_shift_in_ms
                            > self.opts.max_start_silence_time)
                        || (is_final_frame && self.number_end_time_detected == 0)
                    {
                        for t in (self.lastest_confirmed_silence_frame + 1)..cur_frm_idx {
                            self.on_silence_detected(t);
                        }
                        self.on_voice_start(0, true);
                        self.on_voice_end(0, true, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if cur_frm_idx >= self.latency_frm_num_at_start_point() {
                        self.on_silence_detected(
                            cur_frm_idx - self.latency_frm_num_at_start_point(),
                        );
                    }
                } else if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if self.continous_silence_frame_count * frm_shift_in_ms
                        >= self.max_end_sil_frame_cnt_thresh
                    {
                        let mut lookback_frame =
                            self.max_end_sil_frame_cnt_thresh / frm_shift_in_ms;
                        if self.opts.do_extend {
                            lookback_frame -= self.opts.lookahead_time_end_point / frm_shift_in_ms;
                            lookback_frame -= 1;
                            lookback_frame = lookback_frame.max(0);
                        }
                        self.on_voice_end(cur_frm_idx - lookback_frame, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if self.opts.do_extend && !is_final_frame {
                        if self.continous_silence_frame_count
                            <= self.opts.lookahead_time_end_point / frm_shift_in_ms
                        {
                            self.on_voice_detected(cur_frm_idx);
                        }
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                    }
                }
            }
            AudioChangeState::Invalid | AudioChangeState::NoBegin => {}
        }

        if self.vad_state_machine == VadStateMachine::EndPointDetected && self.opts.detect_mode == 1
        {
            self.reset_detection();
        }
    }

    fn detect_common_frames(&mut self, scores: &[Vec<f32>]) {
        if self.vad_state_machine == VadStateMachine::EndPointDetected {
            return;
        }
        let block = scores.len() as i32;
        for i in (0..block).rev() {
            let frame_idx = self.frm_cnt - 1 - i;
            let frame_state = self.get_frame_state(frame_idx, scores);
            self.detect_one_frame(frame_state, frame_idx, false, scores);
        }
        self.idx_pre_chunk += block;
    }

    fn detect_last_frames(&mut self, scores: &[Vec<f32>]) {
        if self.vad_state_machine == VadStateMachine::EndPointDetected {
            return;
        }
        let block = scores.len() as i32;
        for i in (0..block).rev() {
            let frame_idx = self.frm_cnt - 1 - i;
            let frame_state = self.get_frame_state(frame_idx, scores);
            let is_final = i == 0;
            self.detect_one_frame(frame_state, frame_idx, is_final, scores);
        }
    }

    /// Incremental segmentation for streaming (online=true semantics).
    pub fn detect_chunk(
        &mut self,
        scores: &[Vec<f32>],
        waveform: &[f32],
        is_final: bool,
        max_end_silence_ms: i32,
    ) -> Vec<(u64, u64)> {
        self.max_end_sil_frame_cnt_thresh = max_end_silence_ms - self.opts.speech_to_sil_time_thres;
        self.compute_decibel(waveform);
        self.compute_scores(scores);
        if is_final {
            self.detect_last_frames(scores);
        } else {
            self.detect_common_frames(scores);
        }

        let mut segments = Vec::new();
        for i in self.output_data_buf_offset..self.output_data_buf.len() {
            let seg = &self.output_data_buf[i];
            if seg.contain_seg_start_point && seg.contain_seg_end_point {
                segments.push((seg.start_ms.max(0) as u64, seg.end_ms.max(0) as u64));
                self.output_data_buf_offset += 1;
            }
        }
        if is_final {
            self.all_reset_detection();
        }
        segments
    }
}

#[cfg(test)]
mod tests {
    use super::E2EVadModel;

    #[test]
    fn e2e_vad_default_config_matches_funasr_model_conf() {
        let model = E2EVadModel::new(super::E2EVadConfig::default());
        assert_eq!(model.opts.speech_noise_thres, 0.6);
        assert_eq!(model.opts.max_end_silence_time, 800);
        assert_eq!(model.opts.frame_in_ms, 10);
    }
}
