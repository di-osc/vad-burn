use pyo3::prelude::*;

use crate::{
    BurnFsmnVadDetection, BurnFsmnVadModel, BurnFsmnVadStream, BurnFsmnVadTiming, VadOptions,
    VadSegment, Waveform,
};

#[pyclass(name = "VadOptions")]
#[derive(Debug, Clone)]
pub struct PyVadOptions {
    #[pyo3(get, set)]
    pub threshold: f32,
    #[pyo3(get, set)]
    pub min_speech_ms: u64,
    #[pyo3(get, set)]
    pub min_silence_ms: u64,
    #[pyo3(get, set)]
    pub max_segment_ms: u64,
    #[pyo3(get, set)]
    pub pad_ms: u64,
}

#[pymethods]
impl PyVadOptions {
    #[new]
    #[pyo3(signature = (threshold=0.6, min_speech_ms=250, min_silence_ms=500, max_segment_ms=30000, pad_ms=0))]
    fn new(
        threshold: f32,
        min_speech_ms: u64,
        min_silence_ms: u64,
        max_segment_ms: u64,
        pad_ms: u64,
    ) -> Self {
        Self {
            threshold,
            min_speech_ms,
            min_silence_ms,
            max_segment_ms,
            pad_ms,
        }
    }
}

impl From<&PyVadOptions> for VadOptions {
    fn from(options: &PyVadOptions) -> Self {
        Self {
            threshold: options.threshold,
            min_speech_ms: options.min_speech_ms,
            min_silence_ms: options.min_silence_ms,
            max_segment_ms: options.max_segment_ms,
            pad_ms: options.pad_ms,
        }
    }
}

#[pyclass(name = "VadSegment")]
#[derive(Debug, Clone)]
pub struct PyVadSegment {
    #[pyo3(get)]
    pub start_ms: u64,
    #[pyo3(get)]
    pub end_ms: u64,
    #[pyo3(get)]
    pub probability: f32,
}

impl From<VadSegment> for PyVadSegment {
    fn from(segment: VadSegment) -> Self {
        Self {
            start_ms: segment.range.start.0,
            end_ms: segment.range.end.0,
            probability: segment.probability,
        }
    }
}

#[pyclass(name = "VadTiming")]
#[derive(Debug, Clone)]
pub struct PyVadTiming {
    #[pyo3(get)]
    pub frontend_seconds: f64,
    #[pyo3(get)]
    pub forward_seconds: f64,
    #[pyo3(get)]
    pub segmenter_seconds: f64,
}

impl From<BurnFsmnVadTiming> for PyVadTiming {
    fn from(timing: BurnFsmnVadTiming) -> Self {
        Self {
            frontend_seconds: timing.frontend_seconds,
            forward_seconds: timing.forward_seconds,
            segmenter_seconds: timing.segmenter_seconds,
        }
    }
}

#[pyclass(name = "VadDetection")]
#[derive(Debug, Clone)]
pub struct PyVadDetection {
    #[pyo3(get)]
    pub segments: Vec<PyVadSegment>,
    #[pyo3(get)]
    pub frame_scores: Vec<Vec<f32>>,
    #[pyo3(get)]
    pub timing: PyVadTiming,
}

impl From<BurnFsmnVadDetection> for PyVadDetection {
    fn from(detection: BurnFsmnVadDetection) -> Self {
        Self {
            segments: detection.segments.into_iter().map(Into::into).collect(),
            frame_scores: detection.frame_scores,
            timing: detection.timing.into(),
        }
    }
}

#[pyclass(name = "FsmnVad")]
pub struct PyFsmnVad {
    inner: BurnFsmnVadModel,
}

#[pyclass(name = "FsmnVadStream")]
pub struct PyFsmnVadStream {
    inner: BurnFsmnVadStream,
}

#[pymethods]
impl PyFsmnVad {
    #[new]
    fn new(model_dir: &str) -> PyResult<Self> {
        Ok(Self {
            inner: BurnFsmnVadModel::from_pretrained(model_dir)?,
        })
    }

    #[staticmethod]
    fn from_pretrained(model_dir: &str) -> PyResult<Self> {
        Self::new(model_dir)
    }

    fn detect(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        options: Option<&PyVadOptions>,
    ) -> PyResult<Vec<PyVadSegment>> {
        let options = options.map_or_else(VadOptions::default, Into::into);
        let waveform = Waveform::new(samples, sample_rate);
        Ok(self
            .inner
            .detect(&waveform, &options)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    fn detect_with_timing(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        options: Option<&PyVadOptions>,
    ) -> PyResult<PyVadDetection> {
        let options = options.map_or_else(VadOptions::default, Into::into);
        let waveform = Waveform::new(samples, sample_rate);
        Ok(self.inner.detect_with_timing(&waveform, &options)?.into())
    }

    fn stream(&self, options: Option<&PyVadOptions>) -> PyFsmnVadStream {
        let options = options.map_or_else(VadOptions::default, Into::into);
        PyFsmnVadStream {
            inner: self.inner.stream(options),
        }
    }
}

#[pymethods]
impl PyFsmnVadStream {
    fn push(&mut self, samples: Vec<f32>, sample_rate: u32) -> PyResult<Vec<PyVadSegment>> {
        Ok(self
            .inner
            .push(&samples, sample_rate)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    fn finish(&mut self) -> PyResult<Vec<PyVadSegment>> {
        Ok(self.inner.finish()?.into_iter().map(Into::into).collect())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pymodule]
fn vad_burn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFsmnVad>()?;
    m.add_class::<PyFsmnVadStream>()?;
    m.add_class::<PyVadOptions>()?;
    m.add_class::<PyVadSegment>()?;
    m.add_class::<PyVadTiming>()?;
    m.add_class::<PyVadDetection>()?;
    Ok(())
}
