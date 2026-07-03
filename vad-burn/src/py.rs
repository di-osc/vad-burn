use pyo3::prelude::*;

use crate::{
    FireRedVadDetection, FireRedVadModel, FireRedVadStream, FireRedVadTiming, FsmnVadDetection,
    FsmnVadModel, FsmnVadStream, FsmnVadTiming, VadOptions, VadSegment, Waveform,
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

    fn __repr__(&self) -> String {
        format!(
            "VadOptions(threshold={:.3}, min_speech_ms={}, min_silence_ms={}, max_segment_ms={}, pad_ms={})",
            self.threshold,
            self.min_speech_ms,
            self.min_silence_ms,
            self.max_segment_ms,
            self.pad_ms
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

#[pymethods]
impl PyVadSegment {
    fn __repr__(&self) -> String {
        format!(
            "VadSegment(start_ms={}, end_ms={}, probability={:.3})",
            self.start_ms, self.end_ms, self.probability
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

impl From<FsmnVadTiming> for PyVadTiming {
    fn from(timing: FsmnVadTiming) -> Self {
        Self {
            frontend_seconds: timing.frontend_seconds,
            forward_seconds: timing.forward_seconds,
            segmenter_seconds: timing.segmenter_seconds,
        }
    }
}

#[pymethods]
impl PyVadTiming {
    fn __repr__(&self) -> String {
        format!(
            "VadTiming(frontend_seconds={:.6}, forward_seconds={:.6}, segmenter_seconds={:.6})",
            self.frontend_seconds, self.forward_seconds, self.segmenter_seconds
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

impl From<FsmnVadDetection> for PyVadDetection {
    fn from(detection: FsmnVadDetection) -> Self {
        Self {
            segments: detection.segments.into_iter().map(Into::into).collect(),
            frame_scores: detection.frame_scores,
            timing: detection.timing.into(),
        }
    }
}

#[pymethods]
impl PyVadDetection {
    fn __repr__(&self) -> String {
        format!(
            "VadDetection(segments={}, frame_scores={}, timing={})",
            self.segments.len(),
            self.frame_scores.len(),
            self.timing.__repr__()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass(name = "FireRedVadTiming")]
#[derive(Debug, Clone)]
pub struct PyFireRedVadTiming {
    #[pyo3(get)]
    pub frontend_seconds: f64,
    #[pyo3(get)]
    pub forward_seconds: f64,
    #[pyo3(get)]
    pub postprocess_seconds: f64,
    #[pyo3(get)]
    pub frames: usize,
}

impl From<FireRedVadTiming> for PyFireRedVadTiming {
    fn from(timing: FireRedVadTiming) -> Self {
        Self {
            frontend_seconds: timing.frontend_seconds,
            forward_seconds: timing.forward_seconds,
            postprocess_seconds: timing.postprocess_seconds,
            frames: timing.frames,
        }
    }
}

#[pymethods]
impl PyFireRedVadTiming {
    fn __repr__(&self) -> String {
        format!(
            "FireRedVadTiming(frontend_seconds={:.6}, forward_seconds={:.6}, postprocess_seconds={:.6}, frames={})",
            self.frontend_seconds, self.forward_seconds, self.postprocess_seconds, self.frames
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass(name = "FireRedVadDetection")]
#[derive(Debug, Clone)]
pub struct PyFireRedVadDetection {
    #[pyo3(get)]
    pub segments: Vec<PyVadSegment>,
    #[pyo3(get)]
    pub frame_scores: Vec<f32>,
    #[pyo3(get)]
    pub timing: PyFireRedVadTiming,
}

impl From<FireRedVadDetection> for PyFireRedVadDetection {
    fn from(detection: FireRedVadDetection) -> Self {
        Self {
            segments: detection.segments.into_iter().map(Into::into).collect(),
            frame_scores: detection.frame_scores,
            timing: detection.timing.into(),
        }
    }
}

#[pymethods]
impl PyFireRedVadDetection {
    fn __repr__(&self) -> String {
        format!(
            "FireRedVadDetection(segments={}, frame_scores={}, timing={})",
            self.segments.len(),
            self.frame_scores.len(),
            self.timing.__repr__()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass(name = "FsmnVadModel")]
pub struct PyFsmnVadModel {
    inner: FsmnVadModel,
}

#[pyclass(name = "FireRedVadModel")]
pub struct PyFireRedVadModel {
    inner: FireRedVadModel,
}

#[pyclass(name = "FireRedVadStream", unsendable)]
pub struct PyFireRedVadStream {
    inner: FireRedVadStream,
}

#[pyclass(name = "FsmnVadStream", unsendable)]
pub struct PyFsmnVadStream {
    inner: FsmnVadStream,
}

#[pymethods]
impl PyFsmnVadModel {
    #[new]
    fn new(py: Python<'_>, model_dir: &str) -> PyResult<Self> {
        let model_dir = model_dir.to_owned();
        let inner = py.allow_threads(move || FsmnVadModel::from_pretrained(model_dir))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_pretrained(py: Python<'_>, model_dir: &str) -> PyResult<Self> {
        Self::new(py, model_dir)
    }

    #[staticmethod]
    #[pyo3(signature = (repo_id=None, revision=None))]
    fn from_modelscope(
        py: Python<'_>,
        repo_id: Option<&str>,
        revision: Option<&str>,
    ) -> PyResult<Self> {
        let repo_id = repo_id
            .unwrap_or(crate::DEFAULT_MODELSCOPE_REPO_ID)
            .to_owned();
        let revision = revision
            .unwrap_or(crate::DEFAULT_MODELSCOPE_REVISION)
            .to_owned();
        let inner =
            py.allow_threads(move || FsmnVadModel::from_modelscope_revision(&repo_id, &revision))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (samples, sample_rate, options=None))]
    fn detect(
        &self,
        py: Python<'_>,
        samples: Vec<f32>,
        sample_rate: u32,
        options: Option<&PyVadOptions>,
    ) -> PyResult<Vec<PyVadSegment>> {
        let options = options.map_or_else(VadOptions::default, Into::into);
        let waveform = Waveform::new(samples, sample_rate);
        let segments = py.allow_threads(|| self.inner.detect(&waveform, &options))?;
        Ok(segments.into_iter().map(Into::into).collect())
    }

    #[pyo3(signature = (samples, sample_rate, options=None))]
    fn detect_with_timing(
        &self,
        py: Python<'_>,
        samples: Vec<f32>,
        sample_rate: u32,
        options: Option<&PyVadOptions>,
    ) -> PyResult<PyVadDetection> {
        let options = options.map_or_else(VadOptions::default, Into::into);
        let waveform = Waveform::new(samples, sample_rate);
        Ok(py
            .allow_threads(|| self.inner.detect_with_timing(&waveform, &options))?
            .into())
    }

    #[pyo3(signature = (options=None))]
    fn new_stream(&self, options: Option<&PyVadOptions>) -> PyFsmnVadStream {
        let options = options.map_or_else(VadOptions::default, Into::into);
        PyFsmnVadStream {
            inner: self.inner.new_stream(options),
        }
    }

    fn __repr__(&self) -> String {
        "FsmnVadModel()".to_owned()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pymethods]
impl PyFireRedVadModel {
    #[new]
    fn new(py: Python<'_>, model_dir: &str) -> PyResult<Self> {
        let model_dir = model_dir.to_owned();
        let inner = py.allow_threads(move || FireRedVadModel::from_pretrained(model_dir))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_pretrained(py: Python<'_>, model_dir: &str) -> PyResult<Self> {
        Self::new(py, model_dir)
    }

    #[staticmethod]
    #[pyo3(signature = (repo_id=None, revision=None))]
    fn from_modelscope(
        py: Python<'_>,
        repo_id: Option<&str>,
        revision: Option<&str>,
    ) -> PyResult<Self> {
        let repo_id = repo_id
            .unwrap_or(crate::DEFAULT_FIRERED_MODELSCOPE_REPO_ID)
            .to_owned();
        let revision = revision
            .unwrap_or(crate::DEFAULT_FIRERED_MODELSCOPE_REVISION)
            .to_owned();
        let inner = py.allow_threads(move || {
            FireRedVadModel::from_modelscope_revision(&repo_id, &revision)
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (repo_id=None, revision=None))]
    fn from_modelscope_stream(
        py: Python<'_>,
        repo_id: Option<&str>,
        revision: Option<&str>,
    ) -> PyResult<Self> {
        let repo_id = repo_id
            .unwrap_or(crate::DEFAULT_FIRERED_MODELSCOPE_REPO_ID)
            .to_owned();
        let revision = revision
            .unwrap_or(crate::DEFAULT_FIRERED_MODELSCOPE_REVISION)
            .to_owned();
        let inner = py.allow_threads(move || {
            FireRedVadModel::from_modelscope_stream_revision(&repo_id, &revision)
        })?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (samples, sample_rate, options=None))]
    fn detect(
        &self,
        py: Python<'_>,
        samples: Vec<f32>,
        sample_rate: u32,
        options: Option<&PyVadOptions>,
    ) -> PyResult<Vec<PyVadSegment>> {
        let options = options.map_or_else(VadOptions::default, Into::into);
        let waveform = Waveform::new(samples, sample_rate);
        let segments = py.allow_threads(|| self.inner.detect(&waveform, &options))?;
        Ok(segments.into_iter().map(Into::into).collect())
    }

    #[pyo3(signature = (samples, sample_rate, options=None))]
    fn detect_with_timing(
        &self,
        py: Python<'_>,
        samples: Vec<f32>,
        sample_rate: u32,
        options: Option<&PyVadOptions>,
    ) -> PyResult<PyFireRedVadDetection> {
        let options = options.map_or_else(VadOptions::default, Into::into);
        let waveform = Waveform::new(samples, sample_rate);
        Ok(py
            .allow_threads(|| self.inner.detect_with_timing(&waveform, &options))?
            .into())
    }

    #[pyo3(signature = (options=None))]
    fn new_stream(&self, options: Option<&PyVadOptions>) -> PyFireRedVadStream {
        let options = options.map_or_else(VadOptions::default, Into::into);
        PyFireRedVadStream {
            inner: self.inner.new_stream(options),
        }
    }

    fn __repr__(&self) -> String {
        "FireRedVadModel()".to_owned()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pymethods]
impl PyFireRedVadStream {
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

    fn __repr__(&self) -> String {
        "FireRedVadStream()".to_owned()
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

    fn __repr__(&self) -> String {
        "FsmnVadStream()".to_owned()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pymodule]
fn vad_burn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFsmnVadModel>()?;
    m.add_class::<PyFireRedVadModel>()?;
    m.add_class::<PyFsmnVadStream>()?;
    m.add_class::<PyFireRedVadStream>()?;
    m.add_class::<PyVadOptions>()?;
    m.add_class::<PyVadSegment>()?;
    m.add_class::<PyVadTiming>()?;
    m.add_class::<PyVadDetection>()?;
    m.add_class::<PyFireRedVadTiming>()?;
    m.add_class::<PyFireRedVadDetection>()?;
    Ok(())
}
