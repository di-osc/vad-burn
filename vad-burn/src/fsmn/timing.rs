use super::constants::LAYERS;

#[derive(Debug, Clone, Copy, Default)]
pub struct FsmnVadTiming {
    pub frontend_seconds: f64,
    pub forward_seconds: f64,
    pub segmenter_seconds: f64,
    pub forward_ops: FsmnForwardTiming,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FsmnForwardTiming {
    pub input_tensor_seconds: f64,
    pub in_linear1_seconds: f64,
    pub in_linear2_seconds: f64,
    pub block_linear_seconds: [f64; LAYERS],
    pub block_memory_seconds: [f64; LAYERS],
    pub block_affine_seconds: [f64; LAYERS],
    pub out_linear1_seconds: f64,
    pub out_linear2_seconds: f64,
    pub softmax_seconds: f64,
    pub output_tensor_seconds: f64,
}
