pub type Backend = burn::backend::Flex;

pub const LAYERS: usize = 4;
pub const PROJ_DIM: usize = 128;
pub const CACHE_FRAMES: usize = 19;
pub const CONV_KERNEL: usize = 20;
pub const FEAT_DIM: usize = 400;
pub const SAMPLE_RATE: u32 = 16_000;
