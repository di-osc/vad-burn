# vad-burn

`vad-burn` 是基于 Rust 和 Burn 的 VAD 推理库，提供 Rust API、Python 绑定、
离线检测和流式检测。默认使用 Burn Flex CPU 后端，FSMN VAD 也支持 Apple Metal
后端。

完整文档：

- [概览与快速开始](https://github.com/di-osc/libraries/tree/main/docs/vad-burn/index.mdx)
- [Rust 使用](https://github.com/di-osc/libraries/tree/main/docs/vad-burn/rust.mdx)
- [Python 使用](https://github.com/di-osc/libraries/tree/main/docs/vad-burn/python.mdx)
- [API 参考](https://github.com/di-osc/libraries/tree/main/docs/vad-burn/api.mdx)
- [Benchmark 与 RTFX](https://github.com/di-osc/libraries/tree/main/docs/vad-burn/benchmark.mdx)

## 安装

```bash
cargo add vad-burn
pip install vad-burn
```

## Rust

```rust
use vad_burn::{FsmnVadModel, VadOptions, Waveform};

let model = FsmnVadModel::from_modelscope()?;
let waveform = Waveform::new(samples, 16_000);
let segments = model.detect(&waveform, &VadOptions::default())?;
```

## Python

```python
from vad_burn import FsmnVadModel, VadOptions

vad = FsmnVadModel.from_modelscope()
segments = vad.detect(samples, 16000, VadOptions())
```

`samples` 必须是 16 kHz 单声道、归一化到 `[-1.0, 1.0]` 的浮点 PCM。

## Benchmark

```bash
cargo run --release --features metal --example bench_fsmn_vad -- \
  --backend metal \
  --audio assets/vad_example.wav \
  --warmup 2 \
  --repeat 10 \
  --stream-chunk-ms 600
```

更多 Flex、Metal 和 FireRedVAD benchmark 结果见
[Benchmark 文档](https://github.com/di-osc/libraries/tree/main/docs/vad-burn/benchmark.mdx)。

## 开发验证

```bash
cargo fmt --check
cargo test -- --nocapture
cargo test --features metal -- --nocapture
```
