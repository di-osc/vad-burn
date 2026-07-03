# vad-burn

`vad-burn` 是基于 Rust 和 Burn 框架实现的 FSMN VAD 推理库，面向 16kHz 单声道语音检测。当前后端使用 Burn Flex CPU 后端，提供 Rust API、Python 绑定、离线推理和流式推理。

## 特点

- 基于 Rust 和 Burn Flex 实现 VAD 推理，提供 Rust API 和 Python 绑定。
- 仅需 CPU 即可高速推理，离线场景在测试音频上可达到接近 1000x 实时速度。
- 支持 FSMN VAD 和 FireRedVAD，便于在不同 VAD 模型之间切换。
- 同时支持离线整段检测和流式检测。

## Rust 用法

```bash
cargo add vad-burn
```

如果手动编辑 `Cargo.toml`：

```toml
[dependencies]
vad-burn = "0.1.2"
```

```rust
use vad_burn::{FireRedVadModel, FsmnVadModel, VadOptions, Waveform};

let model = FsmnVadModel::from_modelscope()?;
let waveform = Waveform::new(samples, 16_000);
let segments = model.detect(&waveform, &VadOptions::default())?;
```

`FsmnVadModel` 和 `FireRedVadModel` 使用相同的检测接口。`from_modelscope()` 会自动下载并缓存默认模型；也可以加载本地模型目录：

```rust
let model = FsmnVadModel::from_pretrained("/path/to/fsmn-vad")?;
let model = FireRedVadModel::from_pretrained("/path/to/firered-vad")?;
```

流式推理：

```rust
let mut stream = model.new_stream(VadOptions::default());

for chunk in chunks {
    let segments = stream.push(chunk, 16_000)?;
}

let final_segments = stream.finish()?;
```

## Python 用法

```bash
pip install vad-burn
```

```python
from vad_burn import FireRedVadModel, FsmnVadModel, VadOptions

vad = FsmnVadModel.from_modelscope()
segments = vad.detect(samples, 16000, VadOptions())

for segment in segments:
    print(segment.start_ms, segment.end_ms, segment.probability)
```

`FsmnVadModel` 可以直接替换为 `FireRedVadModel`，离线和流式调用方式保持一致。

流式推理：

```python
stream = vad.new_stream(VadOptions())

for chunk in chunks:
    segments = stream.push(chunk, 16000)

final_segments = stream.finish()
```

`samples` 为归一化到 `[-1.0, 1.0]` 的 `float` PCM 样本，采样率需为 `16000`。

## Benchmark

测试环境：

- 后端：Burn Flex CPU, `rayon`
- 设备：MacBook Pro, Apple M1, 8 核 CPU, 16 GB 内存
- 音频：`assets/vad_example.wav`, 16kHz mono PCM, 70.47s
- 构建：`--release`

FSMN VAD：

- 模型：`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch@master`

```bash
cargo run --release -p vad-burn --example bench_fsmn_vad -- \
  --audio assets/vad_example.wav \
  --warmup 2 \
  --repeat 10 \
  --stream-chunk-ms 600
```

| 模式 | 平均耗时 | RTF | 加速比 |
| --- | ---: | ---: | ---: |
| 离线整段 | 73.631 ms | 0.001045 | 957.08x |
| 流式 600ms chunk | 198.425 ms | 0.002816 | 355.15x |

FireRedVAD：

- 模型：`xukaituo/FireRedVAD@master`
- 离线检测使用官方 `VAD` 权重，流式检测使用官方 `Stream-VAD` 权重。

```bash
cargo run --release -p vad-burn --example bench_firered_vad -- \
  --audio assets/vad_example.wav \
  --warmup 2 \
  --repeat 10 \
  --stream-chunk-ms 600
```

| 模式 | 平均耗时 | RTF | 加速比 |
| --- | ---: | ---: | ---: |
| 离线 VAD | 96.833 ms | 0.001374 | 727.75x |
| Stream-VAD 600ms chunk | 176.427 ms | 0.002504 | 399.43x |
