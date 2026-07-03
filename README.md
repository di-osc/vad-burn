# vad-burn

`vad-burn` 是基于 Rust 和 Burn 框架实现的 FSMN VAD 推理库，面向 16kHz 单声道语音检测。当前后端使用 Burn Flex CPU 后端，提供 Rust API、Python 绑定、离线推理和流式推理。

## 特点

- Rust + Burn 实现 FSMN VAD 推理。
- 使用 Burn Flex 后端，CPU 上即可运行。
- 支持离线整段检测和增量式流式 chunk 检测。
- 支持 FireRedVAD 离线推理和官方 Stream-VAD 流式推理。
- 提供 Python 绑定，方便在 Python 音频流水线中调用。
- 内置 benchmark 示例和测试音频 `assets/vad_example.wav`。

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
use vad_burn::{FsmnVadModel, VadOptions, Waveform};

let model = FsmnVadModel::from_modelscope()?;
let waveform = Waveform::new(samples, 16_000);
let segments = model.detect(&waveform, &VadOptions::default())?;
```

默认会从 ModelScope 下载并缓存 `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch@master`。也可以加载本地模型目录：

```rust
let model = FsmnVadModel::from_pretrained("/path/to/fsmn-vad")?;
```

流式推理：

```rust
let mut stream = model.new_stream(VadOptions::default());

for chunk in chunks {
    let segments = stream.push(chunk, 16_000)?;
}

let final_segments = stream.finish()?;
```

FireRedVAD 离线推理：

```rust
use vad_burn::{FireRedVadModel, VadOptions, Waveform};

let model = FireRedVadModel::from_modelscope()?;
let waveform = Waveform::new(samples, 16_000);
let segments = model.detect(&waveform, &VadOptions::default())?;
```

FireRedVAD 使用同一个模型对象。`from_modelscope()` 会同时加载官方 `VAD` 和 `Stream-VAD` 权重，`detect` 走离线模型，`new_stream` 走流式模型：

```rust
use vad_burn::{FireRedVadModel, VadOptions};

let model = FireRedVadModel::from_modelscope()?;
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
from vad_burn import FsmnVadModel, VadOptions

vad = FsmnVadModel.from_modelscope()
segments = vad.detect(samples, 16000, VadOptions())

for segment in segments:
    print(segment.start_ms, segment.end_ms, segment.probability)
```

流式推理：

```python
stream = vad.new_stream(VadOptions())

for chunk in chunks:
    segments = stream.push(chunk, 16000)

final_segments = stream.finish()
```

FireRedVAD 离线推理：

```python
from vad_burn import FireRedVadModel, VadOptions

vad = FireRedVadModel.from_modelscope()
segments = vad.detect(samples, 16000, VadOptions())
```

FireRedVAD 流式推理使用同一个模型对象：

```python
from vad_burn import FireRedVadModel, VadOptions

vad = FireRedVadModel.from_modelscope()
stream = vad.new_stream(VadOptions())

for chunk in chunks:
    segments = stream.push(chunk, 16000)

final_segments = stream.finish()
```

`samples` 为归一化到 `[-1.0, 1.0]` 的 `float` PCM 样本，采样率需为 `16000`。

## Benchmark

测试命令：

```bash
cargo run --release -p vad-burn --example bench_fsmn_vad -- \
  --audio assets/vad_example.wav \
  --warmup 2 \
  --repeat 10 \
  --stream-chunk-ms 600
```

测试环境：

- 模型：FSMN VAD `model.pt` + `am.mvn`
- 后端：Burn Flex CPU, `rayon`
- 设备：MacBook Pro, Apple M1, 8 核 CPU, 16 GB 内存
- 音频：`assets/vad_example.wav`, 16kHz mono PCM, 70.47s
- 构建：`--release`

结果：

| 模式 | 平均耗时 | RTF | 加速比 |
| --- | ---: | ---: | ---: |
| 离线整段 | 73.631 ms | 0.001045 | 957.08x |
| 流式 600ms chunk | 198.425 ms | 0.002816 | 355.15x |

FireRedVAD 测试命令：

```bash
cargo run --release -p vad-burn --example bench_firered_vad -- \
  --audio assets/vad_example.wav \
  --warmup 2 \
  --repeat 10 \
  --stream-chunk-ms 600
```

FireRedVAD 测试结果：

| 模式 | 平均耗时 | RTF | 加速比 |
| --- | ---: | ---: | ---: |
| 离线 VAD | 96.833 ms | 0.001374 | 727.75x |
| Stream-VAD 600ms chunk | 176.427 ms | 0.002504 | 399.43x |
