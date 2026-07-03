# vad-burn

`vad-burn` 是基于 Rust 和 Burn 框架实现的 FSMN VAD 推理库，面向 16kHz 单声道语音检测。当前后端使用 Burn Flex CPU 后端，提供 Rust API、Python 绑定、离线推理和流式推理。

## 特点

- Rust + Burn 实现 FSMN VAD 推理。
- 使用 Burn Flex 后端，CPU 上即可运行。
- 支持离线整段检测和增量式流式 chunk 检测。
- 提供 Python 绑定，方便在 Python 音频流水线中调用。
- 内置 benchmark 示例和测试音频 `assets/vad_example.wav`。

## Rust 用法

```toml
[dependencies]
vad-burn = "0.1"
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
- 后端：Burn Flex CPU
- 设备：MacBook Pro, Apple M1, 8 核 CPU, 16 GB 内存
- 音频：`assets/vad_example.wav`, 16kHz mono PCM, 70.47s
- 构建：`--release`

结果：

| 模式 | 平均耗时 | RTF | 加速比 |
| --- | ---: | ---: | ---: |
| 离线整段 | 70.914 ms | 0.001006 | 993.74x |
| 流式 600ms chunk | 177.846 ms | 0.002524 | 396.24x |
