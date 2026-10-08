# vad-burn

`vad-burn` 是基于 Rust 和 [Burn](https://burn.dev) 0.22 的 VAD 推理库，提供 Rust API、
Python 绑定、离线检测和流式检测。默认使用 Burn Flex CPU 后端，FSMN VAD 也支持
Apple Metal 后端。支持 FSMN VAD 和 FireRedVAD 两种模型，二者共用同一套检测接口。

- 纯 CPU 即可高速推理，离线场景可达 1600x 以上实时速度（FSMN Flex 超过 2000x，Metal 超过 2600x）。
- 同时支持离线整段检测和按 chunk 的流式检测。
- Rust 与 Python 接口对齐，模型可在 FSMN VAD / FireRedVAD 之间直接替换。

## 安装

```bash
cargo add vad-burn
pip install vad-burn
```

需要 GPU 后端时打开对应 Cargo feature（`metal` / `apple-amx`），并用
`from_pretrained_on_device` 传入设备；Burn 0.22 的后端由运行期 `Device` 决定，
模型类型上不带后端泛型。

## Benchmark

### 测试环境

- 设备：MacBook Pro（Apple M4 Pro，14 核 CPU，48 GB 内存）
- 后端：Burn Flex CPU（`rayon`）与 Burn Metal（`--features metal`）
- 音频：`assets/vad_example.wav`，16 kHz mono PCM，70.47 s
- 构建：`--release`，`--warmup 2 --repeat 10`
- RTF = 推理耗时 / 音频时长，加速比 = 1 / RTF

### FSMN VAD

- 模型：`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch@master`
- Flex 与 Metal 均支持；FireRedVAD 当前仅在 Flex 上测过。

```bash
# Flex CPU
cargo run --release -p vad-burn --example bench_fsmn_vad -- \
  --audio assets/vad_example.wav \
  --warmup 2 \
  --repeat 10 \
  --stream-chunk-ms 600

# Metal
cargo run --release --features metal -p vad-burn --example bench_fsmn_vad -- \
  --backend metal \
  --audio assets/vad_example.wav \
  --warmup 2 \
  --repeat 10 \
  --stream-chunk-ms 600
```

| 后端 | 模式 | 平均耗时 | RTF | 加速比 |
| --- | --- | ---: | ---: | ---: |
| Flex | 离线整段 | 32.814 ms | 0.000466 | 2147.58x |
| Flex | 流式 600 ms chunk | 168.029 ms | 0.002384 | 419.40x |
| Metal | 离线整段 | 26.291 ms | 0.000373 | 2680.45x |
| Metal | 流式 600 ms chunk | 395.577 ms | 0.005613 | 178.15x |

Metal 离线比 Flex 快约 25%，细分耗时显示 GPU 算子接近免费，瓶颈转移到结果回读
（`output_tensor`）和 CPU 侧的 fbank（`frontend`）。Metal 流式每 600 ms 需同步一次
GPU 与 CPU，固定开销摊不掉，单次实测波动较大（`min 191 ms / max 1488 ms`），上表数值
仅供参考；短块流式建议用 Flex。

### FireRedVAD

- 模型：`xukaituo/FireRedVAD@master`
- 后端：Flex CPU。
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
| 离线 VAD | 42.357 ms | 0.000601 | 1663.75x |
| Stream-VAD 600 ms chunk | 113.187 ms | 0.001606 | 622.60x |

两个 benchmark example 都会同时打印离线与流式的平均/最小/最大耗时、RTF 和加速比，
FSMN 还会输出 frontend / forward / segmenter 以及各算子的细分耗时，便于定位瓶颈。

## Rust 用法

```rust
use vad_burn::{Audio, FsmnVadModel, VadOptions, Waveform};

let model = FsmnVadModel::from_modelscope()?;
let waveform = Waveform::new(samples, 16_000);
let segments = model.detect(&waveform, &VadOptions::default())?;

let mut audio = Audio::from_path("example.wav")?;
model.annotate(&mut audio, &VadOptions::default())?;
```

`FsmnVadModel` 和 `FireRedVadModel` 使用相同的检测接口。`from_modelscope()` 会自动下载
并缓存默认模型，也可以加载本地模型目录：

```rust
let model = FsmnVadModel::from_pretrained("/path/to/fsmn-vad")?;
let model = FireRedVadModel::from_pretrained("/path/to/firered-vad")?;
```

流式推理：

```rust
let mut session = model.new_session(VadOptions::default());
for chunk in chunks {
    let segments = session.push(chunk, 16_000)?;
}
let final_segments = session.finish()?;
```

## Python 用法

```python
from asr_data import Audio
from vad_burn import FsmnVadModel, VadOptions

vad = FsmnVadModel.from_modelscope()
segments = vad.detect(samples, 16000, VadOptions())

audio = Audio.from_path("example.wav")
vad.annotate(audio, VadOptions())
```

`samples` 必须是 16 kHz 单声道、归一化到 `[-1.0, 1.0]` 的浮点 PCM。
`FsmnVadModel` 可直接替换为 `FireRedVadModel`，离线和流式调用方式保持一致。

## 开发验证

```bash
cargo fmt --check
cargo test -- --nocapture
cargo test --features metal -- --nocapture
```
