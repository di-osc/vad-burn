# vad-burn

`vad-burn` 是基于 Rust 和 [Burn](https://burn.dev) 0.22 的 VAD 推理库，提供 Rust API、
Python 绑定、离线检测和流式检测。默认使用 Burn Flex CPU 后端，FSMN VAD 也支持
Apple Metal 后端。支持 FSMN VAD 和 FireRedVAD 两种模型，二者共用同一套检测接口。

- 纯 CPU 即可高速推理，离线场景可达 1600x 以上实时速度（FSMN Flex 超过 2000x，Metal 接近 6000x）。
- 同时支持离线整段检测和按 chunk 的流式检测。
- Rust 与 Python 接口对齐，模型可在 FSMN VAD / FireRedVAD 之间直接替换。
- 特征前端（fbank / LFR / CMVN）全部用 Burn 实现，无 C++ / CMake 依赖，特征可与推理共用同一后端。

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
| Flex | 离线整段 | 33.803 ms | 0.000480 | 2084.76x |
| Flex | 流式 600 ms chunk | 207.200 ms | 0.002940 | 340.11x |
| Metal | 离线整段 | 11.928 ms | 0.000169 | 5908.02x |
| Metal | 流式 600 ms chunk | 457.022 ms | 0.006485 | 154.20x |

离线路径下前端产出的是设备端张量，Metal 上无需把几十 MB 特征从 CPU 拷到 GPU，
因此 Metal 离线从 28.6 ms 降到 11.9 ms。流式的 600 ms chunk 只有 58 帧，FFT 与
matmul 的每次调用固定开销摊不掉，比 CPU 版参考实现慢约 20%，短块流式耗时仍远低于
实时（Flex 340x）。Metal 流式每块都要同步一次 GPU，单次实测波动较大
（`min 224 ms / max 1475 ms`），短块流式建议用 Flex。

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
| 离线 VAD | 43.951 ms | 0.000624 | 1603.41x |
| Stream-VAD 600 ms chunk | 147.961 ms | 0.002100 | 476.28x |

两个 benchmark example 都会同时打印离线与流式的平均/最小/最大耗时、RTF 和加速比，
FSMN 还会输出 frontend / forward / segmenter 以及各算子的细分耗时，便于定位瓶颈。

## 特征前端

fbank / LFR / CMVN 全部用 Burn 算子实现，不再依赖 `kaldi-native-fbank` 的 C++ 代码：

- 分帧用 `Tensor::unfold`，窗口数公式与 Kaldi `snip_edges=true` 完全一致。
- 去直流、预加重、加窗、幂率谱、三角 Mel 滤波器组、对数压缩都按 Kaldi 语义逐步复刻。
- FFT 使用 `burn-signal` 的 `rfft`（Flex 与 Metal 均有原生实现），正向不归一化，
  只保留 `fft_size / 2` 个 bin，与 Kaldi 丢弃 Nyquist bin 的行为一致。
- 流式路径只保留后续帧仍会用到的尾部样本，内存与帧长同阶。

数值对齐由单测保证：对同一段伪随机波形与 `kaldi-fbank-rust-kautism` 逐帧对拍，
Hamming 窗最大绝对误差 `6.6e-5`、Povey 窗 `3.6e-4`（对数域，RMS 约 `3e-6`）。
该 crate 已降级为 `dev-dependency`，仅在测试中使用。

此外还有一层端到端回归护栏：把 FSMN VAD 与 FireRedVAD 在示例音频上的检测边界
固化成 golden 快照（`tests/fixtures/*.txt`）。特征在浮点层面并非逐位相同，
这条断言保证切分结果不漂移；一旦变化会打印首个差异位置。

```bash
# 逐帧数值对拍
cargo test --lib fbank:: -- --nocapture

# 检测边界 golden 回归
cargo test --lib golden
```

需要刷新 golden 基线时（例如更换模型或有意调整算法），刷新后**必须人工 review diff**：

```bash
UPDATE_GOLDEN=1 cargo test --lib golden
```

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

`tests/fixtures/` 下是检测边界的 golden 快照，锁定 FSMN VAD / FireRedVAD 的切分结果；
用 `UPDATE_GOLDEN=1 cargo test --lib golden` 重建后需人工确认 diff。
