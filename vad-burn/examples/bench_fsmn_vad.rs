use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use vad_burn::{BurnFsmnForwardTiming, BurnFsmnVadModel, VadOptions, Waveform};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let waveform = load_pcm16_wav(&args.audio)?;
    if waveform.sample_rate != 16_000 || waveform.channels != 1 {
        bail!("expected 16kHz mono WAV");
    }

    let options = VadOptions::default();
    let burn = BurnFsmnVadModel::from_pretrained(&args.model)?;

    for _ in 0..args.warmup {
        let _ = burn.detect(&waveform, &options)?;
    }

    let mut runs = Vec::with_capacity(args.repeat);
    let mut burn_segments = Vec::new();
    for _ in 0..args.repeat {
        let start = Instant::now();
        burn_segments = burn.detect(&waveform, &options)?;
        runs.push(start.elapsed());
    }

    let mut stream_runs = Vec::with_capacity(args.repeat);
    let mut stream_segments = Vec::new();
    for _ in 0..args.repeat {
        let start = Instant::now();
        stream_segments = detect_streaming(&burn, &waveform, &options, args.stream_chunk_ms)?;
        stream_runs.push(start.elapsed());
    }

    let diagnostic = burn.detect_with_timing(&waveform, &options)?;
    if burn_segments != diagnostic.segments {
        bail!("Burn fast-path segments differ from timed diagnostic path");
    }
    let timings = [diagnostic.timing];

    let avg = runs.iter().map(Duration::as_secs_f64).sum::<f64>() / runs.len() as f64;
    let timing_count = timings.len() as f64;
    let avg_frontend = timings
        .iter()
        .map(|timing| timing.frontend_seconds)
        .sum::<f64>()
        / timing_count;
    let avg_forward = timings
        .iter()
        .map(|timing| timing.forward_seconds)
        .sum::<f64>()
        / timing_count;
    let avg_segmenter = timings
        .iter()
        .map(|timing| timing.segmenter_seconds)
        .sum::<f64>()
        / timing_count;
    let avg_ops = average_forward_ops(&timings);
    println!(
        "burn_fsmn_vad_bench audio={} model={} duration_ms={} segments={} stream_segments={} warmup={} repeat={} stream_chunk_ms={}",
        args.audio.display(),
        burn.model_dir().display(),
        waveform.duration_ms() as u64,
        burn_segments.len(),
        stream_segments.len(),
        args.warmup,
        args.repeat,
        args.stream_chunk_ms,
    );
    println!(
        "avg_ms={:.3} min_ms={:.3} max_ms={:.3} rtf={:.6} speedup={:.2}x",
        avg * 1000.0,
        runs.iter()
            .map(Duration::as_secs_f64)
            .fold(f64::INFINITY, f64::min)
            * 1000.0,
        runs.iter().map(Duration::as_secs_f64).fold(0.0, f64::max) * 1000.0,
        avg / waveform.duration_seconds(),
        waveform.duration_seconds() / avg,
    );
    let stream_avg =
        stream_runs.iter().map(Duration::as_secs_f64).sum::<f64>() / stream_runs.len() as f64;
    println!(
        "stream_avg_ms={:.3} stream_min_ms={:.3} stream_max_ms={:.3} stream_rtf={:.6} stream_speedup={:.2}x",
        stream_avg * 1000.0,
        stream_runs
            .iter()
            .map(Duration::as_secs_f64)
            .fold(f64::INFINITY, f64::min)
            * 1000.0,
        stream_runs
            .iter()
            .map(Duration::as_secs_f64)
            .fold(0.0, f64::max)
            * 1000.0,
        stream_avg / waveform.duration_seconds(),
        waveform.duration_seconds() / stream_avg,
    );
    println!(
        "diag_components_ms frontend={:.3} forward={:.3} segmenter={:.3}",
        avg_frontend * 1000.0,
        avg_forward * 1000.0,
        avg_segmenter * 1000.0,
    );
    println!(
        "diag_forward_ops_ms input_tensor={:.3} in_linear1={:.3} in_linear2={:.3} out_linear1={:.3} out_linear2={:.3} softmax={:.3} output_tensor={:.3}",
        avg_ops.input_tensor_seconds * 1000.0,
        avg_ops.in_linear1_seconds * 1000.0,
        avg_ops.in_linear2_seconds * 1000.0,
        avg_ops.out_linear1_seconds * 1000.0,
        avg_ops.out_linear2_seconds * 1000.0,
        avg_ops.softmax_seconds * 1000.0,
        avg_ops.output_tensor_seconds * 1000.0,
    );
    for idx in 0..4 {
        println!(
            "diag_forward_block_{idx}_ms linear={:.3} memory={:.3} affine={:.3}",
            avg_ops.block_linear_seconds[idx] * 1000.0,
            avg_ops.block_memory_seconds[idx] * 1000.0,
            avg_ops.block_affine_seconds[idx] * 1000.0,
        );
    }
    Ok(())
}

fn detect_streaming(
    model: &BurnFsmnVadModel,
    waveform: &Waveform,
    options: &VadOptions,
    chunk_ms: u64,
) -> Result<Vec<vad_burn::VadSegment>> {
    let mut stream = model.stream(options.clone());
    let chunk_samples = ((waveform.sample_rate as u64 * chunk_ms) / 1000).max(1) as usize;
    let mut segments = Vec::new();
    let mut offset = 0usize;
    while offset < waveform.samples.len() {
        let end = (offset + chunk_samples).min(waveform.samples.len());
        let is_final = end >= waveform.samples.len();
        segments.extend(stream.accept(
            &waveform.samples[offset..end],
            waveform.sample_rate,
            is_final,
        )?);
        offset = end;
    }
    Ok(segments)
}

fn average_forward_ops(timings: &[vad_burn::BurnFsmnVadTiming]) -> BurnFsmnForwardTiming {
    let count = timings.len() as f64;
    let mut avg = BurnFsmnForwardTiming::default();
    for timing in timings {
        let ops = timing.forward_ops;
        avg.input_tensor_seconds += ops.input_tensor_seconds / count;
        avg.in_linear1_seconds += ops.in_linear1_seconds / count;
        avg.in_linear2_seconds += ops.in_linear2_seconds / count;
        avg.out_linear1_seconds += ops.out_linear1_seconds / count;
        avg.out_linear2_seconds += ops.out_linear2_seconds / count;
        avg.softmax_seconds += ops.softmax_seconds / count;
        avg.output_tensor_seconds += ops.output_tensor_seconds / count;
        for idx in 0..4 {
            avg.block_linear_seconds[idx] += ops.block_linear_seconds[idx] / count;
            avg.block_memory_seconds[idx] += ops.block_memory_seconds[idx] / count;
            avg.block_affine_seconds[idx] += ops.block_affine_seconds[idx] / count;
        }
    }
    avg
}

struct Args {
    audio: PathBuf,
    model: PathBuf,
    warmup: usize,
    repeat: usize,
    stream_chunk_ms: u64,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut audio = PathBuf::from("assets/vad_example.wav");
        let mut model = None;
        let mut warmup = 1usize;
        let mut repeat = 5usize;
        let mut stream_chunk_ms = 600u64;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--audio" => audio = next_path(&mut args, "--audio")?,
                "--model" => model = Some(next_path(&mut args, "--model")?),
                "--warmup" => warmup = next_usize(&mut args, "--warmup")?,
                "--repeat" => repeat = next_usize(&mut args, "--repeat")?,
                "--stream-chunk-ms" => stream_chunk_ms = next_u64(&mut args, "--stream-chunk-ms")?,
                "-h" | "--help" => {
                    println!(
                        "Usage: cargo run -p vad-burn --example bench_fsmn_vad -- --audio WAV --model MODEL_DIR [--stream-chunk-ms 600]"
                    );
                    std::process::exit(0);
                }
                other if !other.starts_with('-') => audio = PathBuf::from(other),
                other => bail!("unknown argument {other:?}"),
            }
        }
        if repeat == 0 {
            bail!("--repeat must be greater than zero");
        }
        let model = model
            .or_else(default_model_path)
            .ok_or_else(|| anyhow::anyhow!("pass --model PATH to fsmn-vad model dir"))?;
        Ok(Self {
            audio,
            model,
            warmup,
            repeat,
            stream_chunk_ms,
        })
    }
}

fn next_path(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(args.next().ok_or_else(|| {
        anyhow::anyhow!("{name} requires a value")
    })?))
}

fn next_usize(args: &mut impl Iterator<Item = String>, name: &str) -> Result<usize> {
    Ok(args
        .next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))?
        .parse()?)
}

fn next_u64(args: &mut impl Iterator<Item = String>, name: &str) -> Result<u64> {
    Ok(args
        .next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))?
        .parse()?)
}

fn default_model_path() -> Option<PathBuf> {
    [
        PathBuf::from("/workspace/data/models/asr/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        PathBuf::from(".cache/fsmn-vad"),
    ]
    .into_iter()
    .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists())
}

fn load_pcm16_wav(path: &Path) -> Result<Waveform> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("expected RIFF/WAVE file");
    }
    let mut offset = 12usize;
    let mut sample_rate = None;
    let mut channels = None;
    let mut bits_per_sample = None;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        offset += 8;
        if offset + chunk_size > bytes.len() {
            bail!("invalid WAV chunk size");
        }
        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    bail!("invalid fmt chunk");
                }
                let audio_format = u16::from_le_bytes(bytes[offset..offset + 2].try_into()?);
                if audio_format != 1 {
                    bail!("expected PCM WAV");
                }
                channels = Some(u16::from_le_bytes(
                    bytes[offset + 2..offset + 4].try_into()?,
                ));
                sample_rate = Some(u32::from_le_bytes(
                    bytes[offset + 4..offset + 8].try_into()?,
                ));
                bits_per_sample = Some(u16::from_le_bytes(
                    bytes[offset + 14..offset + 16].try_into()?,
                ));
            }
            b"data" => data = Some(bytes[offset..offset + chunk_size].to_vec()),
            _ => {}
        }
        offset += chunk_size + (chunk_size % 2);
    }
    let sample_rate = sample_rate.ok_or_else(|| anyhow::anyhow!("missing fmt chunk"))?;
    let channels = channels.ok_or_else(|| anyhow::anyhow!("missing channel count"))?;
    if bits_per_sample != Some(16) {
        bail!("expected 16-bit PCM WAV");
    }
    let data = data.ok_or_else(|| anyhow::anyhow!("missing data chunk"))?;
    let samples = data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
        .collect();
    Ok(Waveform::new_with_channels(samples, sample_rate, channels))
}
