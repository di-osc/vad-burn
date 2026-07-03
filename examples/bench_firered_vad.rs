use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use vad_burn::{FireRedVadModel, VadOptions, VadSegment, Waveform};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let waveform = load_pcm16_wav(&args.audio)?;
    if waveform.sample_rate != 16_000 || waveform.channels != 1 {
        bail!("expected 16kHz mono WAV");
    }

    let options = VadOptions::default();
    let model = if let Some(model_dir) = &args.model {
        FireRedVadModel::from_pretrained(model_dir)?
    } else {
        FireRedVadModel::from_modelscope()?
    };

    for _ in 0..args.warmup {
        let _ = model.detect(&waveform, &options)?;
        let _ = detect_streaming(&model, &waveform, &options, args.stream_chunk_ms)?;
    }

    let mut offline_runs = Vec::with_capacity(args.repeat);
    let mut offline_segments = Vec::new();
    for _ in 0..args.repeat {
        let start = Instant::now();
        offline_segments = model.detect(&waveform, &options)?;
        offline_runs.push(start.elapsed());
    }

    let mut stream_runs = Vec::with_capacity(args.repeat);
    let mut stream_segments = Vec::new();
    for _ in 0..args.repeat {
        let start = Instant::now();
        stream_segments = detect_streaming(&model, &waveform, &options, args.stream_chunk_ms)?;
        stream_runs.push(start.elapsed());
    }

    let diagnostic = model.detect_with_timing(&waveform, &options)?;
    println!(
        "firered_vad_bench audio={} model={} offline_model={} stream_model={} duration_ms={} offline_segments={} stream_segments={} warmup={} repeat={} stream_chunk_ms={}",
        args.audio.display(),
        model.model_dir().display(),
        model.offline_model_dir().display(),
        model.stream_model_dir().display(),
        waveform.duration_ms() as u64,
        offline_segments.len(),
        stream_segments.len(),
        args.warmup,
        args.repeat,
        args.stream_chunk_ms,
    );
    print_stats("offline", &offline_runs, waveform.duration_seconds());
    print_stats("stream", &stream_runs, waveform.duration_seconds());
    println!(
        "offline_diag_components_ms frontend={:.3} forward={:.3} postprocess={:.3} frames={}",
        diagnostic.timing.frontend_seconds * 1000.0,
        diagnostic.timing.forward_seconds * 1000.0,
        diagnostic.timing.postprocess_seconds * 1000.0,
        diagnostic.timing.frames,
    );
    Ok(())
}

fn detect_streaming(
    model: &FireRedVadModel,
    waveform: &Waveform,
    options: &VadOptions,
    chunk_ms: u64,
) -> Result<Vec<VadSegment>> {
    let mut stream = model.new_stream(options.clone());
    let chunk_samples = ((waveform.sample_rate as u64 * chunk_ms) / 1000).max(1) as usize;
    let mut segments = Vec::new();
    let mut offset = 0usize;
    while offset < waveform.samples.len() {
        let end = (offset + chunk_samples).min(waveform.samples.len());
        segments.extend(stream.push(&waveform.samples[offset..end], waveform.sample_rate)?);
        offset = end;
    }
    segments.extend(stream.finish()?);
    Ok(segments)
}

fn print_stats(label: &str, runs: &[Duration], duration_seconds: f64) {
    let avg = runs.iter().map(Duration::as_secs_f64).sum::<f64>() / runs.len() as f64;
    let min = runs
        .iter()
        .map(Duration::as_secs_f64)
        .fold(f64::INFINITY, f64::min);
    let max = runs.iter().map(Duration::as_secs_f64).fold(0.0, f64::max);
    println!(
        "{label}_avg_ms={:.3} {label}_min_ms={:.3} {label}_max_ms={:.3} {label}_rtf={:.6} {label}_speedup={:.2}x",
        avg * 1000.0,
        min * 1000.0,
        max * 1000.0,
        avg / duration_seconds,
        duration_seconds / avg,
    );
}

struct Args {
    audio: PathBuf,
    model: Option<PathBuf>,
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
                        "Usage: cargo run -p vad-burn --example bench_firered_vad -- --audio WAV [--model MODEL_DIR] [--stream-chunk-ms 600]"
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

fn load_pcm16_wav(path: &Path) -> Result<Waveform> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("expected RIFF/WAVE file");
    }
    let mut offset = 12usize;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        offset += 8;
        if offset + size > bytes.len() {
            bail!("truncated WAV chunk");
        }
        match id {
            b"fmt " => {
                if size < 16 {
                    bail!("invalid fmt chunk");
                }
                let audio_format = u16::from_le_bytes(bytes[offset..offset + 2].try_into()?);
                channels = u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into()?);
                sample_rate = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?);
                bits_per_sample = u16::from_le_bytes(bytes[offset + 14..offset + 16].try_into()?);
                if audio_format != 1 {
                    bail!("expected PCM WAV");
                }
            }
            b"data" => data = Some(bytes[offset..offset + size].to_vec()),
            _ => {}
        }
        offset += size + (size % 2);
    }
    if bits_per_sample != 16 {
        bail!("expected 16-bit PCM WAV");
    }
    let samples = data
        .ok_or_else(|| anyhow::anyhow!("missing data chunk"))?
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().expect("i16")) as f32 / 32768.0)
        .collect();
    Ok(Waveform::new_with_channels(samples, sample_rate, channels))
}
