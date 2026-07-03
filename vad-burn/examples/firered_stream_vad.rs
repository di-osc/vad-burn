use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use vad_burn::{FireRedVadModel, VadOptions, Waveform};

fn main() -> Result<()> {
    let audio = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/vad_example.wav"));
    let chunk_ms = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(600);
    let waveform = load_pcm16_wav(&audio)?;
    let model = FireRedVadModel::from_modelscope()?;
    let mut stream = model.new_stream(VadOptions::default());
    let chunk_samples = (waveform.sample_rate as u64 * chunk_ms / 1000) as usize;
    let mut segments = Vec::new();
    for chunk in waveform.samples.chunks(chunk_samples.max(1)) {
        segments.extend(stream.push(chunk, waveform.sample_rate)?);
    }
    let frames = stream.frame_scores().len();
    segments.extend(stream.finish()?);
    println!(
        "firered_stream_vad audio={} model={} chunk_ms={} frames={} segments={}",
        audio.display(),
        model.stream_model_dir().display(),
        chunk_ms,
        frames,
        segments.len(),
    );
    for segment in segments {
        println!(
            "{} {} {:.3}",
            segment.range.start.0, segment.range.end.0, segment.probability
        );
    }
    Ok(())
}

fn load_pcm16_wav(path: &Path) -> Result<Waveform> {
    let bytes = std::fs::read(path)?;
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

    let data = data.ok_or_else(|| anyhow::anyhow!("missing data chunk"))?;
    if bits_per_sample != 16 {
        bail!("expected 16-bit PCM WAV");
    }
    let samples = data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().expect("i16")) as f32 / 32768.0)
        .collect();
    Ok(Waveform::new_with_channels(samples, sample_rate, channels))
}
