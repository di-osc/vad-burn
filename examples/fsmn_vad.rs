//! FSMN VAD：离线 annotate，以及按 chunk 流式 annotate。

use std::path::PathBuf;

use anyhow::Result;
use vad_burn::{Audio, AudioStream, FsmnVadModel, VadOptions};

/// 仓库内置示例音频路径。
fn example_wav() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/vad_example.wav")
}

fn main() -> Result<()> {
    let model = FsmnVadModel::from_modelscope()?;
    let options = VadOptions::default();
    let wav = example_wav();

    // 离线：整段音频按声道标注，采样率在 annotate 内部处理。
    let mut audio = Audio::from_path(&wav)?;
    model.annotate(&mut audio, &options)?;
    println!("{audio}");

    // 流式：调用方自己迭代 chunk，才能使用每一块的中间结果。
    let mut stream = AudioStream::from_path(&wav, 600)?;
    let mut session = model.new_session(options);
    while let Some(chunk) = stream.next() {
        let chunk = chunk?;
        let spans = session.annotate(&mut stream, &chunk)?;
        println!(
            "chunk {} offset={}ms final={} new_spans={}",
            chunk.index,
            chunk.offset_ms,
            chunk.is_final,
            spans.len()
        );
        for span in &spans {
            println!("  {}-{} ms", span.range.start_ms, span.range.end_ms);
        }
    }
    println!("{}", stream.into_audio()?);
    Ok(())
}
