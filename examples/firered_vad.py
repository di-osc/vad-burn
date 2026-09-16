from pathlib import Path

from asr_data import Audio, AudioStream
from vad_burn import FireRedVadModel, VadOptions


def example_wav() -> Path:
    """返回仓库内置示例音频路径。"""
    return Path(__file__).resolve().parents[1] / "assets" / "vad_example.wav"


def main() -> None:
    """演示 FireRedVAD 离线 annotate 与按 chunk 流式 annotate。"""
    model = FireRedVadModel.from_modelscope()
    options = VadOptions(threshold=0.8, min_silence_ms=800)
    wav = example_wav()

    # 离线：整段音频按声道标注，采样率在 annotate 内部处理。
    audio = Audio.from_path(wav)
    model.annotate(audio, options)
    print(audio)

    # 流式：调用方自己迭代 chunk，才能使用每一块的中间结果。
    stream = AudioStream.from_path(wav, 600)
    session = model.new_session(options)
    for chunk in stream:
        spans = session.annotate(chunk)
        print(
            f"chunk {chunk.index} offset={chunk.offset_ms}ms "
            f"final={chunk.is_final} new_spans={len(spans)}"
        )
        for span in spans:
            print(f"  {span.start_ms}-{span.end_ms} ms")
    print(stream.to_audio())


if __name__ == "__main__":
    main()
