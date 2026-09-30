import builtins
from collections import deque
from pathlib import Path
from queue import Queue
from re import match
from sys import stdout
from time import perf_counter
from typing import Iterator

import numpy as np
import whisper

PROJECT_DIR = Path(__file__).resolve().parent
AUDIO_FILE = PROJECT_DIR / "speech.mp3"
RESULT_FILE = PROJECT_DIR / "result.txt"
MODEL_NAME = "large-v3-turbo"
SPLIT_INTO_BATCHES = True

SAMPLE_RATE = 16_000
STREAM_CHUNK_MS = 30
PAUSE_MS = 500
MIN_REPLICA_MS = 1000
PAD_MS = 150
SPEECH_RMS = 0.015


def load_audio_stream(audio_path: Path, chunk_ms: int = STREAM_CHUNK_MS) -> Iterator[np.ndarray]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = whisper.load_audio(str(audio_path), sr=SAMPLE_RATE)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    for start in range(0, len(audio), chunk_size):
        yield audio[start : start + chunk_size]


def split_stream_by_pauses(
    stream: Iterator[np.ndarray],
    *,
    pause_ms: int = PAUSE_MS,
    min_replica_ms: int = MIN_REPLICA_MS,
    pad_ms: int = PAD_MS,
    speech_rms: float = SPEECH_RMS,
    chunk_ms: int = STREAM_CHUNK_MS,
) -> Iterator[tuple[np.ndarray, float]]:
    pause_chunks = max(1, pause_ms // chunk_ms)
    min_chunks = max(1, min_replica_ms // chunk_ms)
    pad_chunks = max(1, pad_ms // chunk_ms)

    speech_chunks: list[np.ndarray] = []
    trailing_silence: list[np.ndarray] = []
    leading_pad: deque[np.ndarray] = deque(maxlen=pad_chunks)
    in_speech = False
    replica_start_chunk = 0

    def build_replica() -> np.ndarray | None:
        if len(speech_chunks) < min_chunks:
            return None
        padded = [*leading_pad, *speech_chunks, *trailing_silence[:pad_chunks]]
        return np.concatenate(padded)

    def replica_start_s() -> float:
        return replica_start_chunk * chunk_ms / 1000

    for chunk_index, chunk in enumerate(stream):
        is_speech = float(np.sqrt(np.mean(np.square(chunk)))) >= speech_rms
        if is_speech:
            if in_speech:
                speech_chunks.extend(trailing_silence)
                trailing_silence.clear()
            else:
                replica_start_chunk = chunk_index
            in_speech = True
            speech_chunks.append(chunk)
            continue

        if not in_speech:
            leading_pad.append(chunk)
            continue

        trailing_silence.append(chunk)
        if len(trailing_silence) < pause_chunks:
            continue

        replica = build_replica()
        if replica is not None:
            yield replica, replica_start_s()

        speech_chunks.clear()
        trailing_silence.clear()
        leading_pad.clear()
        in_speech = False

    replica = build_replica()
    if replica is not None:
        yield replica, replica_start_s()


def parse_whisper_segment_line(message: str) -> tuple[float, float, str] | None:
    found = match(r"^\[(.+?) --> (.+?)\] ?(.*)$", message.strip())
    if found is None:
        return None
    start_s = parse_timestamp(found.group(1))
    end_s = parse_timestamp(found.group(2))
    text = found.group(3).strip()
    if not text:
        return None
    return start_s, max(0.0, end_s - start_s), text


def parse_timestamp(stamp: str) -> float:
    parts = stamp.split(":")
    if len(parts) == 2:
        minutes, rest = parts
        hours = 0
    else:
        hours, minutes, rest = parts
    seconds, milliseconds = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def transcribe_replica(model, audio: np.ndarray, *, use_context: bool) -> tuple[str, float]:
    started = perf_counter()
    result = model.transcribe(
        audio,
        language="ru",
        fp16=False,
        condition_on_previous_text=use_context,
        verbose=None,
    )
    elapsed = perf_counter() - started
    return result["text"].strip(), elapsed


def transcribe_whole_file(model, audio: np.ndarray, lines: list[str]) -> None:
    original_print = builtins.print
    last_mark = perf_counter()
    last_rec_s = 0.0

    def hooked_print(*args, **kwargs) -> None:
        nonlocal last_mark, last_rec_s
        message = " ".join(str(argument) for argument in args)
        parsed = parse_whisper_segment_line(message)
        if parsed is not None:
            start_s, duration_s, text = parsed
            now = perf_counter()
            rec_s = now - last_mark
            if rec_s < 0.08:
                rec_s = last_rec_s
            else:
                last_rec_s = rec_s
                last_mark = now
            emit(text, start_s, duration_s, rec_s, lines, printer=original_print)
            return
        if message.strip():
            original_print(*args, **{**kwargs, "flush": True})

    builtins.print = hooked_print
    try:
        model.transcribe(
            audio,
            language="ru",
            fp16=False,
            condition_on_previous_text=False,
            verbose=True,
        )
    finally:
        builtins.print = original_print


def emit(
    text: str,
    start_s: float,
    duration_s: float,
    rec_s: float,
    lines: list[str],
    *,
    printer=print,
) -> None:
    if not text:
        return
    line = f"[{start_s:.1f}s | {duration_s:.1f}s | {rec_s:.2f}s] {text}"
    printer(line, flush=True)
    lines.append(line)
    RESULT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stdout.reconfigure(line_buffering=True)
    model = whisper.load_model(MODEL_NAME)
    lines: list[str] = []

    if not SPLIT_INTO_BATCHES:
        if not AUDIO_FILE.exists():
            raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")
        audio = whisper.load_audio(str(AUDIO_FILE))
        transcribe_whole_file(model, audio, lines)
        return

    batches: Queue = Queue()
    for batch, start_s in split_stream_by_pauses(load_audio_stream(AUDIO_FILE)):
        batches.put((batch, start_s))
        replica, start_s = batches.get()
        text, rec_s = transcribe_replica(model, replica, use_context=False)
        duration_s = len(replica) / SAMPLE_RATE
        emit(text, start_s, duration_s, rec_s, lines)


if __name__ == "__main__":
    main()
