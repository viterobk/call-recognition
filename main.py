from collections import deque
from pathlib import Path
from sys import stdout
from time import perf_counter
from typing import Iterator

import gigaam
import numpy as np
import torch
from gigaam.utils import AudioDataset

PROJECT_DIR = Path(__file__).resolve().parent
AUDIO_FILE = PROJECT_DIR / "speech.mp3"
RESULT_FILE = PROJECT_DIR / "result.txt"
MODEL_NAME = "v3_e2e_rnnt"
SPLIT_INTO_BATCHES = True

SAMPLE_RATE = 16_000
STREAM_CHUNK_MS = 30
PAUSE_MS = 300
MIN_REPLICA_MS = 2000
PAD_MS = 150
SPEECH_RMS = 0.015
MAX_SEGMENT_S = 25


def load_audio(audio_path: Path) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return gigaam.load_audio(str(audio_path), sample_rate=SAMPLE_RATE).numpy()


def load_audio_stream(audio_path: Path, chunk_ms: int = STREAM_CHUNK_MS) -> Iterator[np.ndarray]:
    audio = load_audio(audio_path)
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


def split_for_model(audio: np.ndarray) -> list[np.ndarray]:
    max_samples = MAX_SEGMENT_S * SAMPLE_RATE
    if len(audio) <= max_samples:
        return [audio]
    return [audio[start : start + max_samples] for start in range(0, len(audio), max_samples)]


def transcribe_batch(model, audios: list[np.ndarray]) -> list[str]:
    if not audios:
        return []
    wavs = [torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)) for audio in audios]
    wav_pad, wav_lens = AudioDataset.collate(wavs)
    wav_pad = wav_pad.to(device=model._device, dtype=model._dtype)
    wav_lens = wav_lens.to(model._device)
    with torch.inference_mode():
        encoded, encoded_len = model(wav_pad, wav_lens)
        decoded = model._decode(encoded, encoded_len, wav_lens, False)
    if model._device.type == "cuda":
        torch.cuda.synchronize()
    return [text.strip() for text, _words in decoded]


def transcribe_replica(model, audio: np.ndarray) -> tuple[str, float]:
    started = perf_counter()
    parts = [text for piece in split_for_model(audio) if (text := transcribe_batch(model, [piece])[0])]
    if model._device.type == "cuda":
        torch.cuda.synchronize()
    return " ".join(parts).strip(), perf_counter() - started


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


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = gigaam.load_model(MODEL_NAME, device=device)
    if device == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("device: cpu", flush=True)
    return model


def recognize(model, replicas: list[tuple[np.ndarray, float]], lines: list[str]) -> None:
    for audio, start_s in replicas:
        text, rec_s = transcribe_replica(model, audio)
        emit(text, start_s, len(audio) / SAMPLE_RATE, rec_s, lines)


def main() -> None:
    stdout.reconfigure(line_buffering=True)
    model = load_model()
    lines: list[str] = []

    if not SPLIT_INTO_BATCHES:
        audio = load_audio(AUDIO_FILE)
        max_samples = MAX_SEGMENT_S * SAMPLE_RATE
        replicas = [
            (audio[start : start + max_samples], start / SAMPLE_RATE)
            for start in range(0, len(audio), max_samples)
        ]
    else:
        replicas = list(split_stream_by_pauses(load_audio_stream(AUDIO_FILE)))

    recognize(model, replicas, lines)


if __name__ == "__main__":
    main()
