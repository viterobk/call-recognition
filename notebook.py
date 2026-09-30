# %% [markdown]
# GigaAM-v3 в Google Colab
#
# 1. **Runtime → Change runtime type → T4 GPU** (на CPU `v3_e2e_rnnt` будет заметно медленнее).
# 2. Запускайте ячейки сверху вниз.
# 3. Если `speech.mp3` ещё нет в `/content`, следующая ячейка откроет диалог загрузки.
# 4. После `%pip` при ошибке импорта: **Runtime → Restart session**, затем снова выполните ячейки с импортами.

# %%
%pip install -q "gigaam @ git+https://github.com/salute-developers/GigaAM.git" torchaudio
!apt-get -qq install -y ffmpeg

# %%
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path
from queue import Queue
from time import perf_counter
from typing import Iterator

import gigaam
import numpy as np
import soundfile as sf
import torch

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    colab_files = None
    IN_COLAB = False

print("colab:", IN_COLAB)
print("cuda:", torch.cuda.is_available())
print("ffmpeg:", shutil.which("ffmpeg"))
if not torch.cuda.is_available():
    print("GPU не найден. В Colab: Runtime → Change runtime type → T4 GPU.")

# %%
PROJECT_DIR = Path("/content") if IN_COLAB else Path.cwd()
AUDIO_FILE = PROJECT_DIR / "speech.mp3"
RESULT_FILE = PROJECT_DIR / "result.txt"

if not AUDIO_FILE.exists():
    if IN_COLAB:
        print("Загрузите speech.mp3")
        uploaded = colab_files.upload()
        for name, data in uploaded.items():
            target = PROJECT_DIR / name
            target.write_bytes(data)
            if name != AUDIO_FILE.name:
                AUDIO_FILE.write_bytes(data)
    else:
        raise FileNotFoundError(f"Нет файла {AUDIO_FILE}")

print("audio:", AUDIO_FILE, "size:", AUDIO_FILE.stat().st_size)

# %%
MODEL_NAME = "v3_e2e_rnnt"
SPLIT_INTO_BATCHES = True

SAMPLE_RATE = 16_000
STREAM_CHUNK_MS = 30
PAUSE_MS = 500
MIN_REPLICA_MS = 1000
PAD_MS = 150
SPEECH_RMS = 0.015
MAX_SEGMENT_S = 25

print("model:", MODEL_NAME)

# %%
def load_audio(audio_path: Path) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"Нет файла {audio_path}. Загрузите speech.mp3 в Colab.")
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


def transcribe_chunk(model, audio: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, audio, SAMPLE_RATE)
        return str(model.transcribe(path)).strip()
    finally:
        Path(path).unlink(missing_ok=True)


def transcribe_waveform(model, audio: np.ndarray) -> str:
    max_samples = MAX_SEGMENT_S * SAMPLE_RATE
    parts: list[str] = []
    for start in range(0, len(audio), max_samples):
        text = transcribe_chunk(model, audio[start : start + max_samples])
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def transcribe_replica(model, audio: np.ndarray) -> tuple[str, float]:
    started = perf_counter()
    text = transcribe_waveform(model, audio)
    return text, perf_counter() - started


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


def transcribe_whole_file(model, audio: np.ndarray, lines: list[str]) -> None:
    max_samples = MAX_SEGMENT_S * SAMPLE_RATE
    for start in range(0, len(audio), max_samples):
        chunk = audio[start : start + max_samples]
        started = perf_counter()
        text = transcribe_chunk(model, chunk)
        emit(
            text,
            start / SAMPLE_RATE,
            len(chunk) / SAMPLE_RATE,
            perf_counter() - started,
            lines,
        )


def transcribe() -> list[str]:
    lines: list[str] = []

    if not SPLIT_INTO_BATCHES:
        transcribe_whole_file(model, load_audio(AUDIO_FILE), lines)
        return lines

    batches: Queue = Queue()
    for batch, start_s in split_stream_by_pauses(load_audio_stream(AUDIO_FILE)):
        batches.put((batch, start_s))
        replica, start_s = batches.get()
        text, rec_s = transcribe_replica(model, replica)
        duration_s = len(replica) / SAMPLE_RATE
        emit(text, start_s, duration_s, rec_s, lines)
    return lines

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = gigaam.load_model(MODEL_NAME, device=device)
model

# %%
lines = transcribe()
lines

# %%
if IN_COLAB and RESULT_FILE.exists():
    colab_files.download(str(RESULT_FILE))
