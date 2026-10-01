from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from time import perf_counter, sleep
from typing import Iterator

import gigaam
import numpy as np

from transcription import MAX_SEGMENT_S, SAMPLE_RATE, TranscriptionService

STREAM_CHUNK_MS = 30
PAUSE_MS = 300
MIN_REPLICA_MS = 2000
PAD_MS = 150
SPEECH_RMS = 0.015
SPLIT_ON_PAUSES = True
REALTIME_DELAY = False

PRINT_LOCK = Lock()


@dataclass(frozen=True)
class FileJobResult:
    worker_id: int
    wall_s: float
    audio_s: float
    skipped_s: float
    decode_s: float
    replicas: int


def load_audio(audio_path: Path) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return gigaam.load_audio(str(audio_path), sample_rate=SAMPLE_RATE).numpy()


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


def audio_batches(audio: np.ndarray, *, split_on_pauses: bool = SPLIT_ON_PAUSES) -> list[tuple[np.ndarray, float]]:
    if not split_on_pauses:
        max_samples = MAX_SEGMENT_S * SAMPLE_RATE
        return [
            (audio[start : start + max_samples], start / SAMPLE_RATE)
            for start in range(0, len(audio), max_samples)
        ]

    chunk_size = int(SAMPLE_RATE * STREAM_CHUNK_MS / 1000)

    def stream() -> Iterator[np.ndarray]:
        for start in range(0, len(audio), chunk_size):
            yield audio[start : start + chunk_size]

    return list(split_stream_by_pauses(stream()))


def process_audio_file(
    audio_path: Path,
    service: TranscriptionService,
    result_path: Path,
    worker_id: int,
) -> FileJobResult:
    started = perf_counter()
    audio = load_audio(audio_path)
    file_s = len(audio) / SAMPLE_RATE
    all_batches = audio_batches(audio)
    skipped = all_batches[:worker_id]
    batches = all_batches[worker_id:]
    skipped_s = sum(len(batch) for batch, _start_s in skipped) / SAMPLE_RATE
    audio_s = file_s - skipped_s
    recognized: Queue[tuple[float, float, Future[tuple[str, float]]] | None] = Queue()
    lines: list[str] = []
    decode_s = 0.0
    logged_error: list[BaseException] = []

    def log_when_ready() -> None:
        nonlocal decode_s
        try:
            while True:
                item = recognized.get()
                if item is None:
                    return
                start_s, duration_s, future = item
                text, rec_s = future.result()
                decode_s += rec_s
                if not text:
                    continue
                line = f"[w{worker_id} | {start_s:.1f}s | {duration_s:.1f}s | {rec_s:.2f}s] {text}"
                with PRINT_LOCK:
                    print(line, flush=True)
                lines.append(line)
                result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as error:
            logged_error.append(error)

    logger = Thread(target=log_when_ready, name=f"log-{worker_id}")
    logger.start()
    try:
        for batch, start_s in batches:
            duration_s = len(batch) / SAMPLE_RATE
            if REALTIME_DELAY:
                sleep(duration_s)
            recognized.put((start_s, duration_s, service.submit(batch)))
    finally:
        recognized.put(None)
        logger.join()
    if logged_error:
        raise logged_error[0]
    return FileJobResult(
        worker_id=worker_id,
        wall_s=perf_counter() - started,
        audio_s=audio_s,
        skipped_s=skipped_s,
        decode_s=decode_s,
        replicas=len(batches),
    )
