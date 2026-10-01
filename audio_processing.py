from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from time import perf_counter, sleep
from typing import Iterator

import gigaam
import numpy as np

from summary import SummaryService
from transcription import MAX_SEGMENT_S, SAMPLE_RATE, TranscriptionService

STREAM_CHUNK_MS = 30
PAUSE_MS = 300
MIN_REPLICA_MS = 2000
PAD_MS = 150
SPEECH_RMS = 0.015
SPLIT_ON_PAUSES = True
REALTIME_DELAY = True
SUMMARY_EVERY_S = 30.0

PRINT_LOCK = Lock()


@dataclass(frozen=True)
class FileJobResult:
    name: str
    wall_s: float
    audio_s: float
    decode_s: float
    summary_s: float
    replicas: int
    summaries: int


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


class ResultLog:
    def __init__(self, path: Path, started: float) -> None:
        self._path = path
        self._started = started
        self._lines: list[str] = []
        self._lock = Lock()

    def write(self, text: str, audio_end_s: float) -> None:
        self._append(f"[{self._stamp(audio_end_s)}] {text}")

    def write_summary(self, label: str, text: str, audio_end_s: float) -> None:
        block = "\n".join(
            [
                "----------------------------------------",
                f"[{self._stamp(audio_end_s)}] {label}",
                text.strip(),
                "----------------------------------------",
            ]
        )
        self._append(block)

    def _stamp(self, audio_end_s: float) -> str:
        lag_s = perf_counter() - self._started - audio_end_s
        return f"{audio_end_s:.1f}s | +{lag_s:.2f}s"

    def _append(self, block: str) -> None:
        clock = datetime.now().strftime("%H:%M:%S")
        stamped = "\n".join(f"{clock} {line}" for line in block.split("\n"))
        with self._lock:
            self._lines.append(stamped)
            self._path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
        with PRINT_LOCK:
            print(f"{self._path.stem} {stamped}", flush=True)


def process_audio_file(
    audio_path: Path,
    transcription: TranscriptionService,
    summary: SummaryService,
    result_path: Path,
    worker_id: int,
) -> FileJobResult:
    started = perf_counter()
    audio = load_audio(audio_path)
    audio_s = len(audio) / SAMPLE_RATE
    batches = audio_batches(audio)
    stream_started = perf_counter()
    log = ResultLog(result_path, stream_started)
    recognized: Queue[tuple[float, float, Future[tuple[str, float]]] | None] = Queue()
    summaries: Queue[tuple[float, str, Future[tuple[str, float]]] | None] = Queue()
    decode_s = 0.0
    summary_s = 0.0
    summary_count = 0
    logged_error: list[BaseException] = []

    def log_summaries() -> None:
        nonlocal summary_s, summary_count
        try:
            while True:
                item = summaries.get()
                if item is None:
                    return
                audio_end_s, label, future = item
                text, elapsed_s = future.result()
                summary_s += elapsed_s
                summary_count += 1
                log.write_summary(label, text, audio_end_s)
        except Exception as error:
            logged_error.append(error)

    def log_when_ready() -> None:
        nonlocal decode_s
        pending_text: list[str] = []
        full_text: list[str] = []
        pending_end_s = 0.0
        call_end_s = 0.0
        next_summary_at = SUMMARY_EVERY_S

        def submit_window(force: bool = False) -> None:
            nonlocal pending_end_s, next_summary_at
            if not force and pending_end_s < next_summary_at:
                return
            text = "\n".join(pending_text)
            audio_end_s = pending_end_s
            pending_text.clear()
            pending_end_s = 0.0
            if not force:
                while next_summary_at <= audio_end_s:
                    next_summary_at += SUMMARY_EVERY_S
            if text.strip():
                if not force:
                    log.write(">>> запрос резюме <<<", audio_end_s)
                summaries.put((audio_end_s, "резюме", summary.submit(text)))

        try:
            while True:
                item = recognized.get()
                if item is None:
                    break
                start_s, duration_s, future = item
                text, rec_s = future.result()
                decode_s += rec_s
                pending_end_s = start_s + duration_s
                call_end_s = pending_end_s
                if text:
                    log.write(text, pending_end_s)
                    pending_text.append(text)
                    full_text.append(text)
                submit_window()
            submit_window(force=True)
            if full_text:
                summaries.put((call_end_s, "резюме звонка", summary.submit("\n".join(full_text))))
        except Exception as error:
            logged_error.append(error)
        finally:
            summaries.put(None)
            summary_logger.join()

    summary_logger = Thread(target=log_summaries, name=f"summary-log-{worker_id}")
    logger = Thread(target=log_when_ready, name=f"log-{worker_id}")
    summary_logger.start()
    logger.start()
    try:
        for batch, start_s in batches:
            duration_s = len(batch) / SAMPLE_RATE
            if REALTIME_DELAY:
                remaining_s = (start_s + duration_s) - (perf_counter() - stream_started)
                if remaining_s > 0:
                    sleep(remaining_s)
            recognized.put((start_s, duration_s, transcription.submit(batch)))
    finally:
        recognized.put(None)
        logger.join()
    if logged_error:
        raise logged_error[0]
    return FileJobResult(
        name=audio_path.stem,
        wall_s=perf_counter() - started,
        audio_s=audio_s,
        decode_s=decode_s,
        summary_s=summary_s,
        replicas=len(batches),
        summaries=summary_count,
    )
