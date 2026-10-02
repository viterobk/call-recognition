from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from time import perf_counter, sleep

import gigaam
import numpy as np

from summary import SummaryService
from transcription import MAX_SEGMENT_S, SAMPLE_RATE, TranscriptionService

STREAM_CHUNK_MS = 30
PAUSE_MS = 300
SPEECH_RMS = 0.008
SPLIT_ON_PAUSES = True
REALTIME_DELAY = True
SUMMARY_EVERY_S = 10.0

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


def split_on_silence(
    audio: np.ndarray,
    *,
    pause_ms: int = PAUSE_MS,
    speech_rms: float = SPEECH_RMS,
    chunk_ms: int = STREAM_CHUNK_MS,
) -> list[tuple[np.ndarray, float]]:
    if len(audio) == 0:
        return []
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    chunks = [audio[start : start + chunk_size] for start in range(0, len(audio), chunk_size)]
    pause_chunks = max(1, pause_ms // chunk_ms)
    starts = [0]
    silence = 0
    silence_start = 0
    seen_speech = False
    for index, chunk in enumerate(chunks):
        is_speech = float(np.sqrt(np.mean(np.square(chunk)))) >= speech_rms
        if is_speech:
            silence = 0
            seen_speech = True
            continue
        if silence == 0:
            silence_start = index
        silence += 1
        if seen_speech and silence == pause_chunks and silence_start > starts[-1]:
            starts.append(silence_start)
    starts.append(len(chunks))
    segments: list[tuple[np.ndarray, float]] = []
    for begin, end in zip(starts, starts[1:]):
        if begin == end:
            continue
        segments.append((np.concatenate(chunks[begin:end]), begin * chunk_ms / 1000))
    return segments


def audio_batches(audio: np.ndarray, *, split_on_pauses: bool = SPLIT_ON_PAUSES) -> list[tuple[np.ndarray, float]]:
    if not split_on_pauses:
        max_samples = MAX_SEGMENT_S * SAMPLE_RATE
        return [
            (audio[start : start + max_samples], start / SAMPLE_RATE)
            for start in range(0, len(audio), max_samples)
        ]
    return split_on_silence(audio)


class ResultLog:
    def __init__(self, path: Path, started: float) -> None:
        self._path = path
        self._started = started
        self._lines: list[str] = []
        self._lock = Lock()

    def write_note(self, text: str) -> None:
        self._append(text)

    def write(self, text: str, audio_end_s: float) -> None:
        self._append(f"[{self._stamp(audio_end_s)}] {text}")

    def write_summary(self, text: str, audio_end_s: float, generation_s: float, average_s: float) -> None:
        clock = datetime.now().strftime("%H:%M:%S")
        body = "\n".join(f"    {line}" for line in text.strip().split("\n"))
        header = (
            f"{clock} [{self._stamp(audio_end_s)}] ========== РЕЗЮМЕ ========== "
            f"генерация {generation_s:.2f}s среднее 10 {average_s:.2f}s"
        )
        self._append(f"{header}\n{body}", clock_each_line=False)

    def _stamp(self, audio_end_s: float) -> str:
        lag_s = perf_counter() - self._started - audio_end_s
        return f"{audio_end_s:.1f}s | +{lag_s:.2f}s"

    def _append(self, block: str, *, clock_each_line: bool = True) -> None:
        if clock_each_line:
            clock = datetime.now().strftime("%H:%M:%S")
            stamped = "\n".join(f"{clock} {line}" for line in block.split("\n"))
        else:
            stamped = block
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
    log.write_note(">>> старт обработки <<<")
    recognized: Queue[tuple[float, float, Future[tuple[str, float]]] | None] = Queue()
    summaries: Queue[tuple[float, str, Future[tuple[str, float, float]]] | None] = Queue()
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
                audio_end_s, _, future = item
                text, elapsed_s, average_s = future.result()
                summary_s += elapsed_s
                summary_count += 1
                log.write_summary(text, audio_end_s, elapsed_s, average_s)
        except Exception as error:
            logged_error.append(error)

    def log_when_ready() -> None:
        nonlocal decode_s
        full_text: list[str] = []
        transcript_lock = Lock()
        schedule_lock = Lock()
        call_done = Event()
        call_end_s = 0.0
        last_summary: Future[tuple[str, float, float]] | None = None

        def current_transcript() -> str:
            with transcript_lock:
                return "\n".join(full_text)

        def schedule_summaries() -> None:
            nonlocal last_summary
            try:
                if call_done.wait(SUMMARY_EVERY_S):
                    return
                while not call_done.is_set():
                    with transcript_lock:
                        has_text = bool(full_text)
                        audio_end_s = call_end_s
                    if not has_text:
                        if call_done.wait(0.05):
                            return
                        continue
                    with schedule_lock:
                        if call_done.is_set():
                            return
                        log.write_note(">>> запрос резюме <<<")
                        last_summary = summary.submit(
                            current_transcript,
                            file=audio_path.stem,
                            label=f"{audio_end_s:.0f}s",
                        )
                        summaries.put((audio_end_s, "резюме", last_summary))
                        future = last_summary
                    future.result()
                    if call_done.wait(SUMMARY_EVERY_S):
                        return
            except Exception as error:
                logged_error.append(error)
                call_done.set()

        scheduler = Thread(target=schedule_summaries, name=f"summary-schedule-{worker_id}")
        scheduler.start()
        try:
            while True:
                item = recognized.get()
                if item is None:
                    break
                start_s, duration_s, future = item
                text, rec_s = future.result()
                decode_s += rec_s
                audio_end_s = start_s + duration_s
                with transcript_lock:
                    call_end_s = audio_end_s
                    if text:
                        full_text.append(text)
                if text:
                    log.write(text, audio_end_s)
            with schedule_lock:
                call_done.set()
            scheduler.join()
            if full_text:
                with transcript_lock:
                    call_audio_end_s = call_end_s
                summaries.put(
                    (
                        call_audio_end_s,
                        "резюме звонка",
                        summary.submit(current_transcript, file=audio_path.stem, label="звонок"),
                    )
                )
        except Exception as error:
            logged_error.append(error)
        finally:
            call_done.set()
            scheduler.join()
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
