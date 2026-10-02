import json
import os
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from time import perf_counter, sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

MODEL_NAME = "gemma4:e2b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
RECENT_SUMMARIES = 10
SUMMARY_WORKERS = 4
NUM_CTX = 8192
OLLAMA_DROPIN = Path("/etc/systemd/system/ollama.service.d/parallel.conf")

PROMPT = """Сделай короткое резюме телефонного звонка на русском языке обычным текстом.
Укажи суть проблемы и предложенные решения.
Не добавляй фактов, которых нет в транскрипции.

Транскрипция:
{transcript}
"""


def _run(argv: list[str], *, input_text: str | None = None, timeout: float = 30) -> subprocess.CompletedProcess[str] | None:
    try:
        completed = subprocess.run(
            argv,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode == 0 or os.geteuid() == 0 or argv[0] == "sudo":
        return completed
    try:
        return subprocess.run(
            ["sudo", "-n", *argv],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return completed


def _command_text(argv: list[str]) -> str | None:
    completed = _run(argv)
    if completed is None or completed.returncode != 0:
        return None
    return completed.stdout


def _ollama_settings() -> tuple[int | None, int | None]:
    text = _command_text(["systemctl", "show", "ollama", "-p", "Environment", "--no-pager"])
    pid_text = _command_text(["systemctl", "show", "ollama", "-p", "MainPID", "--no-pager"])
    if pid_text:
        match = re.search(r"MainPID=(\d+)", pid_text)
        pid = int(match.group(1)) if match else 0
        if pid > 0:
            try:
                raw = Path(f"/proc/{pid}/environ").read_bytes()
            except OSError:
                raw = b""
            if raw:
                text = raw.replace(b"\0", b"\n").decode(errors="replace")
    if not text:
        return None, None
    parallel_match = re.search(r"OLLAMA_NUM_PARALLEL=(\d+)", text)
    context_match = re.search(r"OLLAMA_CONTEXT_LENGTH=(\d+)", text)
    parallel = int(parallel_match.group(1)) if parallel_match else None
    context = int(context_match.group(1)) if context_match else None
    return parallel, context


def _wait_ollama() -> bool:
    for _ in range(30):
        try:
            with urlopen(OLLAMA_TAGS_URL, timeout=2):
                return True
        except OSError:
            sleep(1)
    return False


def _align_ollama(slots: int) -> int:
    parallel, context = _ollama_settings()
    if parallel == slots and context == NUM_CTX:
        return slots
    content = (
        "[Service]\n"
        f'Environment="OLLAMA_NUM_PARALLEL={slots}"\n'
        f'Environment="OLLAMA_CONTEXT_LENGTH={NUM_CTX}"\n'
    )
    if os.geteuid() == 0:
        try:
            OLLAMA_DROPIN.parent.mkdir(parents=True, exist_ok=True)
            OLLAMA_DROPIN.write_text(content, encoding="utf-8")
        except OSError:
            return parallel or 1
    else:
        made = _run(["mkdir", "-p", str(OLLAMA_DROPIN.parent)])
        written = _run(["tee", str(OLLAMA_DROPIN)], input_text=content)
        if made is None or written is None or made.returncode != 0 or written.returncode != 0:
            return parallel or 1
    reloaded = _run(["systemctl", "daemon-reload"], timeout=60)
    restarted = _run(["systemctl", "restart", "ollama"], timeout=90)
    if reloaded is None or restarted is None or reloaded.returncode != 0 or restarted.returncode != 0:
        return parallel or 1
    if not _wait_ollama():
        raise RuntimeError("Ollama не поднялась после перезапуска")
    print(f"summary: Ollama перезапущена, слотов {slots}, контекст {NUM_CTX}", flush=True)
    return slots


def _worker_count() -> int:
    actual = _align_ollama(SUMMARY_WORKERS)
    print(f"summary: {actual} потоков", flush=True)
    if actual < SUMMARY_WORKERS:
        print(
            f"summary: задано {SUMMARY_WORKERS} потоков, Ollama принимает {actual}",
            flush=True,
        )
    return actual


def summarize_call(transcript: str, model_name: str = MODEL_NAME) -> str:
    if not transcript.strip():
        return ""
    payload = {
        "model": model_name,
        "prompt": PROMPT.replace("{transcript}", transcript.strip()),
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_ctx": NUM_CTX},
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode())
    except URLError as error:
        raise RuntimeError(
            f"Не удалось вызвать {model_name}. Запустите Ollama и выполните: ollama pull {model_name}"
        ) from error
    return str(body.get("response", "")).strip()


@dataclass(frozen=True)
class SummarySpan:
    order: int
    file: str
    label: str
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class SummaryService:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._queue: Queue[tuple[Callable[[], str], str, str, Future[tuple[str, float, float]]] | None] = Queue()
        self._workers: list[Thread] = []
        self._started = False
        self._origin = 0.0
        self._order = 0
        self._spans: list[SummarySpan] = []
        self._lock = Lock()

    def start(self) -> None:
        if self._started:
            return
        self._origin = perf_counter()
        worker_count = _worker_count()
        for index in range(worker_count):
            worker = Thread(target=self._serve, name=f"summary-{index}")
            worker.start()
            self._workers.append(worker)
        self._started = True

    def set_origin(self, origin: float) -> None:
        self._origin = origin

    def submit(self, get_transcript: Callable[[], str], *, file: str, label: str) -> Future[tuple[str, float, float]]:
        if not self._started:
            raise RuntimeError("Summary service is not started")
        future: Future[tuple[str, float, float]] = Future()
        self._queue.put((get_transcript, file, label, future))
        return future

    def spans(self) -> list[SummarySpan]:
        with self._lock:
            return list(self._spans)

    def close(self) -> None:
        if not self._started:
            return
        for _worker in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join()
        self._workers.clear()
        self._started = False

    def _serve(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            get_transcript, file, label, future = item
            try:
                transcript = get_transcript()
                started = perf_counter()
                text = summarize_call(transcript, self._model_name)
                duration_s = perf_counter() - started
                with self._lock:
                    self._order += 1
                    self._spans.append(
                        SummarySpan(
                            order=self._order,
                            file=file,
                            label=label,
                            start_s=started - self._origin,
                            duration_s=duration_s,
                        )
                    )
                    recent = self._spans[-RECENT_SUMMARIES:]
                    average_s = sum(span.duration_s for span in recent) / len(recent)
                future.set_result((text, duration_s, average_s))
            except Exception as error:
                future.set_exception(error)
