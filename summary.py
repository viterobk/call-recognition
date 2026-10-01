import json
from concurrent.futures import Future
from queue import Queue
from threading import Thread
from time import perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen

MODEL_NAME = "gemma4:e2b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

PROMPT = """Сделай короткое резюме телефонного звонка на русском языке обычным текстом.
Укажи суть проблемы и предложенные решения.
Не добавляй фактов, которых нет в транскрипции.

Транскрипция:
{transcript}
"""


def summarize_call(transcript: str, model_name: str = MODEL_NAME) -> str:
    if not transcript.strip():
        return ""
    payload = {
        "model": model_name,
        "prompt": PROMPT.replace("{transcript}", transcript.strip()),
        "stream": False,
        "keep_alive": "10m",
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


class SummaryService:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._queue: Queue[tuple[str, Future[tuple[str, float]]] | None] = Queue()
        self._worker = Thread(target=self._serve, name="summary")
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._worker.start()
        self._started = True

    def submit(self, transcript: str) -> Future[tuple[str, float]]:
        if not self._started:
            raise RuntimeError("Summary service is not started")
        future: Future[tuple[str, float]] = Future()
        self._queue.put((transcript, future))
        return future

    def close(self) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._worker.join()
        self._started = False

    def _serve(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            transcript, future = item
            started = perf_counter()
            try:
                text = summarize_call(transcript, self._model_name)
                future.set_result((text, perf_counter() - started))
            except Exception as error:
                future.set_exception(error)
