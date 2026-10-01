from concurrent.futures import Future
from queue import Queue
from threading import Thread
from time import perf_counter

import gigaam
import numpy as np
import torch
from gigaam.utils import AudioDataset

MODEL_NAME = "v3_e2e_rnnt"
SAMPLE_RATE = 16_000
MAX_SEGMENT_S = 25


def load_model(model_name: str = MODEL_NAME):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = gigaam.load_model(model_name, device=device)
    if device == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("device: cpu", flush=True)
    return model


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


def transcribe_audio(model, audio: np.ndarray) -> tuple[str, float]:
    started = perf_counter()
    parts: list[str] = []
    for piece in split_for_model(audio):
        text = transcribe_batch(model, [piece])[0]
        if text:
            parts.append(text)
    if model._device.type == "cuda":
        torch.cuda.synchronize()
    return " ".join(parts).strip(), perf_counter() - started


class TranscriptionService:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._queue: Queue[tuple[np.ndarray, Future[tuple[str, float]]] | None] = Queue()
        self._worker = Thread(target=self._serve, name="transcription")
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._model = load_model(self._model_name)
        self._worker.start()
        self._started = True

    def submit(self, audio: np.ndarray) -> Future[tuple[str, float]]:
        if not self._started:
            raise RuntimeError("Transcription service is not started")
        future: Future[tuple[str, float]] = Future()
        self._queue.put((audio, future))
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
            audio, future = item
            try:
                future.set_result(transcribe_audio(self._model, audio))
            except Exception as error:
                future.set_exception(error)
