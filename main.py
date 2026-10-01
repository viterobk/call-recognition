from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sys import stdout
from time import perf_counter

from audio_processing import FileJobResult, process_audio_file
from transcription import TranscriptionService

PROJECT_DIR = Path(__file__).resolve().parent
AUDIO_FILE = PROJECT_DIR / "speech.mp3"
RESULTS_DIR = PROJECT_DIR / "results"
WORKER_COUNT = 10


def run_load(worker_count: int = WORKER_COUNT) -> list[FileJobResult]:
    RESULTS_DIR.mkdir(exist_ok=True)
    service = TranscriptionService()
    service.start()
    started = perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    process_audio_file,
                    AUDIO_FILE,
                    service,
                    RESULTS_DIR / f"worker-{worker_id}.txt",
                    worker_id,
                )
                for worker_id in range(worker_count)
            ]
            results = [future.result() for future in as_completed(futures)]
    finally:
        service.close()
    wall_s = perf_counter() - started
    results.sort(key=lambda item: item.worker_id)
    report_load(results, wall_s)
    return results


def report_load(results: list[FileJobResult], wall_s: float) -> None:
    decode_s = sum(item.decode_s for item in results)
    lines = [
        f"workers: {len(results)}",
        f"wall: {wall_s:.2f}s",
        f"decode_total: {decode_s:.2f}s",
    ]
    for item in results:
        lines.append(
            f"w{item.worker_id}: wall {item.wall_s:.2f}s | audio {item.audio_s:.1f}s | "
            f"skipped {item.skipped_s:.1f}s | decode {item.decode_s:.2f}s | replicas {item.replicas}"
        )
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "summary.txt").write_text(report + "\n", encoding="utf-8")


def main() -> None:
    stdout.reconfigure(line_buffering=True)
    run_load()


if __name__ == "__main__":
    main()
