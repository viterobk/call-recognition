from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sys import stdout
from time import perf_counter

from audio_processing import FileJobResult, process_audio_file
from summary import SummaryService, SummarySpan
from tools.charts_page import write_page
from transcription import TranscriptionService

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_DIR / "sample_sounds"
RESULTS_DIR = PROJECT_DIR / "results"
GANTT_FILE = RESULTS_DIR / "gantt.csv"
WAIT_FILE = RESULTS_DIR / "wait.csv"


def clear_results(path: Path) -> None:
    path.mkdir(exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def write_gantt(spans: list[SummarySpan]) -> None:
    lines = ["order,file,label,start_s,duration_s,end_s"]
    for span in spans:
        lines.append(
            f'{span.order},"{span.file}",{span.label},{span.start_s:.2f},{span.duration_s:.2f},{span.end_s:.2f}'
        )
    GANTT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_wait(results: list[FileJobResult]) -> None:
    lines = ["file,label,call_s,wait_s"]
    points = [(item.name, wait) for item in results for wait in item.waits]
    points.sort(key=lambda point: (point[0], point[1].call_s))
    for name, wait in points:
        lines.append(f'"{name}",{wait.label},{wait.call_s:.2f},{wait.wait_s:.2f}')
    WAIT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_load(results: list[FileJobResult], wall_s: float) -> None:
    decode_s = sum(item.decode_s for item in results)
    summary_s = sum(item.summary_s for item in results)
    lines = [
        f"files: {len(results)}",
        f"wall: {wall_s:.2f}s",
        f"decode_total: {decode_s:.2f}s",
        f"summary_total: {summary_s:.2f}s",
    ]
    for item in results:
        lines.append(
            f"{item.name}: wall {item.wall_s:.2f}s | audio {item.audio_s:.1f}s | "
            f"decode {item.decode_s:.2f}s | summary {item.summary_s:.2f}s | "
            f"replicas {item.replicas} | summaries {item.summaries}"
        )
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "report.txt").write_text(report + "\n", encoding="utf-8")


def main() -> None:
    stdout.reconfigure(line_buffering=True)
    files = sorted(SAMPLE_DIR.glob("*.mp3"))
    if not files:
        raise FileNotFoundError(f"В {SAMPLE_DIR} нет mp3 файлов")
    clear_results(RESULTS_DIR)

    transcription = TranscriptionService()
    summary = SummaryService()
    try:
        transcription.start()
        summary.start()
        started = perf_counter()
        summary.set_origin(started)
        with ThreadPoolExecutor(max_workers=len(files)) as pool:
            futures = [
                pool.submit(
                    process_audio_file,
                    audio_path,
                    transcription,
                    summary,
                    RESULTS_DIR / f"{audio_path.stem}.txt",
                    worker_id,
                )
                for worker_id, audio_path in enumerate(files)
            ]
            results = [future.result() for future in futures]
        write_gantt(summary.spans())
        write_wait(results)
        report_load(results, perf_counter() - started)
        page = write_page(RESULTS_DIR)
        if page is not None:
            print(page, flush=True)
    finally:
        summary.close()
        transcription.close()


if __name__ == "__main__":
    main()
