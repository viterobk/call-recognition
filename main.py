from pathlib import Path
from sys import stdout
from time import perf_counter

from audio_processing import process_audio_file
from summary import summarize_call
from transcription import TranscriptionService

PROJECT_DIR = Path(__file__).resolve().parent
AUDIO_FILE = PROJECT_DIR / "speech.mp3"
RESULTS_DIR = PROJECT_DIR / "results"
TRANSCRIPT_FILE = RESULTS_DIR / "transcript.txt"
SUMMARY_FILE = RESULTS_DIR / "summary.txt"


def main() -> None:
    stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    service = TranscriptionService()
    service.start()
    try:
        result = process_audio_file(AUDIO_FILE, service, TRANSCRIPT_FILE, worker_id=0)
    finally:
        service.close()

    print(
        f"wall {result.wall_s:.2f}s | audio {result.audio_s:.1f}s | "
        f"decode {result.decode_s:.2f}s | replicas {result.replicas}",
        flush=True,
    )

    started = perf_counter()
    summary = summarize_call(result.transcript)
    elapsed = perf_counter() - started
    SUMMARY_FILE.write_text(summary + ("\n" if summary else ""), encoding="utf-8")
    print(f"summary {elapsed:.2f}s\n{summary}", flush=True)


if __name__ == "__main__":
    main()
