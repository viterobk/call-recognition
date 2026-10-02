#!/usr/bin/env python3
"""Диаграмма Гантта по results/gantt.csv. Зависит только от стандартной библиотеки."""

import csv
import html
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "results" / "gantt.csv"

PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)


def main() -> None:
    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.is_file():
        raise SystemExit("Нет файла: {0}".format(csv_path))
    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit("В {0} нет строк с резюме".format(csv_path))
    page = csv_path.with_name("gantt.html")
    page.write_text(render(rows), encoding="utf-8")
    print(page)
    webbrowser.open(page.as_uri())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["order"]))
    return rows


def render(rows: list[dict[str, str]]) -> str:
    label_width = 280
    chart_width = 920
    row_height = 28
    margin_top = 36
    margin_bottom = 28
    files = unique_files(rows)
    height = margin_top + row_height * len(files) + margin_bottom
    width = label_width + chart_width + 24
    end = max(float(row["end_s"]) for row in rows)
    end = end if end > 0 else 1.0
    colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(files)}
    row_of = {name: index for index, name in enumerate(files)}

    parts = [
        "<!DOCTYPE html>",
        "<html lang='ru'><head><meta charset='utf-8'>",
        "<title>Генерация резюме</title>",
        "<style>body{margin:24px;font:14px sans-serif;background:#fff;color:#222}",
        "p{color:#555}",
        "#tip{position:fixed;display:none;padding:6px 8px;background:#222;color:#fff;",
        "font-size:13px;border-radius:4px;pointer-events:none;white-space:pre-line;z-index:2}</style></head><body>",
        "<div id='tip'></div>",
        "<h1>Генерация резюме</h1>",
        "<p>Полоса — чистое время генерации. Просвет перед ней — ожидание в очереди.</p>",
        "<svg xmlns='http://www.w3.org/2000/svg' width='{0}' height='{1}'>".format(width, height),
    ]
    for tick in ticks(end):
        x = label_width + chart_width * tick / end
        parts.append(
            "<line x1='{0:.1f}' y1='{1}' x2='{0:.1f}' y2='{2}' stroke='#eee'/>".format(
                x, margin_top - 8, height - margin_bottom
            )
        )
        parts.append(
            "<text x='{0:.1f}' y='{1}' text-anchor='middle' font-size='12' fill='#666'>{2:g}s</text>".format(
                x, height - 8, tick
            )
        )
    for name in files:
        y = margin_top + row_of[name] * row_height
        parts.append(
            "<text x='{0}' y='{1}' text-anchor='end' font-size='12'>{2}</text>".format(
                label_width - 8, y + 18, html.escape(name)
            )
        )
    for row in rows:
        y = margin_top + row_of[row["file"]] * row_height
        start = float(row["start_s"])
        duration = float(row["duration_s"])
        x = label_width + chart_width * start / end
        bar = max(2.0, chart_width * duration / end)
        tip = "{0} {1}\nгенерация {2:.2f}s".format(row["file"], row["label"], duration)
        parts.append(
            "<rect x='{0:.1f}' y='{1}' width='{2:.1f}' height='{3}' fill='{4}' rx='3' data-tip='{5}'></rect>".format(
                x, y + 4, bar, row_height - 8, colors[row["file"]], html.escape(tip, quote=True)
            )
        )
    parts.append("</svg>")
    parts.append(
        "<script>var tip=document.getElementById('tip');"
        "document.querySelectorAll('rect[data-tip]').forEach(function(bar){"
        "bar.addEventListener('mousemove',function(event){"
        "tip.textContent=bar.getAttribute('data-tip');tip.style.display='block';"
        "tip.style.left=(event.clientX+12)+'px';tip.style.top=(event.clientY+12)+'px';});"
        "bar.addEventListener('mouseleave',function(){tip.style.display='none';});"
        "});</script></body></html>"
    )
    return "\n".join(parts)


def unique_files(rows: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["file"] not in seen:
            seen.append(row["file"])
    return seen


def ticks(end: float) -> list[float]:
    step = 1.0
    for candidate in (1, 2, 5, 10, 15, 30, 60, 120):
        if end / candidate <= 12:
            step = float(candidate)
            break
    values = [0.0]
    mark = step
    while mark < end:
        values.append(mark)
        mark += step
    values.append(end)
    return values


if __name__ == "__main__":
    main()
