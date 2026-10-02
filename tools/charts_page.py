#!/usr/bin/env python3
"""Графики по results/gantt.csv и results/wait.csv. Только стандартная библиотека."""

import csv
import html
import math
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results"

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
    results = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RESULTS
    if results.is_file():
        results = results.parent
    gantt_rows = read_csv(results / "gantt.csv")
    wait_rows = read_csv(results / "wait.csv")
    if gantt_rows:
        gantt_rows.sort(key=lambda row: int(row["order"]))
    if wait_rows:
        wait_rows.sort(key=lambda row: (row["file"], float(row["call_s"])))
    if not gantt_rows and not wait_rows:
        raise SystemExit("В {0} нет данных gantt.csv и wait.csv".format(results))
    page = results / "charts.html"
    page.write_text(render(gantt_rows, wait_rows), encoding="utf-8")
    print(page)
    webbrowser.open(page.as_uri())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render(gantt_rows: list[dict[str, str]], wait_rows: list[dict[str, str]]) -> str:
    files = unique_files(gantt_rows + wait_rows)
    colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(files)}
    parts = [
        "<!DOCTYPE html>",
        "<html lang='ru'><head><meta charset='utf-8'>",
        "<title>Резюме звонков</title>",
        "<style>",
        "body{margin:24px;font:14px sans-serif;background:#fff;color:#222}",
        "p{color:#555}",
        "h1{margin:28px 0 8px}",
        ".chart{display:flex;align-items:flex-start}",
        ".labels{flex:0 0 280px}",
        ".labels div{height:28px;line-height:28px;padding-right:8px;text-align:right;font-size:12px;",
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
        ".scroll{overflow-x:auto;flex:1;min-width:0}",
        ".y-axis{position:relative;flex:0 0 56px}",
        ".y-axis div{position:absolute;right:8px;font-size:12px;color:#666;transform:translateY(-50%)}",
        ".legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:8px}",
        ".legend span{display:inline-flex;align-items:center;gap:6px;font-size:12px}",
        ".swatch{width:14px;height:14px;border-radius:3px}",
        "#tip{position:fixed;display:none;padding:6px 8px;background:#222;color:#fff;",
        "font-size:13px;border-radius:4px;pointer-events:none;white-space:pre-line;z-index:2}",
        "</style></head><body>",
        "<div id='tip'></div>",
    ]
    if gantt_rows:
        parts.append(render_gantt(gantt_rows, colors))
    else:
        parts.append("<h1>Генерация резюме</h1><p>Нет файла gantt.csv.</p>")
    if wait_rows:
        parts.append(render_wait(wait_rows, colors))
    else:
        parts.append("<h1>Ожидание резюме</h1><p>Нет файла wait.csv.</p>")
    parts.append(
        "<script>var tip=document.getElementById('tip');"
        "document.querySelectorAll('[data-tip]').forEach(function(node){"
        "node.addEventListener('mousemove',function(event){"
        "tip.textContent=node.getAttribute('data-tip');tip.style.display='block';"
        "tip.style.left=(event.clientX+12)+'px';tip.style.top=(event.clientY+12)+'px';});"
        "node.addEventListener('mouseleave',function(){tip.style.display='none';});"
        "});</script></body></html>"
    )
    return "\n".join(parts)


def render_gantt(rows: list[dict[str, str]], colors: dict[str, str]) -> str:
    label_width = 280
    row_height = 28
    margin_top = 36
    margin_bottom = 28
    px_per_second = 10.0
    axis_pad = 20
    files = unique_files(rows)
    height = margin_top + row_height * len(files) + margin_bottom
    end = max(float(row["end_s"]) for row in rows)
    axis_end = grid_end(end, 5.0)
    chart_width = axis_end * px_per_second
    row_of = {name: index for index, name in enumerate(files)}
    parts = [
        "<h1>Генерация резюме</h1>",
        "<p>Полоса — чистое время генерации. Просвет перед ней — ожидание в очереди.</p>",
        "<div class='chart'><div class='labels' style='padding-top:{0}px'>".format(margin_top),
    ]
    for name in files:
        parts.append("<div title='{0}'>{0}</div>".format(html.escape(name)))
    parts.append("</div><div class='scroll'>")
    parts.append(
        "<svg xmlns='http://www.w3.org/2000/svg' width='{0:.0f}' height='{1}'>".format(
            chart_width + axis_pad * 2, height
        )
    )
    for tick in ticks(axis_end, 5.0):
        x = axis_pad + chart_width * tick / axis_end
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
    for row in rows:
        y = margin_top + row_of[row["file"]] * row_height
        start = float(row["start_s"])
        duration = float(row["duration_s"])
        x = axis_pad + chart_width * start / axis_end
        bar = max(2.0, chart_width * duration / axis_end)
        tip = "{0} {1}\nгенерация {2:.2f}s".format(row["file"], row["label"], duration)
        parts.append(
            "<rect x='{0:.1f}' y='{1}' width='{2:.1f}' height='{3}' fill='{4}' rx='3' data-tip='{5}'></rect>".format(
                x, y + 4, bar, row_height - 8, colors[row["file"]], html.escape(tip, quote=True)
            )
        )
    parts.append("</svg></div></div>")
    return "\n".join(parts)


def render_wait(rows: list[dict[str, str]], colors: dict[str, str]) -> str:
    margin_top = 16
    margin_bottom = 28
    plot_height = 280
    px_per_second = 10.0
    axis_pad = 20
    height = margin_top + plot_height + margin_bottom
    call_end = max(float(row["call_s"]) for row in rows)
    wait_end = max(float(row["wait_s"]) for row in rows)
    x_end = grid_end(call_end, 5.0)
    y_step = axis_step(wait_end)
    y_end = grid_end(wait_end, y_step)
    chart_width = x_end * px_per_second
    files = unique_files(rows)
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in files}
    for row in rows:
        grouped[row["file"]].append(row)

    def x_of(call_s: float) -> float:
        return axis_pad + chart_width * call_s / x_end

    def y_of(wait_s: float) -> float:
        return margin_top + plot_height * (1.0 - wait_s / y_end)

    parts = [
        "<h1>Ожидание резюме</h1>",
        "<p>По горизонтали — время звонка. По вертикали — секунды от запроса до готового резюме.</p>",
        "<div class='chart'><div class='y-axis' style='height:{0}px'>".format(height),
    ]
    for tick in ticks(y_end, y_step):
        parts.append("<div style='top:{0:.1f}px'>{1:g}s</div>".format(y_of(tick), tick))
    parts.append("</div><div class='scroll'>")
    parts.append(
        "<svg xmlns='http://www.w3.org/2000/svg' width='{0:.0f}' height='{1}'>".format(
            chart_width + axis_pad * 2, height
        )
    )
    for tick in ticks(y_end, y_step):
        y = y_of(tick)
        parts.append(
            "<line x1='{0}' y1='{1:.1f}' x2='{2:.1f}' y2='{1:.1f}' stroke='#eee'/>".format(
                axis_pad, y, axis_pad + chart_width
            )
        )
    for tick in ticks(x_end, 5.0):
        x = x_of(tick)
        parts.append(
            "<line x1='{0:.1f}' y1='{1}' x2='{0:.1f}' y2='{2}' stroke='#eee'/>".format(
                x, margin_top, margin_top + plot_height
            )
        )
        parts.append(
            "<text x='{0:.1f}' y='{1}' text-anchor='middle' font-size='12' fill='#666'>{2:g}s</text>".format(
                x, height - 8, tick
            )
        )
    for name in files:
        points = " ".join(
            "{0:.1f},{1:.1f}".format(x_of(float(row["call_s"])), y_of(float(row["wait_s"])))
            for row in grouped[name]
        )
        parts.append(
            "<polyline fill='none' stroke='{0}' stroke-width='2' points='{1}'/>".format(colors[name], points)
        )
    for row in rows:
        x = x_of(float(row["call_s"]))
        y = y_of(float(row["wait_s"]))
        tip = "{0} {1}\nзвонок {2:.1f}s\nожидание {3:.2f}s".format(
            row["file"], row["label"], float(row["call_s"]), float(row["wait_s"])
        )
        parts.append(
            "<circle cx='{0:.1f}' cy='{1:.1f}' r='4' fill='{2}' data-tip='{3}'></circle>".format(
                x, y, colors[row["file"]], html.escape(tip, quote=True)
            )
        )
    parts.append("</svg></div></div><div class='legend'>")
    for name in files:
        parts.append(
            "<span><i class='swatch' style='background:{0}'></i>{1}</span>".format(colors[name], html.escape(name))
        )
    parts.append("</div>")
    return "\n".join(parts)


def unique_files(rows: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["file"] not in seen:
            seen.append(row["file"])
    return seen


def axis_step(end: float) -> float:
    for candidate in (1, 2, 5, 10, 15, 30, 60, 120):
        if end / candidate <= 8:
            return float(candidate)
    return 120.0


def grid_end(end: float, step: float) -> float:
    if end <= 0:
        return step
    return step * math.ceil(end / step)


def ticks(end: float, step: float) -> list[float]:
    values = [0.0]
    mark = step
    while mark <= end + 1e-9:
        values.append(mark)
        mark += step
    return values


if __name__ == "__main__":
    main()
