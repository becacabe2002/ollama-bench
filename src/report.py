"""Aggregates graded CSVs into a ranked markdown summary and a sweet-spot scatter."""

from __future__ import annotations

import argparse
import glob
import html
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RANKING_TOLERANCE = 0.2  # coverage moves in steps of 0.2 with n=5
WORD_LIMIT = 200
STAT_METRICS = [
    ("summary_f1", "F1", 3),
    ("alignment", "Alignment", 3),
    ("coverage", "Coverage", 2),
    ("word_count", "Words", 0),
    ("wall_s", "Wall (s)", 1),
    ("load_s", "Load (s)", 2),
    ("prompt_tps", "Prompt t/s", 1),
    ("gen_tps", "Gen t/s", 1),
    ("peak_rss_gb", "Peak RSS (GiB)", 2),
    ("vmhwm_gb", "VmHWM (GiB)", 2),
]

SVG_STYLE = """
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --grid:#e4e3de; --small:#2a78d6; --mid:#eb6834; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --grid:#33322f; --small:#3987e5; --mid:#d95926; }
  }
  text { font: 12px system-ui, sans-serif; fill: var(--ink-2); }
  .title { font-size: 15px; font-weight: 600; fill: var(--ink); }
  .label { fill: var(--ink); }
"""
TIER_COLOR = {"small": "var(--small)", "mid": "var(--mid)"}


def load_graded(patterns: list[str]) -> pd.DataFrame:
    paths = [p for pattern in patterns for p in (glob.glob(pattern) or [pattern])]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for column, _, _ in STAT_METRICS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["swap_touched"] = df["swap_touched"].astype(str).str.lower() == "true"
    return df


def summarize(df: pd.DataFrame, tiers: dict[str, str]) -> pd.DataFrame:
    """One row per model, ranked by median F1 with generation speed breaking ties."""
    ok = df[df["status"] == "ok"]
    columns = [c for c, _, _ in STAT_METRICS]
    stats = ok.groupby("model")[columns].agg(["median", "min", "max"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    summary = stats.reindex(sorted(df["model"].unique())).reset_index()
    summary["tier"] = summary["model"].map(tiers).fillna("?")
    summary["trials_ok"] = summary["model"].map(ok.groupby("model").size()).fillna(0).astype(int)
    summary["trials_total"] = summary["model"].map(df.groupby("model").size())
    return summary.sort_values(
        ["summary_f1_median", "gen_tps_median"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def fmt(value: float, decimals: int) -> str:
    return "–" if pd.isna(value) else f"{value:.{decimals}f}"


def spread(row: pd.Series, metric: str, decimals: int) -> str:
    if pd.isna(row[f"{metric}_median"]):
        return "–"
    low, high = fmt(row[f"{metric}_min"], decimals), fmt(row[f"{metric}_max"], decimals)
    return f"{fmt(row[f'{metric}_median'], decimals)} ({low}–{high})"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def flags(df: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    notes = []
    failed = df[df["status"] != "ok"]
    for row in failed.to_dict("records"):
        notes.append(f"**Failed**: `{row['model']}` trial {row['trial']}: {row['error']}")
    ungraded = df[(df["status"] == "ok") & df["summary_f1"].isna()]
    for row in ungraded.to_dict("records"):
        notes.append(f"**Ungraded**: `{row['model']}` trial {row['trial']}")
    for model in sorted(df[df["swap_touched"]]["model"].unique()):
        notes.append(f"**Swap touched**: `{model}`; its latency and memory figures are suspect")
    over = df[(df["status"] == "ok") & (df["word_count"] >= WORD_LIMIT)]
    for model, group in over.groupby("model"):
        notes.append(f"**Over {WORD_LIMIT} words**: `{model}` in {len(group)} trial(s), max {int(group['word_count'].max())}")
    scored = summary[summary["summary_f1_median"].notna()]
    if not scored.empty:
        leader = scored.iloc[0]
        tied = scored[leader["summary_f1_median"] - scored["summary_f1_median"] < RANKING_TOLERANCE]
        if len(tied) > 1:
            names = ", ".join(f"`{m}`" for m in tied["model"])
            notes.append(
                f"**Quality tie**: {names} sit within {RANKING_TOLERANCE} F1 of the leader. "
                "Coverage moves in 0.2 steps, so decide between them on latency and memory."
            )
    return notes


def nice_axis_max(value: float) -> tuple[float, float]:
    """(axis max, tick step) with a 1/2/5 step and about five ticks."""
    if value <= 0:
        return 1.0, 0.2
    raw = value / 5
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    return math.ceil(value / step) * step, step


def place_labels(points: list[dict], bounds: tuple[float, float, float, float]) -> None:
    """Greedy placement: each label takes the first side clear of dots and earlier labels."""
    left, top, right, bottom = bounds
    char_w, height, gap = 6.6, 14, 9
    placed: list[tuple[float, float, float, float]] = []
    dots = [(p["x"] - 6, p["y"] - 6, p["x"] + 6, p["y"] + 6) for p in points]

    def clear(box: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = box
        if x0 < left or x1 > right or y0 < top or y1 > bottom:
            return False
        return not any(x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1] for b in placed + dots)

    for point in sorted(points, key=lambda p: (p["x"], p["y"])):
        width = len(point["name"]) * char_w
        x, y = point["x"], point["y"]
        options = [
            ("start", x + gap, y + 4),  # right
            ("end", x - gap, y + 4),  # left
            ("middle", x, y - gap - 2),  # above
            ("middle", x, y + gap + 12),  # below
            ("start", x + gap, y - gap),  # upper right
            ("start", x + gap, y + gap + 8),  # lower right
            ("end", x - gap, y - gap),  # upper left
            ("end", x - gap, y + gap + 8),  # lower left
        ]
        for anchor, tx, ty in options:
            x0 = {"start": tx, "end": tx - width, "middle": tx - width / 2}[anchor]
            box = (x0, ty - height + 3, x0 + width, ty + 3)
            if clear(box):
                break
        else:
            anchor, tx, ty = options[1] if x + gap + width > right else options[0]
            x0 = {"start": tx, "end": tx - width}[anchor]
            box = (x0, ty - height + 3, x0 + width, ty + 3)
        placed.append(box)
        point["label"] = (anchor, tx, ty)


def scatter_svg(summary: pd.DataFrame) -> str | None:
    data = summary.dropna(subset=["gen_tps_median", "summary_f1_median"])
    if data.empty:
        return None
    width, height = 720, 470
    left, right, top, bottom = 58, 24, 70, 54
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max, x_step = nice_axis_max(float(data["gen_tps_median"].max()) * 1.05)

    def sx(value: float) -> float:
        return left + value / x_max * plot_w

    def sy(value: float) -> float:
        return top + (1 - value) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" aria-label="Scatter plot of median generation speed against median summary F1 per model">',
        f"<style>{SVG_STYLE}</style>",
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
        f'<text class="title" x="{left}" y="26">Generation speed vs summary quality</text>',
    ]

    # Legend
    legend_x = left
    for tier in ("small", "mid"):
        if tier in set(data["tier"]):
            out.append(f'<circle cx="{legend_x + 5}" cy="46" r="5" fill="{TIER_COLOR[tier]}"/>')
            out.append(f'<text x="{legend_x + 15}" y="50">{tier} tier</text>')
            legend_x += 86

    # Grid and axes
    for i in range(6):
        value = i * 0.2
        y = sy(value)
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')
    tick = 0.0
    while tick <= x_max + 1e-9:
        x = sx(tick)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{height - bottom}" y2="{height - bottom + 5}" stroke="var(--ink-2)"/>')
        out.append(f'<text x="{x:.1f}" y="{height - bottom + 20}" text-anchor="middle">{tick:g}</text>')
        tick += x_step
    out.append(f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle">Median generation speed (tokens/s)</text>')
    out.append(
        f'<text transform="translate(16 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Median summary F1</text>'
    )

    # Marks
    points = [
        {
            "name": r["model"],
            "tier": r["tier"],
            "x": sx(r["gen_tps_median"]),
            "y": sy(r["summary_f1_median"]),
            "tooltip": f"{r['model']}: {r['gen_tps_median']:.1f} t/s, F1 {r['summary_f1_median']:.3f}",
        }
        for r in data.to_dict("records")
    ]
    place_labels(points, (left, top - 14, width - right, height - bottom))
    for point in points:
        color = TIER_COLOR.get(point["tier"], "var(--ink-2)")
        out.append(
            f'<circle cx="{point["x"]:.1f}" cy="{point["y"]:.1f}" r="5" fill="{color}" '
            f'stroke="var(--surface)" stroke-width="2"><title>{html.escape(point["tooltip"])}</title></circle>'
        )
    for point in points:
        anchor, tx, ty = point["label"]
        out.append(
            f'<text class="label" x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}">{html.escape(point["name"])}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def render_markdown(df: pd.DataFrame, summary: pd.DataFrame, svg_name: str | None) -> str:
    ranked = [
        [
            i + 1,
            f"`{r['model']}`",
            r["tier"],
            spread(r, "summary_f1", 3),
            fmt(r["alignment_median"], 3),
            fmt(r["coverage_median"], 2),
            fmt(r["word_count_median"], 0),
            fmt(r["gen_tps_median"], 1),
            fmt(r["wall_s_median"], 1),
            fmt(r["peak_rss_gb_median"], 2),
            f"{r['trials_ok']}/{r['trials_total']}",
        ]
        for i, (_, r) in enumerate(summary.iterrows())
    ]
    sections = [
        "# W1 Ollama benchmark summary",
        f"Runs: {', '.join(str(r) for r in sorted(df['run_id'].unique()))}. "
        f"Judge: {', '.join(sorted(str(j) for j in df['judge_model'].dropna().unique())) or 'none'}.",
        "## Ranking by median summary F1",
        "F1 shows median (min–max) across trials. Other columns show medians.",
        table(
            ["#", "Model", "Tier", "F1", "Align.", "Cover.", "Words", "Gen t/s", "Wall (s)", "RSS (GiB)", "Trials OK"],
            ranked,
        ),
    ]
    notes = flags(df, summary)
    if notes:
        sections += ["## Flags", "\n".join(f"- {note}" for note in notes)]
    if svg_name:
        sections += [
            "## Sweet spot",
            f"![Scatter of generation speed against summary F1. Values are in the ranking table.]({svg_name})",
        ]
    sections += [
        "## Per-model statistics",
        "Each cell shows median (min–max) across successful trials.",
        table(
            ["Model", *[label for _, label, _ in STAT_METRICS]],
            [
                [f"`{r['model']}`", *[spread(r, c, d) for c, _, d in STAT_METRICS]]
                for _, r in summary.iterrows()
            ],
        ),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graded", nargs="+", required=True, help="CSV files; globs allowed")
    parser.add_argument("--models", type=Path, default=ROOT / "config/models.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "results/reports/w1_summary.md")
    args = parser.parse_args()

    tiers = {m["name"]: m.get("tier", "?") for m in yaml.safe_load(args.models.read_text(encoding="utf-8"))["models"]}
    df = load_graded(args.graded)
    summary = summarize(df, tiers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    svg = scatter_svg(summary)
    svg_path = args.out.with_suffix(".svg")
    if svg:
        svg_path.write_text(svg, encoding="utf-8")
    args.out.write_text(render_markdown(df, summary, svg_path.name if svg else None), encoding="utf-8")
    print(f"wrote {args.out}" + (f" and {svg_path}" if svg else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
