"""Draw the score-over-time chart from progress.csv into progress.png and progress-dark.png."""
import csv
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CLOSE = dt.datetime(2026, 11, 5, 8, 59)
START = dt.datetime(2026, 10, 30, 12, 0)
ORDER = [("attention", "Attention"), ("moe", "MoE"), ("decode", "Decode"), ("prefill", "Prefill"), ("bf16", "BF16")]
THEMES = {
    "progress.png": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    },
    "progress-dark.png": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "grid": "#3a3a37",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    },
}


def load():
    rows = {}
    with open(HERE / "progress.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            when = dt.datetime.strptime(row["time_local"], "%Y-%m-%d %H:%M")
            rows.setdefault(row["problem"], []).append((when, float(row["score"])))
    return rows


def draw(rows, filename, theme):
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=170, facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])
    for (key, label), color in zip(ORDER, theme["series"]):
        points = rows[key]
        xs = [p[0] for p in points] + [CLOSE]
        ys = [p[1] for p in points] + [points[-1][1]]
        ax.step(xs, ys, where="post", color=color, linewidth=2, solid_capstyle="round", label=label)
        ax.plot([CLOSE], [ys[-1]], "o", color=color, markersize=7, markeredgecolor=theme["surface"], markeredgewidth=2, zorder=5)
        ax.annotate(
            f"{label} {ys[-1]:.3f}",
            (CLOSE, ys[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=10.5,
            color=theme["primary"],
        )

    note = {"fontsize": 9.5, "color": theme["secondary"], "ha": "left", "va": "top"}
    ax.annotate(
        "Oct 30 to 31: public engines adopted",
        (dt.datetime(2026, 10, 30, 23, 20), 2.60),
        xytext=(dt.datetime(2026, 10, 31, 6, 0), 2.86),
        arrowprops={"arrowstyle": "-", "color": theme["secondary"], "linewidth": 0.8},
        **note,
    )
    ax.annotate(
        "Nov 5: MoE with our own engines\nfor small sizes reaches the board",
        (dt.datetime(2026, 11, 5, 0, 20), 2.443),
        xytext=(dt.datetime(2026, 11, 3, 3, 0), 2.86),
        arrowprops={"arrowstyle": "-", "color": theme["secondary"], "linewidth": 0.8},
        **note,
    )

    ax.set_xlim(START, CLOSE + dt.timedelta(hours=3))
    ax.set_ylim(1.0, 2.95)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("Speedup over baseline", color=theme["secondary"], fontsize=10.5)
    ax.tick_params(colors=theme["secondary"], length=0, labelsize=10)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])

    ax.set_title("Score over time, kernel track", loc="left", color=theme["primary"], fontsize=15, fontweight="semibold", pad=34)
    ax.text(
        0,
        1.03,
        "Best judge check until the first submission on Oct 31, then the board value. Board values are the best of several noisy runs.",
        transform=ax.transAxes,
        color=theme["secondary"],
        fontsize=9.5,
    )
    legend = ax.legend(loc="lower right", ncols=5, frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.0), handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(theme["primary"])
    fig.subplots_adjust(left=0.065, right=0.875, top=0.85, bottom=0.08)
    fig.savefig(HERE / filename, facecolor=theme["surface"])
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    for name, colors in THEMES.items():
        draw(data, name, colors)
