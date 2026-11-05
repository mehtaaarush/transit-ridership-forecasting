"""Static figures for the README (matplotlib, same palette as the dashboard's light theme)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402

from . import analysis, config  # noqa: E402
from .evaluate import summarize  # noqa: E402
from .export import MODEL_NAMES  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, DEEMPH = "#e1e0d9", "#c3c2b7", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIV_NEG, DIV_POS = "#e34948", "#2a78d6"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
            "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlesize": 12.5,
            "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 12,
            "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
            "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
            "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
            "legend.frameon": False, "legend.labelcolor": INK2, "lines.solid_capstyle": "round",
        }
    )


def _compact(x, _pos=None) -> str:
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M".replace(".0M", "M")
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def _save(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def history_figure(daily: pd.DataFrame) -> Path:
    hist = analysis.weekly_history(daily)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist.index, hist["bus"], color=BLUE, lw=1.6, label="Bus")
    ax.plot(hist.index, hist["rail"], color=ORANGE, lw=1.6, label="Rail")
    covid = pd.Timestamp("2020-03-21")
    ax.axvline(covid, color=BASELINE, lw=1)
    ax.text(covid, ax.get_ylim()[1] * 0.97, "  Mar 2020 stay-at-home order", color=INK2, fontsize=9, va="top")
    for col, label in (("bus", "Bus"), ("rail", "Rail")):
        ax.annotate(label, (hist.index[-1], hist[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=9.5, va="center")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_compact))
    ax.set_ylim(bottom=0)
    ax.set_title("CTA boardings per day, weekly average (2001–2026)")
    ax.legend(loc="lower left", ncol=2)
    return _save(fig, "history.png")


def backtest_figure(preds: pd.DataFrame, days: int = 180, horizon: int = 7) -> Path:
    sub = preds[(preds["model"] == "gbm") & (preds["h"] == horizon)].set_index("target_date").sort_index()
    sub = sub[sub.index > sub.index.max() - pd.Timedelta(days=days)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(sub.index, sub["lo"], sub["hi"], color=BLUE, alpha=0.13, lw=0, label="80% prediction interval")
    ax.plot(sub.index, sub["y"], color=INK, lw=1.3, label="Actual")
    ax.plot(sub.index, sub["pred"], color=BLUE, lw=2, label=f"Forecast made {horizon} days ahead")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_compact))
    ax.set_ylim(bottom=0)
    ax.set_title(f"Out-of-sample forecasts vs. actual boardings, last {days} days of the backtest")
    ax.legend(loc="lower left", ncol=3)
    return _save(fig, "backtest.png")


def model_comparison_figure(preds: pd.DataFrame) -> Path:
    overall = summarize(preds).set_index("model")["wape"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = [BLUE if m == "gbm" else DEEMPH for m in overall.index]
    labels = [MODEL_NAMES[m] for m in overall.index]
    ax.barh(labels, overall.values * 100, color=colors, height=0.5)
    for i, v in enumerate(overall.values):
        ax.text(v * 100 + 0.15, i, f"{v:.1%}", va="center", color=INK, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="y", colors=INK2)
    ax.set_title("Backtest error (WAPE, lower is better), 1–14 days ahead")
    return _save(fig, "model_comparison.png")


def effects_figure(daily: pd.DataFrame) -> Path:
    rows = pd.DataFrame(analysis.holiday_effects(daily)).sort_values("effect")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [DIV_NEG if e < 0 else DIV_POS for e in rows["effect"]]
    ax.barh(rows["name"], rows["effect"] * 100, color=colors, height=0.6)
    ax.axvline(0, color=BASELINE, lw=1)
    for i, e in enumerate(rows["effect"]):
        ax.text(e * 100 + (1.2 if e >= 0 else -1.2), i, f"{e:+.0%}".replace("-", "−"), va="center",
                ha="left" if e >= 0 else "right", color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(-85, 60)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%".replace("-", "−")))
    ax.tick_params(axis="y", colors=INK2)
    ax.set_title("Ridership vs. a typical same-weekday day (2022–2026 averages)")
    return _save(fig, "holiday_effects.png")


def make_all(daily: pd.DataFrame, preds: pd.DataFrame) -> list[Path]:
    _style()
    return [history_figure(daily), backtest_figure(preds), model_comparison_figure(preds), effects_figure(daily)]
