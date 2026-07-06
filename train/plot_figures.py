"""Regenerate docs/assets/learning-curves.png from real run data.

Left panel: training-time kills per episode (rolling mean) for the six
iterations that built the champion — sampled (stochastic) policy, so these
curves show learning dynamics, not final strength.

Right panel: the gold standard — deterministic (argmax) evaluation on the
frozen 32-seed set, parsed straight from leaderboard.md so the figure can
never drift from the numbers we actually publish.

Usage (repo root):  .venv/bin/python train/plot_figures.py
"""

from __future__ import annotations

import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE.parent / "docs" / "assets" / "learning-curves.png"

# (run directory, legend label) — the v1→v6 lesson chain, one fix per run
CURVES = [
    ("ppo-l1-first", "v1 baseline"),
    ("ppo-l1-v2", "v2 shaping"),
    ("ppo-l1-v3", "v3 engage-macro"),
    ("ppo-l1-v4", "v4 per-hit reward"),
    ("ppo-l1-v5-vision", "v5 +vision"),
    ("ppo-l1-v6-explore", "v6 +explore-macro"),
]

# (leaderboard row name, bar label) — chronological order
BARS = [
    ("ppo-l1-v5-vision", "v5 no explore\nmacro (post-hoc)"),
    ("ppo-l1-v6-explore", "v6 macro-MLP\n46k params"),
    ("ppo-l1-v8-lstm", "v8 LSTM-128\n452k params"),
    ("ppo-l1-v9c-attn", "v9c entity-attn\n702k params"),
    ("ppo-l1-v10-longep", "v10 = v6 recipe,\n3000-step training"),
    ("ppo-l1-v11-descend", "v11 +descend option\n(doors/barrels/stairs)"),
    ("ppo-l1-v12-drink", "v12 +belt potion\n(deaths 17→10, kills regress)"),
]
CHAMPION = "ppo-l1-v11-descend"


def rolling(xs: list[float], w: int) -> list[float]:
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return out


def load_kills(run: str) -> list[float]:
    path = RUNS / run / "progress.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — this script regenerates the README figure from "
            "LOCAL training logs (train/runs/ is not distributed with the repo); "
            "the shipped figure lives at docs/assets/learning-curves.png"
        )
    kills = []
    with open(path) as f:
        for line in f:
            try:
                kills.append(float(json.loads(line).get("kills", 0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return kills


def load_leaderboard() -> dict[str, dict[str, str]]:
    rows = {}
    text = (HERE / "leaderboard.md").read_text()
    for m in re.finditer(
        r"^\|\s*(ppo-[^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|"
        r"\s*(\d+/\d+)\s*\|",
        text,
        re.M,
    ):
        name, mean, median, mx, zero = m.groups()
        name = re.sub(r"[^a-z0-9-]", "", name)  # strip footnote marks (e.g. ¹)
        rows[name] = {"mean": float(mean), "median": median, "max": mx, "zero": zero}
    return rows


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 4.6), width_ratios=[1.6, 1.4], dpi=150
    )

    for run, label in CURVES:
        kills = load_kills(run)
        smooth = rolling(kills, 100)
        xs = [i / max(len(smooth) - 1, 1) for i in range(len(smooth))]
        ax1.plot(xs, smooth, label=label, linewidth=1.2)
    ax1.set_title("Training curves across six iterations (sampled policy)")
    ax1.set_xlabel("training progress")
    ax1.set_ylabel("kills / episode (rolling 100)")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.25)

    board = load_leaderboard()
    labels = [label for _, label in BARS]
    means = [board[name]["mean"] for name, _ in BARS]
    zeros = [board[name]["zero"] for name, _ in BARS]
    colors = ["#d4880f" if name == CHAMPION else "#c0c0c0" for name, _ in BARS]
    bars = ax2.bar(labels, means, color=colors)
    for bar, mean, zero in zip(bars, means, zeros):
        ax2.annotate(
            f"{mean}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax2.annotate(
            f"zero-kill {zero}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
            ha="center",
            va="center",
            fontsize=7,
            color="#333333",
        )
    ax2.set_title("Deterministic eval, 32 fixed seeds\n(argmax, 1500 steps — gold standard)")
    ax2.set_ylabel("mean kills / episode")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
