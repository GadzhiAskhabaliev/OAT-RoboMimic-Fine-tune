#!/usr/bin/env python3
"""
Parse OAT tmux log and build a detailed training dashboard.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOSS_PATTERN = re.compile(r"Training epoch\s+(\d+):.*?loss=([0-9]+(?:\.[0-9]+)?)")


def parse_losses(log_text: str):
    # tqdm redraws in-place, so normalize carriage returns first.
    normalized = log_text.replace("\r", "\n")
    matches = LOSS_PATTERN.findall(normalized)
    if not matches:
        return [], []
    epochs = [int(e) for e, _ in matches]
    losses = [float(v) for _, v in matches]
    return epochs, losses


def ema(values: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def main():
    parser = argparse.ArgumentParser(description="Build hardcore training visualization from OAT log.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to training log file.")
    parser.add_argument("--output_dir", type=str, default="output/plots", help="Directory for plot artifacts.")
    parser.add_argument("--title", type=str, default="OAT RoboMimic Lift Training Dashboard", help="Plot title.")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(errors="ignore")
    epochs, losses = parse_losses(text)
    if not losses:
        raise RuntimeError(f"No training loss entries found in {log_path}")

    losses_np = np.asarray(losses, dtype=np.float64)
    steps = np.arange(len(losses_np))
    ema_loss = ema(losses_np, alpha=0.06)

    # Aggregate per-epoch metrics.
    per_epoch = defaultdict(list)
    for e, v in zip(epochs, losses):
        per_epoch[e].append(v)
    sorted_epochs = sorted(per_epoch.keys())
    epoch_mean = np.array([np.mean(per_epoch[e]) for e in sorted_epochs], dtype=np.float64)
    epoch_min = np.array([np.min(per_epoch[e]) for e in sorted_epochs], dtype=np.float64)
    epoch_max = np.array([np.max(per_epoch[e]) for e in sorted_epochs], dtype=np.float64)

    best_step = int(np.argmin(losses_np))
    best_loss = float(losses_np[best_step])
    latest_loss = float(losses_np[-1])
    drop_abs = float(losses_np[0] - latest_loss)
    drop_pct = float((drop_abs / losses_np[0]) * 100.0)

    # ----- plotting -----
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2], hspace=0.22, wspace=0.17)

    ax_main = fig.add_subplot(gs[0, :])
    ax_epoch = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # Main: step-level loss + EMA + best marker
    ax_main.plot(steps, losses_np, color="#4cc9f0", alpha=0.22, linewidth=1.0, label="step loss")
    ax_main.plot(steps, ema_loss, color="#f72585", linewidth=2.4, label="EMA loss (alpha=0.06)")
    ax_main.scatter([best_step], [best_loss], s=70, color="#b8f2e6", edgecolor="#ffffff", linewidth=0.8, zorder=5)
    ax_main.annotate(
        f"best={best_loss:.3f} @step {best_step}",
        xy=(best_step, best_loss),
        xytext=(best_step + max(20, len(steps) // 20), best_loss + 0.12),
        arrowprops=dict(arrowstyle="->", color="#b8f2e6", lw=1.2),
        color="#b8f2e6",
        fontsize=10,
    )
    ax_main.set_title(args.title, fontsize=17, pad=14, fontweight="bold")
    ax_main.set_xlabel("Training step (parsed from tqdm updates)", fontsize=11)
    ax_main.set_ylabel("Loss", fontsize=11)
    ax_main.grid(alpha=0.18, linestyle="--")
    ax_main.legend(loc="upper right", framealpha=0.18)

    info = (
        f"first={losses_np[0]:.3f}\n"
        f"latest={latest_loss:.3f}\n"
        f"drop={drop_abs:.3f} ({drop_pct:.1f}%)\n"
        f"epochs seen={min(sorted_epochs)}..{max(sorted_epochs)}\n"
        f"points={len(losses_np)}"
    )
    ax_main.text(
        0.012,
        0.98,
        info,
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#111111", edgecolor="#3a3a3a", alpha=0.74),
    )

    # Epoch aggregates
    ax_epoch.fill_between(sorted_epochs, epoch_min, epoch_max, color="#4361ee", alpha=0.22, label="min-max band")
    ax_epoch.plot(sorted_epochs, epoch_mean, color="#ffd166", linewidth=2.0, label="epoch mean loss")
    ax_epoch.set_title("Per-Epoch Loss Envelope", fontsize=12, pad=8)
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_ylabel("Loss")
    ax_epoch.grid(alpha=0.18, linestyle="--")
    ax_epoch.legend(loc="upper right", framealpha=0.18)

    # Loss distribution
    ax_hist.hist(losses_np, bins=42, color="#06d6a0", alpha=0.78, edgecolor="#000000", linewidth=0.35)
    ax_hist.axvline(np.mean(losses_np), color="#ffffff", linewidth=1.2, linestyle="--", label=f"mean={np.mean(losses_np):.3f}")
    ax_hist.axvline(np.median(losses_np), color="#ef476f", linewidth=1.2, linestyle="-.", label=f"median={np.median(losses_np):.3f}")
    ax_hist.set_title("Loss Distribution", fontsize=12, pad=8)
    ax_hist.set_xlabel("Loss")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.12, linestyle="--")
    ax_hist.legend(loc="upper right", framealpha=0.18)

    png_path = output_dir / "lift_training_dashboard.png"
    svg_path = output_dir / "lift_training_dashboard.svg"
    summary_path = output_dir / "lift_training_dashboard_summary.json"

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "log_path": str(log_path),
        "points": int(len(losses_np)),
        "first_loss": float(losses_np[0]),
        "latest_loss": latest_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "loss_drop_abs": drop_abs,
        "loss_drop_pct": drop_pct,
        "min_epoch_seen": int(min(sorted_epochs)),
        "max_epoch_seen": int(max(sorted_epochs)),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
