#!/usr/bin/env python3
"""Grade past predictions against box scores and refit model weights."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.learning import MIN_SAMPLES_FIT, learn_from_history, load_graded
from src.model import load_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade HR predictions vs box scores and refine model weights"
    )
    parser.add_argument(
        "--through",
        default=None,
        help="Grade history through this date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=45,
        help="Only consider history within this many days (default: 45)",
    )
    args = parser.parse_args(argv)

    console = Console()
    through = date.fromisoformat(args.through) if args.through else date.today() - timedelta(days=1)

    console.print(f"[dim]Grading prediction history through {through.isoformat()}...[/dim]")
    result = learn_from_history(through=through, lookback_days=args.lookback_days)

    console.print(f"[+] Days with new/updated grades: {result['dates_graded']}")
    console.print(f"[+] Rows newly appended: {result['rows_added']}")
    console.print(f"[+] Total graded samples: {result['graded_total']} (min to fit: {MIN_SAMPLES_FIT})")

    if result["fitted"] and result["params"]:
        p = result["params"]
        console.print("[green][+] Refit complete -> data/learned_params.json[/green]")
        console.print(
            "    weights: "
            f"batter={p.weight_batter:.3f} pitcher={p.weight_pitcher:.3f} "
            f"park={p.weight_park:.3f} weather={p.weight_weather:.3f} "
            f"rep={p.weight_repertoire:.3f}"
        )
        console.print(
            f"    base_hr_pa={p.base_hr_pa:.4f}  cal_a={p.cal_a:.3f}  cal_b={p.cal_b:.3f}"
        )
        if p.metrics:
            console.print(
                f"    brier {p.metrics.get('brier_prior')} -> {p.metrics.get('brier_new')} | "
                f"logloss {p.metrics.get('logloss_prior')} -> {p.metrics.get('logloss_new')} | "
                f"hr_rate={p.metrics.get('hr_rate')}"
            )
    else:
        need = max(0, MIN_SAMPLES_FIT - result["graded_total"])
        console.print(
            f"[yellow]Not enough graded rows to refit yet "
            f"({need} more needed). Keep running `python run.py` daily.[/yellow]"
        )

    params = load_params()
    graded = load_graded()
    if not graded.empty and "hr_hit" in graded.columns:
        console.print(
            f"[dim]Observed HR rate in graded set: {graded['hr_hit'].mean()*100:.1f}% "
            f"({int(graded['hr_hit'].sum())}/{len(graded)})[/dim]"
        )
    if params.n_samples:
        console.print(f"[dim]Active learned params: {params.n_samples} samples[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
