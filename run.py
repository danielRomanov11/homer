#!/usr/bin/env python3
"""Zero-arg entrypoint: Daily MLB Home Run Predictor."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.display import export_csv, render_table
from src.ingestion import build_batter_slate
from src.learning import learn_from_history, save_prediction_history
from src.model import load_params, score_slate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily MLB Home Run Predictor")
    parser.add_argument(
        "--date",
        dest="slate_date",
        default=None,
        help="Slate date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--skip-learn",
        action="store_true",
        help="Skip grading prior days / refitting learned weights",
    )
    args = parser.parse_args(argv)

    console = Console()
    slate_date = date.fromisoformat(args.slate_date) if args.slate_date else date.today()

    if not args.skip_learn:
        # When projecting today, grade through yesterday; when backfilling, grade through that date.
        through = min(slate_date, date.today() - timedelta(days=1))
        console.print("[dim]Grading prior predictions and updating learned weights...[/dim]")
        result = learn_from_history(through=through)
        console.print(
            f"[dim][+] Graded {result['dates_graded']} day(s), "
            f"{result['graded_total']} labeled rows "
            f"(need {result['min_samples']} to fit).[/dim]"
        )
        if result["fitted"] and result["params"]:
            p = result["params"]
            console.print(
                f"[dim][+] Updated learned params from {p.n_samples} samples "
                f"-> data/learned_params.json[/dim]"
            )

    params = load_params()
    if params.n_samples:
        updated = f", updated {params.updated_at[:10]}" if params.updated_at else ""
        console.print(f"[dim]Using learned weights ({params.n_samples} samples{updated}).[/dim]")
    else:
        console.print("[dim]Using default PRD weights (not enough graded history yet).[/dim]")

    console.print(f"[dim]Building slate for {slate_date.isoformat()}...[/dim]")
    console.print(
        "[dim]Loading season baselines + ~35-day form (first run may take a bit)...[/dim]"
    )

    try:
        rows, ctx = build_batter_slate(slate_date)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to build slate:[/red] {exc}")
        return 1

    if not rows:
        console.print(
            f"[yellow]No MLB games (or lineups) found for {slate_date.isoformat()}.[/yellow]"
        )
        return 0

    projections = score_slate(rows, ctx, params=params)
    n_games = len({r.game_pk for r in rows})
    csv_path = export_csv(projections, slate_date)
    hist_path = save_prediction_history(projections, slate_date)
    render_table(projections, slate_date, n_games=n_games, csv_path=csv_path)
    try:
        shown = hist_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        shown = hist_path.as_posix()
    console.print(f"[dim][+] Logged predictions for learning -> {shown}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
