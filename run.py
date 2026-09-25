#!/usr/bin/env python3
"""Zero-arg entrypoint: Daily MLB Home Run Predictor."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Allow running as `python run.py` from repo root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.display import export_csv, render_table
from src.ingestion import build_batter_slate
from src.model import score_slate


def main() -> int:
    console = Console()
    slate_date = date.today()
    console.print(f"[dim]Building slate for {slate_date.isoformat()}…[/dim]")

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

    projections = score_slate(rows, ctx)
    n_games = len({r.game_pk for r in rows})
    csv_path = export_csv(projections, slate_date)
    render_table(projections, slate_date, n_games=n_games, csv_path=csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
