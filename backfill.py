#!/usr/bin/env python3
"""Backfill prediction history from a start year through yesterday, then grade + refit."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.ingestion import baseline_season_for, build_batter_slate, iter_schedule_dates, refresh_statcast_cache
from src.learning import history_path, learn_from_history, save_prediction_history
from src.model import default_params, score_slate


def _month_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        last = monthrange(y, m)[1]
        w0 = max(start, date(y, m, 1))
        w1 = min(end, date(y, m, last))
        if w0 <= w1:
            windows.append((w0, w1))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return windows


def collect_game_dates(start: date, end: date) -> list[date]:
    dates: list[date] = []
    for w0, w1 in _month_windows(start, end):
        # Restrict months to MLB regular-season window
        if w0.month in {1, 2, 12}:
            continue
        try:
            chunk = iter_schedule_dates(w0, w1, game_types="R")
            dates.extend(chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! schedule range {w0}..{w1} failed: {exc}")
            time.sleep(2)
    return sorted(set(dates))


def preload_baselines(years: list[int], console: Console) -> None:
    for year in sorted(set(years)):
        console.print(f"[dim]Caching baselines for season {year}...[/dim]")
        try:
            refresh_statcast_cache(year, force=False)
            console.print(f"[dim]  OK season {year}[/dim]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]  Failed season {year}: {exc}[/yellow]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill HR predictor history and refit")
    parser.add_argument("--start", default="2022-03-01", help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score days that already have history files",
    )
    parser.add_argument(
        "--grade-every",
        type=int,
        default=30,
        help="Grade+merge every N newly scored days (default 30)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="Optional cap on number of days to score (0 = all)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Seconds to sleep between days (default 1.5; helps Open-Meteo limits)",
    )
    args = parser.parse_args(argv)

    console = Console(force_terminal=True)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    if end < start:
        console.print("[red]end must be >= start[/red]")
        return 1

    console.print(f"Backfill {start.isoformat()} -> {end.isoformat()}")
    console.print("Discovering regular-season game dates...")
    game_dates = collect_game_dates(start, end)
    console.print(f"[+] {len(game_dates)} days with games")

    already = [d for d in game_dates if history_path(d).exists()]
    if args.force:
        remaining = list(game_dates)
        console.print(
            f"[+] Already logged: {len(already)}  |  Will re-score: {len(remaining)} (--force)"
        )
    else:
        remaining = [d for d in game_dates if not history_path(d).exists()]
        console.print(
            f"[+] Already logged: {len(already)}  |  Remaining: {len(remaining)} "
            "(resume skips logged days)"
        )

    if remaining:
        console.print(f"[dim]Next date: {remaining[0].isoformat()}[/dim]")
    else:
        console.print("[green]Nothing left to score in this range.[/green]")

    if args.max_days and args.max_days > 0:
        remaining = remaining[: args.max_days]
        console.print(f"[dim]Capped to next {len(remaining)} days (--max-days)[/dim]")

    work_dates = remaining

    # Prior-season baselines only for years we will actually process
    baseline_src = work_dates if work_dates else game_dates
    baseline_years = sorted({baseline_season_for(d) for d in baseline_src})
    preload_baselines(baseline_years, console)

    params = default_params()  # stable prior while logging factors
    scored = 0
    skipped = 0 if args.force else len(already)
    failed = 0
    since_grade = 0

    for i, d in enumerate(work_dates, start=1):
        console.print(f"[dim][{i}/{len(work_dates)}] scoring {d.isoformat()}...[/dim]", highlight=False)
        try:
            rows, ctx = build_batter_slate(d)
            if not rows:
                skipped += 1
                console.print(f"[dim][{i}/{len(work_dates)}] {d.isoformat()}  (no batters)[/dim]")
                continue
            projections = score_slate(rows, ctx, params=params)
            save_prediction_history(projections, d)
            scored += 1
            since_grade += 1
            console.print(
                f"[{i}/{len(work_dates)}] {d.isoformat()}  "
                f"games~{len({r.game_pk for r in rows})}  batters={len(rows)}"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            console.print(f"[yellow][{i}/{len(work_dates)}] {d.isoformat()} FAILED: {exc}[/yellow]")
            if failed <= 3:
                console.print(f"[dim]{traceback.format_exc(limit=2)}[/dim]")
            time.sleep(max(3.0, args.pause * 2))
            continue

        if since_grade >= args.grade_every:
            console.print("[dim]Interim grade/merge...[/dim]")
            learn_from_history(through=d, lookback_days=max(45, (d - start).days + 5))
            since_grade = 0

        # Be polite to public APIs (esp. Open-Meteo archive)
        time.sleep(args.pause)

    console.print(
        f"Scored={scored} skipped={skipped} failed={failed}. Final grade + refit..."
    )
    result = learn_from_history(through=end, lookback_days=max(45, (end - start).days + 5))
    console.print(
        f"[+] Graded days touched={result['dates_graded']}  "
        f"total labeled={result['graded_total']}  fitted={result['fitted']}"
    )
    if result["fitted"] and result["params"]:
        p = result["params"]
        console.print(
            f"[green][+] learned_params.json updated "
            f"(n={p.n_samples}, brier {p.metrics.get('brier_prior')} -> {p.metrics.get('brier_new')})[/green]"
        )
        console.print(
            f"    weights batter={p.weight_batter:.3f} pitcher={p.weight_pitcher:.3f} "
            f"park={p.weight_park:.3f} weather={p.weight_weather:.3f} rep={p.weight_repertoire:.3f}"
        )
    return 0 if failed == 0 or scored > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
