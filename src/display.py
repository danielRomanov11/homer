"""Terminal dashboard and CSV export."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console

from .model import Projection, projections_to_records

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def _pitcher_cell(p: Projection) -> str:
    last = p.pitcher
    parts = (
        last.replace(" (opener)", "")
        .replace(" (bullpen)", "")
        .split()
    )
    suffix = ""
    if "(opener)" in p.pitcher:
        suffix = " (opener)"
    elif "Bullpen" in p.pitcher:
        return p.pitcher
    elif "(bullpen)" in p.pitcher:
        suffix = " (bullpen)"
    if len(parts) >= 2:
        short = f"{parts[0][0]}. {' '.join(parts[1:])}{suffix}"
    else:
        short = last
    return f"{short} ({p.pitcher_hand})"


def _fmt(text: str, width: int) -> str:
    text = text or ""
    if len(text) > width:
        return text[: width - 3] + "..."
    return text.ljust(width)


def render_table(
    projections: list[Projection],
    slate_date: date,
    n_games: int,
    csv_path: Path | None = None,
    top_n: int = 25,
) -> None:
    console = Console(force_terminal=True)
    width = 108
    header = f"DAILY MLB HOME RUN PREDICTOR | {slate_date.isoformat()}"
    console.print("=" * width)
    console.print(header)
    console.print("=" * width)

    col_header = (
        f"{'Rank':>4}  "
        f"{_fmt('Player', 22)}  "
        f"{_fmt('Team', 4)}  "
        f"{_fmt('Opp', 4)}  "
        f"{_fmt('Pitcher (Hand)', 18)}  "
        f"{_fmt('Venue', 20)}  "
        f"{_fmt('Wind / Temp', 14)}  "
        f"{'P(HR)':>6}"
    )
    console.print(col_header)
    console.print("-" * width)

    for p in projections[:top_n]:
        line = (
            f"{p.rank:>4}  "
            f"{_fmt(p.player, 22)}  "
            f"{_fmt(p.team, 4)}  "
            f"{_fmt(p.opp, 4)}  "
            f"{_fmt(_pitcher_cell(p), 18)}  "
            f"{_fmt(p.venue, 20)}  "
            f"{_fmt(p.wind_temp, 14)}  "
            f"{p.p_hr * 100:5.1f}%"
        )
        if p.rain_risk:
            console.print(f"[yellow]{line}[/yellow]")
        elif p.p_hr >= 0.22:
            console.print(f"[bold green]{line}[/bold green]")
        elif p.p_hr >= 0.16:
            console.print(f"[green]{line}[/green]")
        else:
            console.print(line)

    console.print("-" * width)
    console.print(f"[+] Processed {n_games} games, {len(projections)} batters.")
    if csv_path:
        try:
            shown = csv_path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            shown = csv_path.as_posix()
        console.print(f"[+] Saved full slate projections to {shown}")


def export_csv(projections: list[Projection], slate_date: date) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"hr_picks_{slate_date.isoformat()}.csv"
    records = projections_to_records(projections)
    df = pd.DataFrame(records)
    if "factors" in df.columns:
        factors = pd.json_normalize(df["factors"])
        factors = factors.add_prefix("factor_")
        df = pd.concat([df.drop(columns=["factors"]), factors], axis=1)
    cols = [
        "rank",
        "player",
        "team",
        "opp",
        "pitcher",
        "pitcher_hand",
        "venue",
        "wind_temp",
        "p_hr_pct",
        "p_hr",
        "p_hr_pa",
        "lineup_slot",
        "projected_lineup",
        "rain_risk",
        "bat_side",
        "game_pk",
    ]
    extra = [c for c in df.columns if c not in cols]
    df = df[[c for c in cols if c in df.columns] + extra]
    df.to_csv(path, index=False)
    return path
