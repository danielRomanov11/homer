"""Recent form: process metrics over a lookback window, shrunk toward season.

Best practices baked in:
- Use quality-of-contact (barrel / hard-hit / FB), not raw HR totals
- Window ends the day before the slate (no same-day leakage)
- Empirical-Bayes shrink: w = n / (n + prior_n); low samples → form ≈ 1.0
- Tight clamps so form adjusts, never dominates season / park / weather
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
STATCAST_DIR = CACHE_DIR / "statcast_months"
FORM_AGG_DIR = CACHE_DIR / "form"

# ~5 weeks of games; early-season windows naturally shrink via sample size
LOOKBACK_DAYS = 35
# Pseudo-counts: need roughly a month of BIP before form is trusted fully
PRIOR_BBE_BATTER = 40.0
PRIOR_BBE_PITCHER = 30.0
MIN_BBE_USE = 8  # below this, treat as missing → form 1.0
FORM_TTL_HOURS = 12  # refresh in-progress month / live aggregates

# League baselines for normalizing recent scores (same anchors as season factors)
LEAGUE_BARREL_BAT = 9.0
LEAGUE_HARD_BAT = 40.0
LEAGUE_FB = 35.0
LEAGUE_BARREL_PIT = 7.0
LEAGUE_HR9 = 1.20


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def form_window(slate_date: date, lookback_days: int = LOOKBACK_DAYS) -> tuple[date, date]:
    """Inclusive [start, end] of games used for form; end is day before slate."""
    end = slate_date - timedelta(days=1)
    start = end - timedelta(days=lookback_days - 1)
    return start, end


def _month_starts(start: date, end: date) -> list[date]:
    months: list[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        months.append(date(y, m, 1))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return months


def _month_path(year: int, month: int) -> Path:
    STATCAST_DIR.mkdir(parents=True, exist_ok=True)
    return STATCAST_DIR / f"statcast_{year}_{month:02d}.parquet"


def _month_meta_path(year: int, month: int) -> Path:
    return STATCAST_DIR / f"statcast_{year}_{month:02d}.meta.json"


def _month_fresh(year: int, month: int, through: date) -> bool:
    """Completed months are immutable; current/partial months respect TTL."""
    path = _month_path(year, month)
    meta_p = _month_meta_path(year, month)
    if not path.exists():
        return False
    month_end = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1) - timedelta(
        days=1
    )
    if month_end < through and month_end < date.today() - timedelta(days=2):
        return True
    if not meta_p.exists():
        return False
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(meta["fetched_at"])
        return datetime.now(timezone.utc) - fetched < timedelta(hours=FORM_TTL_HOURS)
    except Exception:  # noqa: BLE001
        return False


def _fetch_statcast_range(start: date, end: date) -> pd.DataFrame:
    """Pull Statcast for [start, end]; empty frame on failure."""
    if end < start:
        return pd.DataFrame()
    try:
        from pybaseball import statcast
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    try:
        df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _ensure_statcast_month(year: int, month: int, through: date) -> pd.DataFrame:
    """Load or refresh one calendar month of Statcast (clipped to `through`)."""
    path = _month_path(year, month)
    if _month_fresh(year, month, through):
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            pass

    month_start = date(year, month, 1)
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    month_end = min(next_month - timedelta(days=1), through)
    if month_end < month_start:
        return pd.DataFrame()

    # Prefer append for in-progress months: only fetch missing tail
    existing = pd.DataFrame()
    fetch_start = month_start
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if not existing.empty and "game_date" in existing.columns:
                gd = pd.to_datetime(existing["game_date"], errors="coerce")
                last = gd.max()
                if pd.notna(last):
                    last_d = last.date() if hasattr(last, "date") else date.fromisoformat(str(last)[:10])
                    fetch_start = last_d + timedelta(days=1)
        except Exception:  # noqa: BLE001
            existing = pd.DataFrame()

    if fetch_start > month_end:
        return existing

    # Chunk by week to keep requests smaller / more resilient
    chunks: list[pd.DataFrame] = []
    cursor = fetch_start
    while cursor <= month_end:
        chunk_end = min(cursor + timedelta(days=6), month_end)
        part = _fetch_statcast_range(cursor, chunk_end)
        if not part.empty:
            chunks.append(part)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.35)

    if chunks:
        fresh = pd.concat(chunks, ignore_index=True)
        if not existing.empty:
            combined = pd.concat([existing, fresh], ignore_index=True)
        else:
            combined = fresh
        # Dedupe on common pitch keys when present
        subset = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        keep = [
            c
            for c in (
                "game_date",
                "game_pk",
                "batter",
                "pitcher",
                "launch_speed",
                "launch_angle",
                "launch_speed_angle",
                "events",
                "type",
                "at_bat_number",
                "pitch_number",
            )
            if c in combined.columns
        ]
        combined = combined[keep] if keep else combined
        try:
            combined.to_parquet(path, index=False)
            _month_meta_path(year, month).write_text(
                json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "through": through.isoformat()}),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass
        return combined

    return existing


def load_statcast_window(start: date, end: date) -> pd.DataFrame:
    """Statcast pitches for games in [start, end], from monthly cache."""
    frames: list[pd.DataFrame] = []
    for m0 in _month_starts(start, end):
        df = _ensure_statcast_month(m0.year, m0.month, through=end)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "game_date" not in out.columns:
        return pd.DataFrame()
    gd = pd.to_datetime(out["game_date"], errors="coerce")
    mask = (gd >= pd.Timestamp(start)) & (gd <= pd.Timestamp(end))
    return out.loc[mask].copy()


def _batted_balls(df: pd.DataFrame) -> pd.DataFrame:
    """In-play contact with measured launch data."""
    if df.empty:
        return df
    out = df.copy()
    if "type" in out.columns:
        out = out[out["type"].astype(str).str.upper() == "X"]
    # Require a measured EV (filters fouls / missing tracking)
    if "launch_speed" in out.columns:
        out = out[pd.to_numeric(out["launch_speed"], errors="coerce").notna()]
    return out


def _agg_contact(group: pd.DataFrame) -> dict[str, float]:
    n = len(group)
    if n == 0:
        return {"n_bbe": 0.0, "barrel_pct": float("nan"), "hard_hit_pct": float("nan"), "fb_pct": float("nan"), "hr": 0.0}

    if "launch_speed_angle" in group.columns:
        lsa = pd.to_numeric(group["launch_speed_angle"], errors="coerce")
        barrels = (lsa == 6).sum()
    else:
        # Fallback barrel proxy: EV >= 98 and LA in 26–30 (Savant-ish)
        ev = pd.to_numeric(group.get("launch_speed"), errors="coerce")
        la = pd.to_numeric(group.get("launch_angle"), errors="coerce")
        barrels = ((ev >= 98) & (la >= 26) & (la <= 30)).sum()

    ev = pd.to_numeric(group["launch_speed"], errors="coerce") if "launch_speed" in group.columns else pd.Series(dtype=float)
    hard = int((ev >= 95).sum()) if len(ev) else 0

    la = pd.to_numeric(group["launch_angle"], errors="coerce") if "launch_angle" in group.columns else pd.Series(dtype=float)
    # Fly-ball band (excl. popups): ~25–50°
    fb = int(((la >= 25) & (la <= 50)).sum()) if len(la) else 0

    hr = 0
    if "events" in group.columns:
        hr = int(group["events"].astype(str).str.lower().eq("home_run").sum())

    return {
        "n_bbe": float(n),
        "barrel_pct": 100.0 * float(barrels) / n,
        "hard_hit_pct": 100.0 * float(hard) / n,
        "fb_pct": 100.0 * float(fb) / n,
        "hr": float(hr),
    }


def aggregate_batter_form(statcast_df: pd.DataFrame) -> pd.DataFrame:
    bip = _batted_balls(statcast_df)
    if bip.empty or "batter" not in bip.columns:
        return pd.DataFrame(columns=["player_id", "n_bbe", "barrel_pct", "hard_hit_pct", "fb_pct"])
    rows = []
    for pid, grp in bip.groupby(pd.to_numeric(bip["batter"], errors="coerce")):
        if pd.isna(pid):
            continue
        stats = _agg_contact(grp)
        rows.append({"player_id": int(pid), **stats})
    return pd.DataFrame(rows)


def aggregate_pitcher_form(statcast_df: pd.DataFrame) -> pd.DataFrame:
    bip = _batted_balls(statcast_df)
    if bip.empty or "pitcher" not in bip.columns:
        return pd.DataFrame(columns=["player_id", "n_bbe", "barrel_pct", "hard_hit_pct", "fb_pct", "hr9"])
    rows = []
    for pid, grp in bip.groupby(pd.to_numeric(bip["pitcher"], errors="coerce")):
        if pd.isna(pid):
            continue
        stats = _agg_contact(grp)
        # HR/9 proxy: scale HR/BBE to league HR/9 using ~27 BIP / 9 IP heuristic
        n = max(stats["n_bbe"], 1.0)
        hr_per_bbe = stats["hr"] / n
        # League ~0.04 HR/BBE ↔ ~1.2 HR/9 → hr9 ≈ hr_per_bbe * (1.2 / 0.04) = hr_per_bbe * 30
        stats["hr9"] = hr_per_bbe * 30.0
        rows.append({"player_id": int(pid), **stats})
    return pd.DataFrame(rows)


def _contact_score_batter(barrel: float, hard: float, fb: float) -> float:
    return 0.5 * (barrel / LEAGUE_BARREL_BAT) + 0.3 * (hard / LEAGUE_HARD_BAT) + 0.2 * (fb / LEAGUE_FB)


def _contact_score_pitcher(hr9: float, barrel: float, fb: float) -> float:
    """Higher = more HR vulnerability (same direction as season pitcher factor)."""
    return 0.5 * (hr9 / LEAGUE_HR9) + 0.3 * (barrel / LEAGUE_BARREL_PIT) + 0.2 * (fb / LEAGUE_FB)


def shrink_form_ratio(
    recent_score: float,
    season_score: float,
    n_bbe: float,
    prior_n: float,
    lo: float = 0.88,
    hi: float = 1.12,
) -> float:
    """Empirical-Bayes blend of recent/season ratio toward 1.0."""
    if n_bbe < MIN_BBE_USE or season_score <= 1e-6 or recent_score != recent_score:
        return 1.0
    ratio = recent_score / season_score
    w = float(n_bbe) / (float(n_bbe) + float(prior_n))
    return _clamp(w * ratio + (1.0 - w) * 1.0, lo, hi)


def _agg_cache_paths(slate_date: date) -> tuple[Path, Path, Path]:
    FORM_AGG_DIR.mkdir(parents=True, exist_ok=True)
    start, end = form_window(slate_date)
    stem = f"form_{start.isoformat()}_{end.isoformat()}"
    return (
        FORM_AGG_DIR / f"{stem}_batters.parquet",
        FORM_AGG_DIR / f"{stem}_pitchers.parquet",
        FORM_AGG_DIR / f"{stem}.meta.json",
    )


def _agg_fresh(meta_path: Path, end: date) -> bool:
    if not meta_path.exists():
        return False
    # Historical windows fully in the past are immutable
    if end < date.today() - timedelta(days=2):
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(meta["fetched_at"])
        return datetime.now(timezone.utc) - fetched < timedelta(hours=FORM_TTL_HOURS)
    except Exception:  # noqa: BLE001
        return False


def load_form_tables(slate_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Batter/pitcher form aggregates for the slate's lookback window.
    Returns empty frames when Statcast is unavailable (callers treat as neutral form).
    """
    bat_p, pit_p, meta_p = _agg_cache_paths(slate_date)
    start, end = form_window(slate_date)
    if end < start:
        return pd.DataFrame(), pd.DataFrame()

    if (
        bat_p.exists()
        and pit_p.exists()
        and _agg_fresh(meta_p, end)
    ):
        try:
            return pd.read_parquet(bat_p), pd.read_parquet(pit_p)
        except Exception:  # noqa: BLE001
            pass

    raw = load_statcast_window(start, end)
    batters = aggregate_batter_form(raw)
    pitchers = aggregate_pitcher_form(raw)
    try:
        batters.to_parquet(bat_p, index=False)
        pitchers.to_parquet(pit_p, index=False)
        meta_p.write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "n_pitches": int(len(raw)),
                }
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass
    return batters, pitchers


def batter_form_factor(
    player_id: int,
    season_barrel: float,
    season_hard: float,
    season_fb: float,
    form_df: pd.DataFrame,
) -> float:
    if form_df is None or form_df.empty or "player_id" not in form_df.columns:
        return 1.0
    match = form_df[form_df["player_id"] == int(player_id)]
    if match.empty:
        return 1.0
    row = match.iloc[0]
    n = float(row["n_bbe"]) if pd.notna(row.get("n_bbe")) else 0.0
    recent = _contact_score_batter(
        float(row["barrel_pct"]) if pd.notna(row.get("barrel_pct")) else season_barrel,
        float(row["hard_hit_pct"]) if pd.notna(row.get("hard_hit_pct")) else season_hard,
        float(row["fb_pct"]) if pd.notna(row.get("fb_pct")) else season_fb,
    )
    season = _contact_score_batter(season_barrel, season_hard, season_fb)
    return shrink_form_ratio(recent, season, n, PRIOR_BBE_BATTER)


def pitcher_form_factor(
    pitcher_id: int | None,
    season_hr9: float,
    season_barrel: float,
    season_fb: float,
    form_df: pd.DataFrame,
) -> float:
    if pitcher_id is None or form_df is None or form_df.empty or "player_id" not in form_df.columns:
        return 1.0
    match = form_df[form_df["player_id"] == int(pitcher_id)]
    if match.empty:
        return 1.0
    row = match.iloc[0]
    n = float(row["n_bbe"]) if pd.notna(row.get("n_bbe")) else 0.0
    recent = _contact_score_pitcher(
        float(row["hr9"]) if pd.notna(row.get("hr9")) else season_hr9,
        float(row["barrel_pct"]) if pd.notna(row.get("barrel_pct")) else season_barrel,
        float(row["fb_pct"]) if pd.notna(row.get("fb_pct")) else season_fb,
    )
    season = _contact_score_pitcher(season_hr9, season_barrel, season_fb)
    return shrink_form_ratio(recent, season, n, PRIOR_BBE_PITCHER)


def form_summary(slate_date: date) -> dict[str, Any]:
    """Debug helper: window + cache sizes."""
    start, end = form_window(slate_date)
    bat, pit = load_form_tables(slate_date)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_batters": int(len(bat)),
        "n_pitchers": int(len(pit)),
    }
