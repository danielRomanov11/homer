"""Self-learning loop: prediction history, box-score grading, weight refit."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingestion import MLB_API, _get_json
from .model import (
    BASE_HR_PA,
    WEIGHT_BATTER,
    WEIGHT_PARK,
    WEIGHT_PITCHER,
    WEIGHT_REPERTOIRE,
    WEIGHT_WEATHER,
    ModelParams,
    Projection,
    save_params,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / ".cache" / "history"
GRADED_PATH = HISTORY_DIR / "graded.parquet"
MIN_SAMPLES_FIT = 80
PRIOR_STRENGTH = 200.0  # pseudo-counts blending toward default weights
FACTOR_COLS = ("batter", "pitcher", "park", "weather", "repertoire")


def history_path(slate_date: date) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR / f"predictions_{slate_date.isoformat()}.parquet"


def save_prediction_history(
    projections: list[Projection],
    slate_date: date,
) -> Path:
    """Persist daily projections (with factors + player_id) for later grading."""
    records = []
    for p in projections:
        factors = p.factors or {}
        records.append(
            {
                "slate_date": slate_date.isoformat(),
                "player_id": int(p.player_id),
                "player": p.player,
                "team": p.team,
                "opp": p.opp,
                "game_pk": int(p.game_pk),
                "lineup_slot": int(p.lineup_slot),
                "projected_lineup": bool(p.projected_lineup),
                "p_hr": float(p.p_hr),
                "p_hr_pa": float(p.p_hr_pa),
                "factor_batter": float(factors.get("batter", 1.0)),
                "factor_pitcher": float(factors.get("pitcher", 1.0)),
                "factor_park": float(factors.get("park", 1.0)),
                "factor_weather": float(factors.get("weather", 1.0)),
                "factor_repertoire": float(factors.get("repertoire", 1.0)),
                "n_pa": float(factors.get("n_pa", 4.0)),
                "logged_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    df = pd.DataFrame(records)
    path = history_path(slate_date)
    df.to_parquet(path, index=False)
    return path


def list_history_dates() -> list[date]:
    if not HISTORY_DIR.exists():
        return []
    out: list[date] = []
    for p in HISTORY_DIR.glob("predictions_*.parquet"):
        try:
            out.append(date.fromisoformat(p.stem.replace("predictions_", "")))
        except ValueError:
            continue
    return sorted(out)


def load_history(slate_date: date) -> pd.DataFrame:
    path = history_path(slate_date)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def fetch_game_hr_map(game_pk: int) -> dict[int, int]:
    """Return {player_id: home_runs} from boxscore. Empty if game not final / missing."""
    try:
        box = _get_json(f"{MLB_API}/game/{game_pk}/boxscore")
    except Exception:  # noqa: BLE001
        return {}
    hr_map: dict[int, int] = {}
    for side in ("home", "away"):
        players = (box.get("teams") or {}).get(side, {}).get("players") or {}
        for _key, pdata in players.items():
            person = pdata.get("person") or {}
            pid = person.get("id")
            if not pid:
                continue
            batting = (pdata.get("stats") or {}).get("batting") or {}
            if not batting:
                continue
            try:
                hr = int(batting.get("homeRuns") or 0)
            except (TypeError, ValueError):
                hr = 0
            hr_map[int(pid)] = hr
    return hr_map


def _game_is_final(game_pk: int) -> bool:
    try:
        data = _get_json(f"{MLB_API}/schedule", params={"gamePk": game_pk, "sportId": 1})
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if int(g.get("gamePk", 0)) == int(game_pk):
                    return (g.get("status") or {}).get("abstractGameState") == "Final"
    except Exception:  # noqa: BLE001
        return False
    return False


def grade_history_date(slate_date: date, assume_final: bool = False) -> pd.DataFrame:
    """Join predictions to box scores; return graded rows for that date."""
    hist = load_history(slate_date)
    if hist.empty:
        return pd.DataFrame()

    graded_rows: list[dict[str, Any]] = []
    for game_pk, grp in hist.groupby("game_pk"):
        if not assume_final and not _game_is_final(int(game_pk)):
            continue
        hr_map = fetch_game_hr_map(int(game_pk))
        for _, row in grp.iterrows():
            pid = int(row["player_id"])
            hr_count = int(hr_map.get(pid, 0))
            graded_rows.append(
                {
                    **row.to_dict(),
                    "hr_count": hr_count,
                    "hr_hit": int(hr_count > 0),
                    "graded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return pd.DataFrame(graded_rows)


def append_graded(df: pd.DataFrame) -> int:
    """Merge graded rows into master graded store; returns rows in df."""
    if df.empty:
        return 0
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    key_cols = ["slate_date", "player_id", "game_pk"]
    df = df.copy()
    df["slate_date"] = df["slate_date"].astype(str)
    if GRADED_PATH.exists():
        existing = pd.read_parquet(GRADED_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined["slate_date"] = combined["slate_date"].astype(str)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined.to_parquet(GRADED_PATH, index=False)
    else:
        df.to_parquet(GRADED_PATH, index=False)
    return len(df)


def load_graded() -> pd.DataFrame:
    if not GRADED_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(GRADED_PATH)


def grade_pending(through: date | None = None, lookback_days: int = 45) -> tuple[int, int]:
    """
    Grade any history dates with completed games up through `through`.
    Returns (dates_graded, new_unique_rows).
    """
    through = through or (date.today() - timedelta(days=1))
    dates = [d for d in list_history_dates() if d <= through]
    if lookback_days:
        cutoff = through - timedelta(days=lookback_days)
        dates = [d for d in dates if d >= cutoff]

    before = load_graded()
    before_keys: set[tuple] = set()
    graded_dates: set[str] = set()
    if not before.empty:
        before_keys = set(
            zip(
                before["slate_date"].astype(str),
                before["player_id"].astype(int),
                before["game_pk"].astype(int),
            )
        )
        # Dates already fully present in graded store (skip re-fetch)
        counts = before.groupby(before["slate_date"].astype(str)).size().to_dict()
        for d in dates:
            hist = load_history(d)
            if hist.empty:
                continue
            if counts.get(d.isoformat(), 0) >= len(hist):
                graded_dates.add(d.isoformat())

    dates_done = 0
    new_rows = 0
    for d in dates:
        if d.isoformat() in graded_dates:
            continue
        graded = grade_history_date(d, assume_final=(d < date.today()))
        if graded.empty:
            continue
        graded = graded.copy()
        graded["slate_date"] = graded["slate_date"].astype(str)
        fresh = graded[
            ~graded.apply(
                lambda r: (str(r["slate_date"]), int(r["player_id"]), int(r["game_pk"]))
                in before_keys,
                axis=1,
            )
        ]
        append_graded(graded)
        dates_done += 1
        new_rows += len(fresh)
        for _, r in graded.iterrows():
            before_keys.add((str(r["slate_date"]), int(r["player_id"]), int(r["game_pk"])))
    return dates_done, new_rows


def _predict_p(
    factors: np.ndarray,
    n_pa: np.ndarray,
    weights: np.ndarray,
    base_hr_pa: float,
    cal_a: float,
    cal_b: float,
) -> np.ndarray:
    """factors shape (n, 5) raw multipliers; weights sum to 1."""
    log_f = np.log(np.clip(factors, 1e-6, None))
    log_blend = log_f @ weights
    multiplier = np.exp(log_blend)
    p_pa = np.clip(base_hr_pa * multiplier, 0.010, 0.14)
    p_raw = 1.0 - np.power(1.0 - p_pa, n_pa)
    p_raw = np.clip(p_raw, 0.04, 0.40)
    # Platt on logit
    p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
    logit = np.log(p_raw / (1.0 - p_raw))
    cal = 1.0 / (1.0 + np.exp(-(cal_a + cal_b * logit)))
    return np.clip(cal, 0.02, 0.45)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_params(graded: pd.DataFrame | None = None) -> ModelParams | None:
    """
    Refit factor weights + Platt calibration from graded history.
    Blends toward PRD priors until enough samples accumulate.
    """
    graded = load_graded() if graded is None else graded
    if graded.empty or len(graded) < MIN_SAMPLES_FIT:
        return None

    y = graded["hr_hit"].astype(float).to_numpy()
    factors = graded[
        [f"factor_{c}" for c in FACTOR_COLS]
    ].astype(float).to_numpy()
    n_pa = graded["n_pa"].astype(float).to_numpy()

    prior = np.array(
        [WEIGHT_BATTER, WEIGHT_PITCHER, WEIGHT_PARK, WEIGHT_WEATHER, WEIGHT_REPERTOIRE],
        dtype=float,
    )
    prior = prior / prior.sum()

    # Unconstrained logits for softmax weights
    w_logits = np.log(prior)

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def objective(x: np.ndarray) -> float:
        # x: [5 logits, base_hr_pa, cal_a, cal_b]
        weights = softmax(x[:5])
        base = float(np.clip(x[5], 0.015, 0.06))
        cal_a, cal_b = float(x[6]), float(np.clip(x[7], 0.2, 3.0))
        p = _predict_p(factors, n_pa, weights, base, cal_a, cal_b)
        nll = _log_loss(y, p)
        # Prior pull on weights + base
        w_pen = float(np.sum((weights - prior) ** 2)) * (PRIOR_STRENGTH / max(len(y), 1))
        base_pen = ((base - BASE_HR_PA) / 0.02) ** 2 * 0.05
        cal_pen = (cal_a**2) * 0.01 + ((cal_b - 1.0) ** 2) * 0.02
        return nll + w_pen + base_pen + cal_pen

    x0 = np.concatenate([w_logits, np.array([BASE_HR_PA, 0.0, 1.0])])

    try:
        from scipy.optimize import minimize

        res = minimize(objective, x0, method="L-BFGS-B")
        x = res.x if res.success else x0
    except Exception:  # noqa: BLE001
        # Fallback: coordinate-free prior + simple calibration on residuals
        x = x0
        p0 = _predict_p(factors, n_pa, prior, BASE_HR_PA, 0.0, 1.0)
        # Univariate Platt via logit regression (closed-form-ish Newton)
        logit = np.log(np.clip(p0, 1e-6, 1 - 1e-6) / np.clip(1 - p0, 1e-6, 1 - 1e-6))
        # ridge logistic for a, b
        a, b = 0.0, 1.0
        for _ in range(40):
            eta = a + b * logit
            p = 1 / (1 + np.exp(-eta))
            w = p * (1 - p)
            # update a
            ga = np.sum(p - y) + 0.01 * a
            ha = np.sum(w) + 0.01
            a -= ga / max(ha, 1e-6)
            gb = np.sum((p - y) * logit) + 0.01 * (b - 1)
            hb = np.sum(w * logit * logit) + 0.01
            b -= gb / max(hb, 1e-6)
        x[6], x[7] = a, float(np.clip(b, 0.2, 3.0))

    weights = softmax(x[:5])
    # Blend toward prior based on sample size
    n = float(len(y))
    mix = n / (n + PRIOR_STRENGTH)
    weights = mix * weights + (1.0 - mix) * prior
    weights = weights / weights.sum()

    base = float(np.clip(mix * x[5] + (1.0 - mix) * BASE_HR_PA, 0.015, 0.06))
    cal_a = float(mix * x[6])
    cal_b = float(np.clip(mix * x[7] + (1.0 - mix) * 1.0, 0.2, 3.0))

    params = ModelParams(
        base_hr_pa=base,
        weight_batter=float(weights[0]),
        weight_pitcher=float(weights[1]),
        weight_park=float(weights[2]),
        weight_weather=float(weights[3]),
        weight_repertoire=float(weights[4]),
        cal_a=cal_a,
        cal_b=cal_b,
        n_samples=int(n),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Metrics for reporting (stored on params via save sidecar optional)
    p_new = _predict_p(factors, n_pa, weights, base, cal_a, cal_b)
    p_old = _predict_p(
        factors,
        n_pa,
        prior,
        BASE_HR_PA,
        0.0,
        1.0,
    )
    params.metrics = {
        "brier_new": round(_brier(y, p_new), 5),
        "brier_prior": round(_brier(y, p_old), 5),
        "logloss_new": round(_log_loss(y, p_new), 5),
        "logloss_prior": round(_log_loss(y, p_old), 5),
        "hr_rate": round(float(y.mean()), 4),
        "mean_p_new": round(float(p_new.mean()), 4),
    }
    save_params(params)
    return params


def learn_from_history(
    through: date | None = None,
    lookback_days: int = 45,
) -> dict[str, Any]:
    """Grade pending history and refit params when enough samples exist."""
    dates_graded, rows_added = grade_pending(through=through, lookback_days=lookback_days)
    graded = load_graded()
    n = len(graded)
    params = None
    if n >= MIN_SAMPLES_FIT:
        params = fit_params(graded)
    return {
        "dates_graded": dates_graded,
        "rows_added": rows_added,
        "graded_total": n,
        "fitted": params is not None,
        "params": params,
        "min_samples": MIN_SAMPLES_FIT,
    }
