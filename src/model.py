"""Empirical HR probability scoring engine (with optional learned params)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ingestion import BatterRow, SlateContext

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "data" / "learned_params.json"

# League baseline HR per PA (~3.3% — slightly above raw league ~3% for slate UX)
BASE_HR_PA = 0.033

# Priors: season talent dominates; form is a small shrunk residual
WEIGHT_BATTER = 0.30
WEIGHT_PITCHER = 0.20
WEIGHT_PARK = 0.18
WEIGHT_WEATHER = 0.13
WEIGHT_REPERTOIRE = 0.05
WEIGHT_BATTER_FORM = 0.07
WEIGHT_PITCHER_FORM = 0.07

FACTOR_NAMES = (
    "batter",
    "pitcher",
    "park",
    "weather",
    "repertoire",
    "batter_form",
    "pitcher_form",
)


@dataclass
class ModelParams:
    base_hr_pa: float = BASE_HR_PA
    weight_batter: float = WEIGHT_BATTER
    weight_pitcher: float = WEIGHT_PITCHER
    weight_park: float = WEIGHT_PARK
    weight_weather: float = WEIGHT_WEATHER
    weight_repertoire: float = WEIGHT_REPERTOIRE
    weight_batter_form: float = WEIGHT_BATTER_FORM
    weight_pitcher_form: float = WEIGHT_PITCHER_FORM
    # Platt calibration: sigmoid(a + b * logit(p_raw))
    cal_a: float = 0.0
    cal_b: float = 1.0
    n_samples: int = 0
    updated_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def normalized_weights(self) -> tuple[float, ...]:
        ws = [
            self.weight_batter,
            self.weight_pitcher,
            self.weight_park,
            self.weight_weather,
            self.weight_repertoire,
            self.weight_batter_form,
            self.weight_pitcher_form,
        ]
        s = sum(ws) or 1.0
        return tuple(w / s for w in ws)


def default_params() -> ModelParams:
    return ModelParams()


def load_params(path: Path | None = None) -> ModelParams:
    path = path or PARAMS_PATH
    if not path.exists():
        return default_params()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ModelParams(
            base_hr_pa=float(raw.get("base_hr_pa", BASE_HR_PA)),
            weight_batter=float(raw.get("weight_batter", WEIGHT_BATTER)),
            weight_pitcher=float(raw.get("weight_pitcher", WEIGHT_PITCHER)),
            weight_park=float(raw.get("weight_park", WEIGHT_PARK)),
            weight_weather=float(raw.get("weight_weather", WEIGHT_WEATHER)),
            weight_repertoire=float(raw.get("weight_repertoire", WEIGHT_REPERTOIRE)),
            weight_batter_form=float(raw.get("weight_batter_form", WEIGHT_BATTER_FORM)),
            weight_pitcher_form=float(raw.get("weight_pitcher_form", WEIGHT_PITCHER_FORM)),
            cal_a=float(raw.get("cal_a", 0.0)),
            cal_b=float(raw.get("cal_b", 1.0)),
            n_samples=int(raw.get("n_samples", 0)),
            updated_at=raw.get("updated_at"),
            metrics=dict(raw.get("metrics") or {}),
        )
    except Exception:  # noqa: BLE001
        return default_params()


def save_params(params: ModelParams, path: Path | None = None) -> Path:
    path = path or PARAMS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_hr_pa": params.base_hr_pa,
        "weight_batter": params.weight_batter,
        "weight_pitcher": params.weight_pitcher,
        "weight_park": params.weight_park,
        "weight_weather": params.weight_weather,
        "weight_repertoire": params.weight_repertoire,
        "weight_batter_form": params.weight_batter_form,
        "weight_pitcher_form": params.weight_pitcher_form,
        "cal_a": params.cal_a,
        "cal_b": params.cal_b,
        "n_samples": params.n_samples,
        "updated_at": params.updated_at,
        "metrics": params.metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@dataclass
class Projection:
    rank: int
    player_id: int
    player: str
    team: str
    opp: str
    pitcher: str
    pitcher_hand: str
    venue: str
    wind_temp: str
    p_hr: float
    p_hr_pa: float
    lineup_slot: int
    projected_lineup: bool
    rain_risk: bool
    bat_side: str
    game_pk: int
    factors: dict[str, float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def projected_pa(lineup_slot: int) -> float:
    if lineup_slot <= 3:
        return 4.2
    if lineup_slot <= 6:
        return 4.0
    return 3.8


def _batter_factor(row: BatterRow) -> float:
    """Quality of contact multiplier around 1.0."""
    barrel = row.barrel_pct / 9.0
    hard = row.hard_hit_pct / 40.0
    fb = row.fb_pct / 35.0
    raw = 0.5 * barrel + 0.3 * hard + 0.2 * fb
    return _clamp(raw, 0.55, 1.85)


def _pitcher_factor(row: BatterRow) -> float:
    """Higher = more vulnerable (more HR allowed)."""
    hr9 = row.pitcher.hr9 / 1.20
    barrel = row.pitcher.barrel_pct / 7.0
    fb = row.pitcher.fb_pct / 35.0
    raw = 0.5 * hr9 + 0.3 * barrel + 0.2 * fb
    platoon = (
        (row.bat_side == "L" and row.pitcher.hand == "R")
        or (row.bat_side == "R" and row.pitcher.hand == "L")
    )
    if platoon:
        raw *= 1.08
    else:
        raw *= 0.96
    return _clamp(raw, 0.55, 1.75)


def _park_factor(row: BatterRow, venues: dict[str, Any]) -> float:
    v = venues.get(str(row.venue_id), {})
    if row.bat_side == "L":
        pf = float(v.get("hr_factor_lhb", 1.0))
    else:
        pf = float(v.get("hr_factor_rhb", 1.0))
    elev = float(v.get("elevation_ft", 0) or 0)
    elev_adj = 1.0 + max(0.0, (elev - 500.0) / 5000.0) * 0.15
    return _clamp(pf * elev_adj, 0.70, 1.55)


def wind_label(wind_mph: float, wind_dir: float, azimuth: float, indoors: bool) -> str:
    if indoors or wind_mph < 1.5:
        return "Indoors" if indoors else "Calm"
    rel = (wind_dir - azimuth) % 360.0
    if rel > 180:
        rel -= 360
    if abs(rel) <= 40:
        return f"{wind_mph:.0f}mph OUT"
    if abs(rel) >= 140:
        return f"{wind_mph:.0f}mph IN"
    if 40 < rel < 140:
        return f"{wind_mph:.0f}mph LF"
    return f"{wind_mph:.0f}mph RF"


def _weather_factor(row: BatterRow, venues: dict[str, Any]) -> tuple[float, str]:
    v = venues.get(str(row.venue_id), {})
    azimuth = float(v.get("azimuth_deg", 0) or 0)
    w = row.weather
    label = wind_label(w.wind_mph, w.wind_dir_deg, azimuth, w.indoors)
    temp_label = f"{w.temp_f:.0f}°"
    display = f"{label} / {temp_label}" if not w.indoors else f"Indoors / {temp_label}"

    if w.indoors:
        return 1.0, display

    temp_mult = 1.0 + ((w.temp_f - 70.0) / 10.0) * 0.01
    rel_rad = math.radians((w.wind_dir_deg - azimuth) % 360.0)
    out_component = math.cos(rel_rad)
    wind_mult = 1.0 + (out_component * w.wind_mph / 10.0) * 0.08
    humid_mult = 1.0 - max(0.0, (w.humidity - 50.0) / 100.0) * 0.03
    return _clamp(temp_mult * wind_mult * humid_mult, 0.80, 1.25), display


def _repertoire_factor(row: BatterRow) -> float:
    ff = row.pitcher.ff_usage
    if ff >= 50:
        contact = (row.hard_hit_pct / 40.0 + row.barrel_pct / 9.0) / 2.0
        return _clamp(0.95 + 0.08 * (contact - 1.0) + 0.03, 0.90, 1.12)
    if ff <= 30:
        return 0.98
    return 1.0


def _apply_calibration(p_raw: float, params: ModelParams) -> float:
    p = _clamp(p_raw, 1e-6, 1 - 1e-6)
    logit = math.log(p / (1.0 - p))
    cal = 1.0 / (1.0 + math.exp(-(params.cal_a + params.cal_b * logit)))
    return _clamp(cal, 0.02, 0.45)


def score_batter(
    row: BatterRow,
    venues: dict[str, Any],
    params: ModelParams | None = None,
) -> Projection:
    params = params or default_params()
    wb, wp, wpark, ww, wr, wbf, wpf = params.normalized_weights()

    b = _batter_factor(row)
    p = _pitcher_factor(row)
    park = _park_factor(row, venues)
    weather, wind_temp = _weather_factor(row, venues)
    rep = _repertoire_factor(row)
    b_form = _clamp(float(row.form_factor or 1.0), 0.85, 1.15)
    p_form = _clamp(float(row.pitcher.form_factor or 1.0), 0.85, 1.15)

    log_blend = (
        wb * math.log(b)
        + wp * math.log(p)
        + wpark * math.log(park)
        + ww * math.log(weather)
        + wr * math.log(rep)
        + wbf * math.log(b_form)
        + wpf * math.log(p_form)
    )
    multiplier = math.exp(log_blend)
    p_hr_pa = _clamp(params.base_hr_pa * multiplier, 0.010, 0.14)
    n_pa = projected_pa(row.lineup_slot)
    p_hr_game = 1.0 - (1.0 - p_hr_pa) ** n_pa
    p_hr_game = _clamp(p_hr_game, 0.04, 0.40)
    p_hr_game = _apply_calibration(p_hr_game, params)

    pitcher_label = row.pitcher.name
    if row.pitcher.is_opener_or_bullpen and "Bullpen" not in pitcher_label and "opener" not in pitcher_label:
        pitcher_label = f"{pitcher_label} (bullpen)"

    return Projection(
        rank=0,
        player_id=row.player_id,
        player=row.player_name + ("*" if row.projected_lineup else ""),
        team=row.team,
        opp=row.opp,
        pitcher=pitcher_label,
        pitcher_hand=row.pitcher.hand,
        venue=row.venue_name,
        wind_temp=wind_temp,
        p_hr=p_hr_game,
        p_hr_pa=p_hr_pa,
        lineup_slot=row.lineup_slot,
        projected_lineup=row.projected_lineup,
        rain_risk=row.rain_risk,
        bat_side=row.bat_side,
        game_pk=row.game_pk,
        factors={
            "batter": round(b, 3),
            "pitcher": round(p, 3),
            "park": round(park, 3),
            "weather": round(weather, 3),
            "repertoire": round(rep, 3),
            "batter_form": round(b_form, 3),
            "pitcher_form": round(p_form, 3),
            "n_pa": n_pa,
        },
    )


def score_slate(
    rows: list[BatterRow],
    ctx: SlateContext,
    params: ModelParams | None = None,
) -> list[Projection]:
    params = params or load_params()
    projections = [score_batter(r, ctx.venues, params=params) for r in rows]
    projections.sort(key=lambda x: x.p_hr, reverse=True)
    for i, proj in enumerate(projections, start=1):
        proj.rank = i
    return projections


def projections_to_records(projections: list[Projection]) -> list[dict[str, Any]]:
    records = []
    for p in projections:
        d = asdict(p)
        d["p_hr_pct"] = round(p.p_hr * 100, 2)
        records.append(d)
    return records
