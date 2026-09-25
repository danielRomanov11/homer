"""Empirical HR probability scoring engine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .ingestion import BatterRow, SlateContext

# League baseline HR per PA (~3.3% — slightly above raw league ~3% for slate UX)
BASE_HR_PA = 0.033

WEIGHT_BATTER = 0.35
WEIGHT_PITCHER = 0.25
WEIGHT_PARK = 0.20
WEIGHT_WEATHER = 0.15
WEIGHT_REPERTOIRE = 0.05


@dataclass
class Projection:
    rank: int
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
    # Typical qualified: barrel ~8–10%, hard hit ~40%, FB ~35%
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
    # Platoon advantage
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
    # Extra air-density lift beyond park factor (Coors already high PF; mild additive)
    elev_adj = 1.0 + max(0.0, (elev - 500.0) / 5000.0) * 0.15
    return _clamp(pf * elev_adj, 0.70, 1.55)


def wind_label(wind_mph: float, wind_dir: float, azimuth: float, indoors: bool) -> str:
    if indoors or wind_mph < 1.5:
        return "Indoors" if indoors else "Calm"
    # Relative angle: 0 = blowing out to CF, 180 = in from CF
    rel = (wind_dir - azimuth) % 360.0
    if rel > 180:
        rel -= 360
    # Classify
    if abs(rel) <= 40:
        return f"{wind_mph:.0f}mph OUT"
    if abs(rel) >= 140:
        return f"{wind_mph:.0f}mph IN"
    if 40 < rel < 140:
        # wind from RF toward LF roughly → helps LF pull? use LF/RF labels
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

    # ~1% carry per +10°F vs 70°F
    temp_mult = 1.0 + ((w.temp_f - 70.0) / 10.0) * 0.01
    # Wind vector: cos(rel) positive = outward
    rel_rad = math.radians((w.wind_dir_deg - azimuth) % 360.0)
    out_component = math.cos(rel_rad)  # +1 out to CF
    # Scale: 10 mph full out ≈ +8% HR chance component
    wind_mult = 1.0 + (out_component * w.wind_mph / 10.0) * 0.08
    # Mild humidity penalty (dense air)
    humid_mult = 1.0 - max(0.0, (w.humidity - 50.0) / 100.0) * 0.03
    return _clamp(temp_mult * wind_mult * humid_mult, 0.80, 1.25), display


def _repertoire_factor(row: BatterRow) -> float:
    """Small bump when pitcher leans on FF and batter has positive FF matchup proxy."""
    ff = row.pitcher.ff_usage
    if ff >= 50:
        # Proxied by hard-hit / barrel above average as "handles velocity"
        contact = (row.hard_hit_pct / 40.0 + row.barrel_pct / 9.0) / 2.0
        return _clamp(0.95 + 0.08 * (contact - 1.0) + 0.03, 0.90, 1.12)
    if ff <= 30:
        # Soft-toss / junk: power hitters still OK but slight down-weight
        return 0.98
    return 1.0


def score_batter(row: BatterRow, venues: dict[str, Any]) -> Projection:
    b = _batter_factor(row)
    p = _pitcher_factor(row)
    park = _park_factor(row, venues)
    weather, wind_temp = _weather_factor(row, venues)
    rep = _repertoire_factor(row)

    # Weighted geometric blend of multipliers → effective rate multiplier
    log_blend = (
        WEIGHT_BATTER * math.log(b)
        + WEIGHT_PITCHER * math.log(p)
        + WEIGHT_PARK * math.log(park)
        + WEIGHT_WEATHER * math.log(weather)
        + WEIGHT_REPERTOIRE * math.log(rep)
    )
    multiplier = math.exp(log_blend)
    p_hr_pa = _clamp(BASE_HR_PA * multiplier, 0.010, 0.14)
    n_pa = projected_pa(row.lineup_slot)
    p_hr_game = 1.0 - (1.0 - p_hr_pa) ** n_pa
    p_hr_game = _clamp(p_hr_game, 0.04, 0.40)

    pitcher_label = row.pitcher.name
    if row.pitcher.is_opener_or_bullpen and "Bullpen" not in pitcher_label and "opener" not in pitcher_label:
        pitcher_label = f"{pitcher_label} (bullpen)"

    return Projection(
        rank=0,
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
            "n_pa": n_pa,
        },
    )


def score_slate(rows: list[BatterRow], ctx: SlateContext) -> list[Projection]:
    projections = [score_batter(r, ctx.venues) for r in rows]
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
