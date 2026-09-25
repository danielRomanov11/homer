"""Data ingestion: MLB Stats API, Open-Meteo, and Statcast/FanGraphs baselines."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
VENUES_PATH = ROOT / "data" / "venues.json"
CACHE_META = CACHE_DIR / "statcast_meta.json"
BATTERS_CACHE = CACHE_DIR / "batters_season.parquet"
PITCHERS_CACHE = CACHE_DIR / "pitchers_season.parquet"
BULLPEN_CACHE = CACHE_DIR / "bullpen_season.parquet"

MLB_API = "https://statsapi.mlb.com/api/v1"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_HOURS = 24
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "homer-hr-predictor/1.0"})


@dataclass
class Weather:
    temp_f: float
    humidity: float
    wind_mph: float
    wind_dir_deg: float
    indoors: bool = False
    rain_risk: bool = False


@dataclass
class PitcherInfo:
    id: int | None
    name: str
    hand: str  # L / R
    is_opener_or_bullpen: bool = False
    hr9: float = 1.2
    barrel_pct: float = 7.0
    fb_pct: float = 35.0
    ff_usage: float = 40.0


@dataclass
class BatterRow:
    player_id: int
    player_name: str
    team: str
    opp: str
    bat_side: str
    lineup_slot: int
    projected_lineup: bool
    pitcher: PitcherInfo
    venue_id: int
    venue_name: str
    game_pk: int
    game_time: datetime
    weather: Weather
    rain_risk: bool = False
    barrel_pct: float = 8.0
    hard_hit_pct: float = 40.0
    fb_pct: float = 35.0
    batter_vs_ff: float = 0.0


@dataclass
class SlateContext:
    slate_date: date
    venues: dict[str, Any]
    batters_df: pd.DataFrame
    pitchers_df: pd.DataFrame
    bullpen_by_team: dict[str, dict[str, float]]
    people: dict[int, dict[str, Any]] = field(default_factory=dict)


def _get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=45)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def load_venues() -> dict[str, Any]:
    with VENUES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    # MLB season year rolls in spring; Jan–Feb belong to upcoming season year
    return today.year if today.month >= 3 else today.year - 1


def _cache_fresh() -> bool:
    if not (BATTERS_CACHE.exists() and PITCHERS_CACHE.exists() and CACHE_META.exists()):
        return False
    try:
        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(meta["fetched_at"])
        return datetime.now(timezone.utc) - fetched < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:  # noqa: BLE001
        return False


def _normalize_pct(series: pd.Series) -> pd.Series:
    """Convert 0–1 fractions to 0–100 percentages when needed."""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if s.dropna().max() <= 1.5:
        return s * 100.0
    return s


def _savant_contact_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Baseball Savant exitvelo/barrels leaderboard columns."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    id_col = next((c for c in out.columns if c.lower() in {"player_id", "id", "playerid"}), None)
    if id_col and id_col != "player_id":
        out = out.rename(columns={id_col: "player_id"})
    out["player_id"] = pd.to_numeric(out.get("player_id"), errors="coerce")

    rename: dict[str, str] = {}
    for c in out.columns:
        cl = c.lower().replace(" ", "_")
        if cl in {"brl_percent", "brl_pct"} or (cl.startswith("brl") and "pa" not in cl):
            rename[c] = "barrel_pct"
        elif cl in {"ev95percent", "hardhit_percent", "hard_hit_percent"}:
            rename[c] = "hard_hit_pct"
        elif cl in {"anglesweetspotpercent"}:
            rename[c] = "sweet_spot_pct"
        elif cl in {"avg_hit_angle"}:
            rename[c] = "avg_hit_angle"
        elif cl in {"last_name,_first_name", "last_name, first_name"}:
            rename[c] = "name_raw"
    out = out.rename(columns=rename)

    for col in ("barrel_pct", "hard_hit_pct", "sweet_spot_pct", "avg_hit_angle"):
        if col not in out.columns:
            out[col] = float("nan")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["barrel_pct"] = _normalize_pct(out["barrel_pct"])
    out["hard_hit_pct"] = _normalize_pct(out["hard_hit_pct"])
    out["sweet_spot_pct"] = _normalize_pct(out["sweet_spot_pct"])

    # Proxy FB% from launch angle (higher LA → more fly balls)
    # Map ~10° → 25% FB, ~20° → 40% FB
    angle = out["avg_hit_angle"].fillna(12.0)
    out["fb_pct"] = _clamp_series(25.0 + (angle - 10.0) * 1.5, 18.0, 55.0)
    return out


def _clamp_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def _team_id_abbrev_map(season: int) -> dict[int, str]:
    data = _get_json(f"{MLB_API}/teams", params={"sportId": 1, "season": season})
    return {
        int(t["id"]): t.get("abbreviation") or t.get("teamCode", "?").upper()
        for t in data.get("teams", [])
    }


def _mlb_pitching_season(season: int) -> pd.DataFrame:
    """Season pitching lines from MLB Stats API (HR/9, IP, GS) — no FanGraphs."""
    data = _get_json(
        f"{MLB_API}/stats",
        params={
            "stats": "season",
            "group": "pitching",
            "season": season,
            "sportIds": 1,
            "playerPool": "all",
            "limit": 2000,
        },
    )
    team_map = _team_id_abbrev_map(season)
    rows: list[dict[str, Any]] = []
    for split in (data.get("stats") or [{}])[0].get("splits", []):
        player = split.get("player") or {}
        team = split.get("team") or {}
        stat = split.get("stat") or {}
        team_id = team.get("id")
        abbrev = team.get("abbreviation") or (team_map.get(int(team_id)) if team_id else None) or "?"
        ip_raw = stat.get("inningsPitched", "0")
        try:
            # MLB uses "123.1" / "123.2" outs notation
            ip_s = str(ip_raw)
            if "." in ip_s:
                whole, frac = ip_s.split(".", 1)
                ip = float(whole) + (int(frac) / 3.0)
            else:
                ip = float(ip_s)
        except ValueError:
            ip = 0.0
        rows.append(
            {
                "player_id": int(player["id"]) if player.get("id") else None,
                "name": player.get("fullName", ""),
                "team": abbrev,
                "hr9": float(stat.get("homeRunsPer9") or 0) if stat.get("homeRunsPer9") not in (None, "") else float("nan"),
                "gs": float(stat.get("gamesStarted") or 0),
                "ip": ip,
                "ff_usage": 40.0,  # repertoire detail not in this endpoint; model treats as neutral band
            }
        )
    return pd.DataFrame(rows)


def refresh_statcast_cache(season: int | None = None, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Load or refresh season batter/pitcher baselines (24h TTL)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    season = season or current_season()

    if not force and _cache_fresh():
        batters = pd.read_parquet(BATTERS_CACHE)
        pitchers = pd.read_parquet(PITCHERS_CACHE)
        bullpen = _bullpen_from_pitchers(pitchers)
        return batters, pitchers, bullpen

    from pybaseball import statcast_batter_exitvelo_barrels, statcast_pitcher_exitvelo_barrels

    batters = _savant_contact_frame(statcast_batter_exitvelo_barrels(season, minBBE=25))
    savant_p = _savant_contact_frame(statcast_pitcher_exitvelo_barrels(season, minBBE=25))
    savant_p = savant_p.rename(
        columns={
            "barrel_pct": "barrel_pct",
            "fb_pct": "fb_pct",
            "hard_hit_pct": "hard_hit_allowed_pct",
        }
    )[["player_id", "barrel_pct", "fb_pct", "hard_hit_allowed_pct"]]

    mlb_p = _mlb_pitching_season(season)
    pitchers = mlb_p.merge(savant_p, on="player_id", how="left", suffixes=("", "_savant"))
    if "barrel_pct" not in pitchers.columns:
        pitchers["barrel_pct"] = float("nan")
    if "fb_pct" not in pitchers.columns:
        pitchers["fb_pct"] = float("nan")
    pitchers["barrel_pct"] = pitchers["barrel_pct"].fillna(7.0)
    pitchers["fb_pct"] = pitchers["fb_pct"].fillna(35.0)
    pitchers["ff_usage"] = pitchers["ff_usage"].fillna(40.0)

    batters.to_parquet(BATTERS_CACHE, index=False)
    pitchers.to_parquet(PITCHERS_CACHE, index=False)
    bullpen = _bullpen_from_pitchers(pitchers)
    pd.DataFrame([{"team": k, **v} for k, v in bullpen.items()]).to_parquet(BULLPEN_CACHE, index=False)

    CACHE_META.write_text(
        json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "season": season}),
        encoding="utf-8",
    )
    return batters, pitchers, bullpen


def _bullpen_from_pitchers(pitchers: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Team bullpen proxy = average of pitchers with few starts or short outings."""
    df = pitchers.copy()
    if "gs" in df.columns and "ip" in df.columns:
        ip_per_gs = df["ip"] / df["gs"].clip(lower=1)
        pens = df[(df["gs"].fillna(0) <= 2) | (ip_per_gs < 2.0)]
    else:
        pens = df
    out: dict[str, dict[str, float]] = {}
    if "team" not in pens.columns or pens.empty:
        return out
    for team, grp in pens.groupby(pens["team"].astype(str).str.upper()):
        if team in {"- - -", "---", "2TEAMS", "NAN", "?"}:
            continue
        out[team] = {
            "hr9": float(grp["hr9"].mean()) if grp["hr9"].notna().any() else 1.25,
            "barrel_pct": float(grp["barrel_pct"].mean()) if grp["barrel_pct"].notna().any() else 7.5,
            "fb_pct": float(grp["fb_pct"].mean()) if grp["fb_pct"].notna().any() else 35.0,
            "ff_usage": float(grp["ff_usage"].mean()) if grp["ff_usage"].notna().any() else 40.0,
        }
    return out


def fetch_schedule(slate_date: date) -> list[dict[str, Any]]:
    data = _get_json(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "date": slate_date.isoformat(),
            "hydrate": "lineups,probablePitcher(note),venue,weather,team",
        },
    )
    games: list[dict[str, Any]] = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return games


def fetch_people(person_ids: list[int]) -> dict[int, dict[str, Any]]:
    people: dict[int, dict[str, Any]] = {}
    uniq = sorted({int(i) for i in person_ids if i})
    for i in range(0, len(uniq), 50):
        chunk = uniq[i : i + 50]
        data = _get_json(
            f"{MLB_API}/people",
            params={"personIds": ",".join(str(x) for x in chunk)},
        )
        for p in data.get("people", []):
            people[int(p["id"])] = p
    return people


def fetch_weather(
    lat: float,
    lon: float,
    game_time: datetime,
    indoors: bool,
) -> Weather:
    if indoors:
        return Weather(temp_f=72.0, humidity=45.0, wind_mph=0.0, wind_dir_deg=0.0, indoors=True)

    data = _get_json(
        OPEN_METEO,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
            "forecast_days": 3,
        },
    )
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return Weather(temp_f=72.0, humidity=50.0, wind_mph=5.0, wind_dir_deg=0.0)

    # Match closest local hour string
    target = game_time.astimezone().replace(tzinfo=None) if game_time.tzinfo else game_time
    target_str = target.strftime("%Y-%m-%dT%H:00")
    if target_str in times:
        idx = times.index(target_str)
    else:
        # nearest by parsing
        best_i, best_diff = 0, float("inf")
        for i, t in enumerate(times):
            try:
                dt = datetime.fromisoformat(t)
            except ValueError:
                continue
            diff = abs((dt - target.replace(tzinfo=None)).total_seconds())
            if diff < best_diff:
                best_diff, best_i = diff, i
        idx = best_i

    precip = hourly.get("precipitation_probability", [0] * len(times))
    rain = False
    try:
        rain = float(precip[idx] or 0) >= 60
    except (TypeError, ValueError, IndexError):
        rain = False

    def _at(key: str, default: float) -> float:
        vals = hourly.get(key, [])
        try:
            return float(vals[idx])
        except (TypeError, ValueError, IndexError):
            return default

    return Weather(
        temp_f=_at("temperature_2m", 72.0),
        humidity=_at("relative_humidity_2m", 50.0),
        wind_mph=_at("wind_speed_10m", 5.0),
        wind_dir_deg=_at("wind_direction_10m", 0.0),
        indoors=False,
        rain_risk=rain,
    )


def _team_abbrev(team_obj: dict) -> str:
    return (
        team_obj.get("abbreviation")
        or team_obj.get("teamName")
        or team_obj.get("name", "?")[:3].upper()
    )


def _parse_game_time(game: dict) -> datetime:
    raw = game.get("gameDate") or ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _lineup_players(game: dict, side: str) -> tuple[list[dict], bool]:
    lineups = game.get("lineups") or {}
    key = "homePlayers" if side == "home" else "awayPlayers"
    players = lineups.get(key) or []
    if players:
        return players[:9], False
    return [], True


def _fetch_prior_lineup(team_id: int, before: date) -> list[dict]:
    """Fallback: batting order from most recent completed game."""
    start = (before - timedelta(days=10)).isoformat()
    end = before.isoformat()
    data = _get_json(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start,
            "endDate": end,
            "hydrate": "lineups",
        },
    )
    games = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    games.sort(key=lambda g: g.get("gameDate", ""), reverse=True)
    for g in games:
        status = (g.get("status") or {}).get("abstractGameState", "")
        if status not in {"Final", "Live"} and not g.get("lineups"):
            continue
        home_id = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
        away_id = g.get("teams", {}).get("away", {}).get("team", {}).get("id")
        side = "home" if home_id == team_id else "away" if away_id == team_id else None
        if not side:
            continue
        players, missing = _lineup_players(g, side)
        if not missing and players:
            return players[:9]
    return []


def _fetch_roster_top9(team_id: int) -> list[dict]:
    data = _get_json(
        f"{MLB_API}/teams/{team_id}/roster",
        params={"rosterType": "active"},
    )
    position_priority = {
        "DH": 0,
        "C": 1,
        "1B": 2,
        "2B": 3,
        "SS": 4,
        "3B": 5,
        "LF": 6,
        "CF": 7,
        "RF": 8,
        "OF": 7,
        "IF": 4,
        "UT": 9,
    }
    hitters = []
    for entry in data.get("roster", []):
        person = entry.get("person") or {}
        pos = (entry.get("position") or {}).get("abbreviation", "")
        if pos in {"P", "TWP"} and (entry.get("position") or {}).get("type") == "Pitcher":
            # Two-way players can still hit; include TWP lightly
            if pos == "P":
                continue
        hitters.append(
            {
                "id": person.get("id"),
                "fullName": person.get("fullName", "Unknown"),
                "primaryPosition": entry.get("position") or {},
                "_prio": position_priority.get(pos, 20),
            }
        )
    hitters.sort(key=lambda x: x["_prio"])
    return hitters[:9]


def resolve_lineup(game: dict, side: str, slate_date: date) -> tuple[list[dict], bool]:
    players, projected = _lineup_players(game, side)
    if players:
        return players, False
    team = game.get("teams", {}).get(side, {}).get("team", {})
    team_id = team.get("id")
    if not team_id:
        return [], True
    prior = _fetch_prior_lineup(int(team_id), slate_date)
    if prior:
        return prior, True
    roster = _fetch_roster_top9(int(team_id))
    return roster, True


def _pitcher_from_stats(
    pitcher_id: int | None,
    name: str,
    hand: str,
    pitchers_df: pd.DataFrame,
    bullpen: dict[str, dict[str, float]],
    opp_team: str,
) -> PitcherInfo:
    info = PitcherInfo(id=pitcher_id, name=name or "TBD", hand=hand or "R")
    row = None
    if pitcher_id is not None and "player_id" in pitchers_df.columns:
        match = pitchers_df[pitchers_df["player_id"] == pitcher_id]
        if not match.empty:
            row = match.iloc[0]
    if row is None and name and "name" in pitchers_df.columns:
        match = pitchers_df[pitchers_df["name"].str.lower() == name.lower()]
        if not match.empty:
            row = match.iloc[0]

    use_bullpen = pitcher_id is None or name in {"", "TBD"}
    if row is not None:
        ip = float(row["ip"]) if pd.notna(row.get("ip")) else 0.0
        gs = float(row["gs"]) if pd.notna(row.get("gs")) else 0.0
        ip_per_gs = (ip / gs) if gs > 0 else 0.0
        if gs > 0 and ip_per_gs < 2.0:
            use_bullpen = True
        else:
            info.hr9 = float(row["hr9"]) if pd.notna(row.get("hr9")) else 1.2
            info.barrel_pct = float(row["barrel_pct"]) if pd.notna(row.get("barrel_pct")) else 7.0
            info.fb_pct = float(row["fb_pct"]) if pd.notna(row.get("fb_pct")) else 35.0
            info.ff_usage = float(row["ff_usage"]) if pd.notna(row.get("ff_usage")) else 40.0
            info.is_opener_or_bullpen = False
            return info

    if use_bullpen:
        pen = bullpen.get(opp_team.upper()) or bullpen.get(opp_team) or {
            "hr9": 1.25,
            "barrel_pct": 7.5,
            "fb_pct": 36.0,
            "ff_usage": 40.0,
        }
        info.hr9 = pen["hr9"]
        info.barrel_pct = pen["barrel_pct"]
        info.fb_pct = pen["fb_pct"]
        info.ff_usage = pen.get("ff_usage", 40.0)
        info.is_opener_or_bullpen = True
        if info.name in {"", "TBD"} or pitcher_id is None:
            info.name = f"{opp_team} Bullpen"
        else:
            info.name = f"{info.name} (opener)"
    return info


def _batter_metrics(player_id: int, batters_df: pd.DataFrame) -> dict[str, float]:
    defaults = {"barrel_pct": 8.0, "hard_hit_pct": 40.0, "fb_pct": 35.0, "batter_vs_ff": 0.0}
    if batters_df.empty or "player_id" not in batters_df.columns:
        return defaults
    match = batters_df[batters_df["player_id"] == player_id]
    if match.empty:
        return defaults
    row = match.iloc[0]
    out = dict(defaults)
    for k in ("barrel_pct", "hard_hit_pct", "fb_pct"):
        if k in row and pd.notna(row[k]):
            out[k] = float(row[k])
    return out


def build_batter_slate(slate_date: date | None = None) -> tuple[list[BatterRow], SlateContext]:
    slate_date = slate_date or date.today()
    venues = load_venues()
    batters_df, pitchers_df, bullpen = refresh_statcast_cache(current_season(slate_date))
    games = fetch_schedule(slate_date)

    # Collect ids for people hydrate
    person_ids: list[int] = []
    resolved_lineups: dict[tuple[int, str], tuple[list[dict], bool]] = {}
    for g in games:
        for side in ("home", "away"):
            players, projected = resolve_lineup(g, side, slate_date)
            resolved_lineups[(g["gamePk"], side)] = (players, projected)
            for p in players:
                if p.get("id"):
                    person_ids.append(int(p["id"]))
            pp = g.get("teams", {}).get(side, {}).get("probablePitcher") or {}
            if pp.get("id"):
                person_ids.append(int(pp["id"]))

    people = fetch_people(person_ids)
    rows: list[BatterRow] = []

    for g in games:
        venue = g.get("venue") or {}
        venue_id = int(venue.get("id") or 0)
        vmeta = venues.get(str(venue_id), {})
        venue_name = vmeta.get("name") or venue.get("name") or "Unknown"
        roof = vmeta.get("roof", "open")
        indoors = roof == "dome"
        lat = vmeta.get("lat")
        lon = vmeta.get("lon")
        game_time = _parse_game_time(g)

        # Optional API weather hint for rain
        api_weather = g.get("weather") or {}
        rain_hint = "rain" in str(api_weather.get("condition", "")).lower()

        if lat is None or lon is None:
            # hydrate venue location
            try:
                vdata = _get_json(f"{MLB_API}/venues/{venue_id}", params={"hydrate": "location"})
                loc = (vdata.get("venues") or [{}])[0].get("location") or {}
                coords = loc.get("defaultCoordinates") or {}
                lat = coords.get("latitude", 39.0)
                lon = coords.get("longitude", -98.0)
            except Exception:  # noqa: BLE001
                lat, lon = 39.0, -98.0

        weather = fetch_weather(float(lat), float(lon), game_time, indoors=indoors)
        weather.rain_risk = weather.rain_risk or rain_hint

        home = g.get("teams", {}).get("home", {})
        away = g.get("teams", {}).get("away", {})
        home_team = _team_abbrev(home.get("team") or {})
        away_team = _team_abbrev(away.get("team") or {})

        # Probable pitchers: batter faces the OPPOSING starter
        for side, team_abbr, opp_abbr, opp_side in (
            ("home", home_team, away_team, "away"),
            ("away", away_team, home_team, "home"),
        ):
            players, projected = resolved_lineups.get((g["gamePk"], side), ([], True))
            opp_pp = (g.get("teams", {}).get(opp_side, {}) or {}).get("probablePitcher") or {}
            pid = opp_pp.get("id")
            pname = opp_pp.get("fullName") or "TBD"
            hand = "R"
            if pid and int(pid) in people:
                hand = ((people[int(pid)].get("pitchHand") or {}).get("code")) or "R"

            pitcher = _pitcher_from_stats(
                int(pid) if pid else None,
                pname,
                hand,
                pitchers_df,
                bullpen,
                opp_team=opp_abbr,
            )

            for slot, player in enumerate(players, start=1):
                batter_id = int(player.get("id") or 0)
                if not batter_id:
                    continue
                bat_side = "R"
                if batter_id in people:
                    bat_side = ((people[batter_id].get("batSide") or {}).get("code")) or "R"
                # Switch hitters: assume advantage vs pitcher hand
                if bat_side == "S":
                    bat_side = "L" if pitcher.hand == "R" else "R"

                metrics = _batter_metrics(batter_id, batters_df)
                rows.append(
                    BatterRow(
                        player_id=batter_id,
                        player_name=player.get("fullName") or people.get(batter_id, {}).get("fullName", "Unknown"),
                        team=team_abbr,
                        opp=opp_abbr,
                        bat_side=bat_side,
                        lineup_slot=slot,
                        projected_lineup=projected,
                        pitcher=pitcher,
                        venue_id=venue_id,
                        venue_name=venue_name,
                        game_pk=int(g["gamePk"]),
                        game_time=game_time,
                        weather=weather,
                        rain_risk=weather.rain_risk,
                        barrel_pct=metrics["barrel_pct"],
                        hard_hit_pct=metrics["hard_hit_pct"],
                        fb_pct=metrics["fb_pct"],
                        batter_vs_ff=metrics["batter_vs_ff"],
                    )
                )

    ctx = SlateContext(
        slate_date=slate_date,
        venues=venues,
        batters_df=batters_df,
        pitchers_df=pitchers_df,
        bullpen_by_team=bullpen,
        people=people,
    )
    return rows, ctx
