# Daily MLB Home Run Predictor

Zero-cost CLI that ranks today's top projected home-run hitters using free public data only.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python run.py
```

No API keys required. Output prints in the terminal and saves to `output/hr_picks_YYYY-MM-DD.csv`.

Each `run.py` also logs predictions under `.cache/history/` and grades prior days against box scores so the model can refine its weights over time.

## Self-learning

```bash
# Explicit grade + refit (also runs automatically at the start of run.py)
python grade.py

# Backfill history from 2022 through yesterday, then grade + refit
python backfill.py --start 2022-03-01

# Backfill a past slate, then grade it once games are final
python run.py --date 2026-09-24 --skip-learn
python grade.py --through 2026-09-24
```

Flow:

1. **Log** — every slate writes factors + `p_hr` + `player_id` / `game_pk`
2. **Grade** — finished games are joined to MLB box scores (`homeRuns`)
3. **Fit** — after ~80 labeled rows, refit factor weights + Platt calibration into [`data/learned_params.json`](data/learned_params.json)
4. **Apply** — subsequent runs load learned params (blended toward PRD priors until the sample is large)

## Data sources

| Source | Use |
|--------|-----|
| [MLB Stats API](https://statsapi.mlb.com/api/v1/) | Schedule, lineups, probable pitchers, venues, box scores |
| [Open-Meteo](https://api.open-meteo.com/) | Game-time temp, wind, humidity |
| [pybaseball](https://github.com/jldbc/pybaseball) / Baseball Savant | Season barrel / hard-hit contact baselines + rolling form |
| MLB Stats API season pitching | HR/9, IP, GS (opener / bullpen detection) |
| `data/venues.json` | Park factors, roof type, orientation |

Season metrics cache under `.cache/` and refresh automatically every 24 hours. Form windows cache under `.cache/statcast_months/` and `.cache/form/`.

## Model (summary)

Weighted blend → per-PA HR chance → game probability \(1-(1-p)^{N_{PA}}\), then optional calibration:

- 30% batter quality of contact (season; default; learned over time)
- 20% pitcher HR vulnerability (+ platoon)
- 18% park / elevation
- 13% weather (temp + wind vs park azimuth)
- 5% pitch-repertoire matchup when available
- 7% batter form (last ~35 days barrel / hard-hit / FB, empirical-Bayes shrunk to season)
- 7% pitcher form (same window: barrels allowed + HR/BBE proxy, shrunk to season)

Form uses Statcast batted balls ending the day before the slate (no leakage). Thin samples stay near 1.0; clamps keep form from dominating.