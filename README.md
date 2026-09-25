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

No API keys. No arguments. Output prints in the terminal and saves to `output/hr_picks_YYYY-MM-DD.csv`.

## Data sources

| Source | Use |
|--------|-----|
| [MLB Stats API](https://statsapi.mlb.com/api/v1/) | Schedule, lineups, probable pitchers, venues |
| [Open-Meteo](https://api.open-meteo.com/) | Game-time temp, wind, humidity |
| [pybaseball](https://github.com/jldbc/pybaseball) / Baseball Savant | Season barrel / hard-hit contact baselines |
| MLB Stats API season pitching | HR/9, IP, GS (opener / bullpen detection) |
| `data/venues.json` | Park factors, roof type, orientation |

Season metrics cache under `.cache/` and refresh automatically every 24 hours.

## Model (summary)

Weighted blend → per-PA HR chance → game probability \(1-(1-p)^{N_{PA}}\):

- 35% batter quality of contact
- 25% pitcher HR vulnerability (+ platoon)
- 20% park / elevation
- 15% weather (temp + wind vs park azimuth)
- 5% pitch-repertoire matchup when available
