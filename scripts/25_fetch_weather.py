"""Daily weather for each city, 2014-2024, from the Open-Meteo historical archive (ERA5 reanalysis; free, no key).

One point per city (config.CITY_CENTERS): daily mean temperature (C), precipitation (mm) and snowfall (cm). A few hundred KB in total.
Run from the repo root:  python scripts/25_fetch_weather.py      Writes data/raw/weather_daily.csv (city, date, temp_c, precip_mm, snow_cm).
Source: https://open-meteo.com/en/docs/historical-weather-api
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from activity_index import config as cfg  # noqa: E402

URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = {"temperature_2m_mean": "temp_c", "precipitation_sum": "precip_mm", "snowfall_sum": "snow_cm"}


def main():
    parts = []
    for city, (lat, lon) in cfg.CITY_CENTERS.items():
        r = requests.get(URL, params={"latitude": lat, "longitude": lon, "start_date": "2014-01-01", "end_date": "2024-12-31",
                                      "daily": ",".join(DAILY), "timezone": "America/New_York"}, timeout=120)
        r.raise_for_status()
        d = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date", **DAILY}).assign(city=city)
        print(f"[{city}] {len(d)} days, mean {d['temp_c'].mean():.1f} C, {d['precip_mm'].sum() / 11:.0f} mm of precipitation a year")
        parts.append(d[["city", "date", "temp_c", "precip_mm", "snow_cm"]])
        time.sleep(1)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cfg.RAW / "weather_daily.csv", index=False)
    print("wrote", cfg.RAW / "weather_daily.csv", len(out), "rows")


if __name__ == "__main__":
    main()
