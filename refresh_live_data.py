"""Fetch live test features, score the supplied CatBoost model, and rebuild the existing HTML map.

Primary live sources:
  - Open-Meteo ECMWF IFS HRES (weather/rainfall/soil/runoff)
  - Open-Meteo Global Flood API (GloFAS river discharge context)
  - Official IMD district rainfall/warnings/nowcast when IMD_API_TOKEN is supplied
  - NASA POWER climatology when NASA_POWER_ENABLE=1
Supplementary adapters:
  - NASA GPM IMERG
  - NASA SMAP
  - Copernicus ERA5/CDS
  - Copernicus Sentinel-1 STAC catalogue
  - HydroSHEDS, OpenStreetMap, JRC Global Surface Water, Bhuvan/NRSC
  - NVIDIA FourCastNet

The trained 34-feature CatBoost model is never retrained or structurally changed here.
"""
from live_pipeline import refresh

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(json.dumps(refresh(force=args.force), indent=2))
