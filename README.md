# Flood Risk Watch — Final Live-API Integration

This package preserves the supplied Flood Risk Watch HTML, SVG map, colors, selectors, zone cards and risk thresholds. It replaces only the data path: live API data are fetched, mapped onto the already-trained CatBoost feature contract, scored with the supplied `flood_risk_catboost.cbm`, and written back into the same HTML data object.

## Model

The supplied model is a `CatBoostClassifier` trained on the uploaded synthetic prototype dataset. The 34-feature contract and risk thresholds are copied from the supplied model metadata.

**Risk thresholds**
- Green: `< 0.33`
- Yellow: `0.33 to < 0.66`
- Red: `>= 0.66`

## Run the website

```bash
pip install -r requirements.txt
python refresh_live_data.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The map/UI is the supplied HTML design.

## Run Streamlit alternative

```bash
streamlit run app.py
```

## Live refresh behaviour

`refresh_live_data.py` processes the 553 existing map zones in batches. It fetches:

1. Open-Meteo ECMWF live/hourly weather, rainfall, shallow soil moisture and runoff.
2. Open-Meteo Flood API / GloFAS river discharge.
3. Official IMD district rainfall/warning/nowcast when `IMD_API_TOKEN` is set.
4. NASA POWER rainfall climatology when `NASA_POWER_ENABLE=1`.
5. Optional NASA/Copernicus/Sentinel/static adapters when `ENABLE_OPTIONAL_SOURCES=1`.
6. NVIDIA FourCastNet context when `NVIDIA_API_KEY` is set.

The CatBoost model file is not modified.

## Credentials

Copy `.env.example` to `.env` and export the variables before running the refresh job.

### IMD
Register at the official IMD API Management portal and set:

```text
IMD_API_TOKEN=...
```

### NASA Earthdata / GPM / SMAP
Set the Earthdata credentials and enable:

```text
ENABLE_OPTIONAL_SOURCES=1
```

### Copernicus CDS / ERA5
Configure the CDS personal access token according to the official CDSAPI instructions, then set:

```text
ERA5_ENABLE=1
```

### NVIDIA FourCastNet
Set:

```text
NVIDIA_API_KEY=...
```

FourCastNet is used as a supplementary forecast context source. Its public inference contract does not provide a direct, location-specific `precipitation_mm` / `runoff_mm` replacement for this CatBoost model, so it is deliberately not mixed into the 34-feature score.

## Offline / no-network mode

For an offline run that keeps the supplied map data unchanged:

```bash
LIVE_OFFLINE=1 python refresh_live_data.py
```

## Important model-validity note

The supplied training dataset is explicitly labelled synthetic. The live pipeline is therefore a **live-data software integration around the existing trained prototype**, not an operational flood-warning model. Re-training/validation against observed flood labels is still required before operational use.
