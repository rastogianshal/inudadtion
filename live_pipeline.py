import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()
from catboost import CatBoostClassifier
from optional_sources import build_optional_status

BASE = Path(__file__).resolve().parent
TEMPLATE_HTML = BASE / "flood_risk_watch_template.html"
OUTPUT_HTML = BASE / "flood_risk_watch.html"
LIVE_JSON = BASE / "live_flood_data.json"
SOURCE_STATUS_JSON = BASE / "live_source_status.json"
MODEL_FILE = BASE / "flood_risk_catboost.cbm"
DATA_FILE = BASE / "assam_bihar_odisha_500k_weather_flood_dataset.csv"

FEATURES = [
    "state","district","latitude","longitude","air_temperature_2m_C",
    "dewpoint_temperature_2m_C","relative_humidity_pct","surface_pressure_hPa",
    "wind_u_10m_ms","wind_v_10m_ms","wind_speed_10m_ms","precipitation_mm",
    "rain_3h_mm","rain_6h_mm","rain_12h_mm","rain_24h_mm","rain_72h_mm",
    "rain_7d_mm","rainfall_anomaly_7d","soil_moisture_0_10",
    "soil_moisture_anomaly","elevation_m","slope_deg","flow_accumulation",
    "hand_m","distance_to_river_m","drainage_density_km_per_km2",
    "catchment_area_km2","builtup_fraction","impervious_fraction",
    "road_density","building_density","surface_water_occurrence","runoff_mm"
]
CAT_FEATURES = ["state", "district"]

OPEN_METEO_ECMWF = "https://api.open-meteo.com/v1/ecmwf"
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"
NASA_POWER_CLIM = "https://power.larc.nasa.gov/api/temporal/climatology/point"
IMD_BASE = "https://api.imd.gov.in/api/v1"
NVIDIA_FOURCASTNET = "https://climate.api.nvidia.com/v1/nvidia/fourcastnet"

REQUEST_TIMEOUT = int(os.getenv("LIVE_API_TIMEOUT", "35"))
BATCH_SIZE = int(os.getenv("OPEN_METEO_BATCH_SIZE", "50"))


def tier(p: float) -> str:
    if p < 0.33:
        return "green"
    if p < 0.66:
        return "yellow"
    return "red"


def norm_name(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_flood_data(html: str) -> Dict[str, Any]:
    m = re.search(r"const FLOOD_DATA = (.*?);\nconst FEATURE_IMPORTANCE", html, flags=re.S)
    if not m:
        raise RuntimeError("Could not locate FLOOD_DATA in HTML template.")
    return json.loads(m.group(1))


def replace_flood_data(html: str, payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    pattern = r"const FLOOD_DATA = (.*?);\nconst FEATURE_IMPORTANCE"
    replacement = "const FLOOD_DATA = " + blob + ";\nconst FEATURE_IMPORTANCE"
    out, n = re.subn(pattern, replacement, html, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("Could not replace FLOOD_DATA in HTML template.")
    return out


def flatten_zones(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for state, districts in payload.get("states", {}).items():
        for d in districts:
            for z in d.get("zones", []):
                rows.append({
                    "state": state,
                    "district": d["district"],
                    "zone_id": int(z["id"]),
                    "lat": float(z["lat"]),
                    "lon": float(z["lon"]),
                    "zone": z,
                    "district_obj": d,
                })
    return rows


def batch(items: List[Dict[str, Any]], n: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), n):
        yield items[i:i+n]


def fetch_open_meteo_batch(items: List[Dict[str, Any]], session: requests.Session) -> Dict[Tuple[str,int], Dict[str, Any]]:
    if not items:
        return {}
    lats = ",".join(f"{x['lat']:.5f}" for x in items)
    lons = ",".join(f"{x['lon']:.5f}" for x in items)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": ",".join([
            "temperature_2m", "dew_point_2m", "relative_humidity_2m",
            "surface_pressure", "wind_speed_10m", "wind_direction_10m",
            "precipitation", "runoff", "soil_moisture_0_to_7cm"
        ]),
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    r = session.get(OPEN_METEO_ECMWF, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]
    out: Dict[Tuple[str,int], Dict[str, Any]] = {}
    for meta, obj in zip(items, data):
        hourly = obj.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            continue
        target = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        # Open-Meteo may lag a little; select closest available hourly time.
        idx = min(range(len(times)), key=lambda i: abs(pd.Timestamp(times[i], tz="UTC").to_pydatetime() - target))

        def val(name: str) -> Optional[float]:
            vals = hourly.get(name, [])
            return None if idx >= len(vals) else vals[idx]

        def series(name: str) -> List[Optional[float]]:
            return hourly.get(name, []) or []

        precip = series("precipitation")
        soil = series("soil_moisture_0_to_7cm")
        runoff = series("runoff")

        def rolling_sum(hours: int) -> float:
            start = max(0, idx - hours + 1)
            return float(sum(v for v in precip[start:idx+1] if v is not None))

        soil_hist = [v for v in soil[max(0, idx - 7*24 + 1):idx+1] if v is not None]
        soil_now = val("soil_moisture_0_to_7cm")
        soil_mean = float(np.mean(soil_hist)) if soil_hist else float(soil_now or 0.0)
        soil_anom = ((float(soil_now) - soil_mean) / max(abs(soil_mean), 0.05)) if soil_now is not None else 0.0

        wind_speed = float(val("wind_speed_10m") or 0.0)
        wind_dir = float(val("wind_direction_10m") or 0.0)
        wind_u = -wind_speed * math.sin(math.radians(wind_dir))
        wind_v = -wind_speed * math.cos(math.radians(wind_dir))

        out[(meta["district"], meta["zone_id"])] = {
            "air_temperature_2m_C": val("temperature_2m"),
            "dewpoint_temperature_2m_C": val("dew_point_2m"),
            "relative_humidity_pct": val("relative_humidity_2m"),
            "surface_pressure_hPa": val("surface_pressure"),
            "wind_u_10m_ms": wind_u,
            "wind_v_10m_ms": wind_v,
            "wind_speed_10m_ms": wind_speed,
            "precipitation_mm": float(val("precipitation") or 0.0),
            "rain_3h_mm": rolling_sum(3),
            "rain_6h_mm": rolling_sum(6),
            "rain_12h_mm": rolling_sum(12),
            "rain_24h_mm": rolling_sum(24),
            "rain_72h_mm": rolling_sum(72),
            "rain_7d_mm": rolling_sum(7*24),
            "soil_moisture_0_10": soil_now,
            "soil_moisture_anomaly": soil_anom,
            "runoff_mm": float(val("runoff") or 0.0),
            "source": "Open-Meteo ECMWF IFS HRES",
        }
    return out


def fetch_open_meteo_flood_batch(items: List[Dict[str, Any]], session: requests.Session) -> Dict[Tuple[str,int], Dict[str, Any]]:
    if not items:
        return {}
    lats = ",".join(f"{x['lat']:.5f}" for x in items)
    lons = ",".join(f"{x['lon']:.5f}" for x in items)
    params = {
        "latitude": lats,
        "longitude": lons,
        "daily": "river_discharge",
        "past_days": 7,
        "forecast_days": 3,
        "timezone": "UTC",
    }
    r = session.get(OPEN_METEO_FLOOD, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]
    out: Dict[Tuple[str,int], Dict[str, Any]] = {}
    for meta, obj in zip(items, data):
        vals = [v for v in (obj.get("daily", {}).get("river_discharge") or []) if v is not None]
        current = float(vals[-1]) if vals else None
        trend = "stable"
        if len(vals) >= 3:
            delta = vals[-1] - vals[-3]
            if delta > max(0.10 * abs(vals[-3]), 1.0):
                trend = "rising"
            elif delta < -max(0.10 * abs(vals[-3]), 1.0):
                trend = "falling"
        out[(meta["district"], meta["zone_id"])] = {
            "river_discharge_m3s": current,
            "river_discharge_trend": trend,
            "source": "Open-Meteo Global Flood API (GloFAS v4)",
        }
    return out


def imd_headers() -> Dict[str, str]:
    token = os.getenv("IMD_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def imd_get(path: str, session: requests.Session) -> Optional[Any]:
    try:
        r = session.get(IMD_BASE + path, headers=imd_headers(), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def list_like_json(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "result", "results"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
        return [obj]
    return []


def fetch_imd_context(session: requests.Session) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    status = {"enabled": bool(os.getenv("IMD_API_TOKEN")), "available": False, "details": []}
    if not os.getenv("IMD_API_TOKEN"):
        status["details"].append("IMD_API_TOKEN not set; official IMD calls skipped.")
        return {}, status

    rainfall = list_like_json(imd_get("/districtrainfall", session))
    warnings = list_like_json(imd_get("/districtwarning", session))
    nowcasts = list_like_json(imd_get("/districtnowcast", session))
    context: Dict[str, Dict[str, Any]] = {}

    for row in rainfall:
        name = norm_name(row.get("District"))
        if not name:
            continue
        dep = row.get("Weekly Departure Per")
        try:
            dep_val = float(str(dep).replace("%", "").strip()) / 100.0
        except Exception:
            dep_val = None
        daily_actual = pd.to_numeric(str(row.get("Daily Actual", "nan")), errors="coerce")
        weekly_actual = pd.to_numeric(str(row.get("Weekly Actual", "nan")), errors="coerce")
        context.setdefault(name, {})
        if pd.notna(daily_actual):
            context[name]["imd_rainfall_today_mm"] = float(daily_actual)
        if pd.notna(weekly_actual):
            context[name]["imd_rainfall_week_mm"] = float(weekly_actual)
        if dep_val is not None:
            context[name]["rainfall_anomaly_7d"] = dep_val

    for row in warnings:
        name = norm_name(row.get("District"))
        if not name:
            continue
        codes = [str(row.get(k)) for k in ("Day_1", "Day_2", "Day_3") if row.get(k) not in (None, "")]
        colors = [str(row.get(k)) for k in ("Day1_Color", "Day2_Color", "Day3_Color") if row.get(k) not in (None, "")]
        context.setdefault(name, {})
        context[name]["imd_warning"] = ", ".join(codes) if codes else "No warning"
        if colors:
            context[name]["imd_warning_color"] = colors[0]

    for row in nowcasts:
        name = norm_name(row.get("Station") or row.get("District") or row.get("Station_Name"))
        if not name:
            continue
        context.setdefault(name, {})
        if row.get("message"):
            context[name]["imd_nowcast"] = row.get("message")
        if row.get("color") is not None:
            context[name]["imd_nowcast_color"] = row.get("color")

    status["available"] = bool(context)
    status["details"].append(f"district rainfall rows={len(rainfall)}")
    status["details"].append(f"district warning rows={len(warnings)}")
    status["details"].append(f"district nowcast rows={len(nowcasts)}")
    return context, status


def nasa_power_climatology(lat: float, lon: float, session: requests.Session) -> Optional[float]:
    try:
        params = {
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "format": "JSON",
        }
        r = session.get(NASA_POWER_CLIM, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        vals = data.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
        if not vals:
            return None
        month = datetime.now(timezone.utc).month
        candidates = []
        for key, value in vals.items():
            if str(key).startswith(str(month).zfill(2)):
                try:
                    candidates.append(float(value))
                except Exception:
                    pass
        if candidates:
            # Approximate weekly climatology from daily/monthly POWER climatology representation.
            return float(np.nanmean(candidates) * 7.0)
        numeric = [float(v) for v in vals.values() if isinstance(v, (int, float))]
        return float(np.nanmean(numeric) * 7.0) if numeric else None
    except Exception:
        return None


def district_centroids(payload: Dict[str, Any]) -> Dict[Tuple[str, str], Tuple[float, float]]:
    out = {}
    for state, districts in payload.get("states", {}).items():
        for d in districts:
            out[(state, d["district"])] = (float(d["lat"]), float(d["lon"]))
    return out


def district_static_defaults() -> Dict[Tuple[str, str], Dict[str, float]]:
    if not DATA_FILE.exists():
        return {}
    usecols = [
        "state", "district", "road_density", "building_density", "elevation_m",
        "slope_deg", "flow_accumulation", "hand_m", "distance_to_river_m",
        "drainage_density_km_per_km2", "catchment_area_km2", "builtup_fraction",
        "impervious_fraction", "surface_water_occurrence"
    ]
    df = pd.read_csv(DATA_FILE, usecols=usecols)
    grouped = df.groupby(["state", "district"])[usecols[2:]].median(numeric_only=True)
    return {idx: row.to_dict() for idx, row in grouped.iterrows()}


def build_feature_row(meta: Dict[str, Any], live: Dict[str, Any], imd: Dict[str, Any], static_defaults: Dict[str, float], power_normal: Optional[float]) -> Dict[str, Any]:
    z = meta["zone"]
    row: Dict[str, Any] = {
        "state": meta["state"],
        "district": meta["district"],
        "latitude": meta["lat"],
        "longitude": meta["lon"],
    }
    # Static geospatial features remain in the same representation used by training.
    for k in [
        "elevation_m", "slope_deg", "flow_accumulation", "hand_m", "distance_to_river_m",
        "drainage_density_km_per_km2", "catchment_area_km2", "builtup_fraction",
        "impervious_fraction", "surface_water_occurrence"
    ]:
        value = z.get(k)
        if value is None:
            value = static_defaults.get(k, 0.0)
        row[k] = float(value)
    row["road_density"] = float(z.get("road_density", static_defaults.get("road_density", 0.0)))
    row["building_density"] = float(z.get("building_density", static_defaults.get("building_density", 0.0)))

    for k in [
        "air_temperature_2m_C", "dewpoint_temperature_2m_C", "relative_humidity_pct",
        "surface_pressure_hPa", "wind_u_10m_ms", "wind_v_10m_ms", "wind_speed_10m_ms",
        "precipitation_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm",
        "rain_72h_mm", "rain_7d_mm", "soil_moisture_0_10", "soil_moisture_anomaly", "runoff_mm"
    ]:
        row[k] = live.get(k)

    row["rainfall_anomaly_7d"] = imd.get("rainfall_anomaly_7d")
    if row["rainfall_anomaly_7d"] is None and power_normal is not None:
        row["rainfall_anomaly_7d"] = (float(row["rain_7d_mm"] or 0.0) - power_normal) / max(power_normal, 1.0)
    if row["rainfall_anomaly_7d"] is None:
        # Last-resort deterministic fallback consistent with the supplied refresh logic.
        baseline_7d = float(os.getenv("FALLBACK_7D_RAIN_BASELINE_MM", "120.0"))
        row["rainfall_anomaly_7d"] = (float(row["rain_7d_mm"] or 0.0) - baseline_7d) / baseline_7d

    # Keep model input complete even if an API omits a value.
    for k in FEATURES:
        if k not in row or row[k] is None or (isinstance(row[k], float) and math.isnan(row[k])):
            if k in static_defaults:
                row[k] = float(static_defaults[k])
            elif k in ("soil_moisture_anomaly", "rainfall_anomaly_7d"):
                row[k] = 0.0
            else:
                row[k] = 0.0
    return row


def load_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(MODEL_FILE))
    return model


def score_zones(payload: Dict[str, Any], live_by_zone: Dict[Tuple[str, int], Dict[str, Any]], flood_by_zone: Dict[Tuple[str, int], Dict[str, Any]], imd_ctx: Dict[str, Dict[str, Any]], power_by_district: Dict[Tuple[str, str], Optional[float]], static_defaults: Dict[Tuple[str, str], Dict[str, float]], model: CatBoostClassifier) -> Dict[str, Any]:
    for meta in flatten_zones(payload):
        key = (meta["district"], meta["zone_id"])
        live = live_by_zone.get(key, {})
        flood = flood_by_zone.get(key, {})
        dctx = imd_ctx.get(norm_name(meta["district"]), {})
        defaults = static_defaults.get((meta["state"], meta["district"]), {})
        z = meta["zone"]
        # If the live weather provider is unavailable for a zone, keep the last displayed
        # prediction instead of pretending zeros are observations.
        if live:
            row = build_feature_row(meta, live, dctx, defaults, power_by_district.get((meta["state"], meta["district"])))
            frame = pd.DataFrame([row], columns=FEATURES)
            frame["state"] = frame["state"].astype(str)
            frame["district"] = frame["district"].astype(str)
            p = float(model.predict_proba(frame)[:, 1][0])
            z["p"] = round(p, 4)
            z["t"] = tier(p)
            for k in ["air_temperature_2m_C", "dewpoint_temperature_2m_C", "relative_humidity_pct", "surface_pressure_hPa", "wind_speed_10m_ms", "precipitation_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm", "rain_72h_mm", "rain_7d_mm", "rainfall_anomaly_7d", "soil_moisture_0_10", "soil_moisture_anomaly", "runoff_mm"]:
                if k in row:
                    v = row[k]
                    z[k] = round(float(v), 4) if isinstance(v, (int, float, np.number)) else v
        else:
            z["live_refresh_state"] = "stale_no_live_weather"
        z.update({k: v for k, v in flood.items() if v is not None and k != "source"})
        if dctx.get("imd_rainfall_today_mm") is not None:
            z["imd_rainfall_today_mm"] = round(float(dctx["imd_rainfall_today_mm"]), 2)
        if dctx.get("imd_warning"):
            z["imd_warning"] = dctx["imd_warning"]
        if power_by_district.get((meta["state"], meta["district"])) is not None:
            z["rainfall_normal_7d_mm"] = round(float(power_by_district[(meta["state"], meta["district"])]), 2)
        z["live_data_timestamp"] = datetime.now(timezone.utc).isoformat()
        z["live_weather_source"] = live.get("source", "fallback/static")
        z["live_hydrology_source"] = flood.get("source", "not available")

    for state, districts in payload.get("states", {}).items():
        for d in districts:
            probs = [float(z.get("p", 0.0)) for z in d.get("zones", [])]
            d["avgProb"] = round(float(np.mean(probs)) if probs else 0.0, 4)
            d["avgTier"] = tier(d["avgProb"])
            ctx = imd_ctx.get(norm_name(d["district"]), {})
            if ctx.get("imd_rainfall_today_mm") is not None:
                d["imd_rainfall_today_mm"] = round(float(ctx["imd_rainfall_today_mm"]), 2)
            if ctx.get("imd_warning"):
                d["imd_warning"] = ctx["imd_warning"]
    payload["generated"] = True
    payload["live"] = True
    payload["lastLiveRefresh"] = datetime.now(timezone.utc).isoformat()
    return payload


def fetch_fourcastnet_status(session: requests.Session) -> Dict[str, Any]:
    token = os.getenv("NVIDIA_API_KEY", "").strip()
    if not token:
        return {"enabled": False, "available": False, "details": ["NVIDIA_API_KEY not set; FourCastNet skipped."]}
    body = {
        "input_id": int(os.getenv("FOURCASTNET_INPUT_ID", "0")),
        "variables": os.getenv("FOURCASTNET_VARIABLES", "w10m,t2m,msl,tcwv,z500"),
        "simulation_length": int(os.getenv("FOURCASTNET_STEPS", "4")),
        "ensemble_size": int(os.getenv("FOURCASTNET_ENSEMBLE", "1")),
        "noise_amplitude": float(os.getenv("FOURCASTNET_NOISE", "0")),
    }
    try:
        r = session.post(
            NVIDIA_FOURCASTNET,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return {"enabled": True, "available": True, "details": ["FourCastNet inference completed for configured built-in input."]}
    except Exception as exc:
        return {"enabled": True, "available": False, "details": [f"FourCastNet request failed: {exc}"]}


def refresh(force: bool = False) -> Dict[str, Any]:
    html = TEMPLATE_HTML.read_text(encoding="utf-8")
    payload = extract_flood_data(html)
    session = requests.Session()
    session.headers.update({"User-Agent": "FloodRiskWatch/1.0"})

    zones = flatten_zones(payload)
    live_by_zone: Dict[Tuple[str,int], Dict[str, Any]] = {}
    flood_by_zone: Dict[Tuple[str,int], Dict[str, Any]] = {}
    source_status: Dict[str, Any] = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }

    if os.getenv("LIVE_OFFLINE", "0") == "1":
        source_status["sources"] = {
            "open_meteo_ecmwf": {"enabled": False, "available": False, "details": ["LIVE_OFFLINE=1"]},
            "open_meteo_flood_glofas": {"enabled": False, "available": False, "details": ["LIVE_OFFLINE=1"]},
            "imd": {"enabled": False, "available": False, "details": ["LIVE_OFFLINE=1"]},
            "nasa_power": {"enabled": False, "available": False, "details": ["LIVE_OFFLINE=1"]},
            "optional_external_sources": {"enabled": False, "details": ["LIVE_OFFLINE=1"]},
            "nvidia_fourcastnet": {"enabled": False, "available": False, "details": ["LIVE_OFFLINE=1"]},
        }
        source_status["sources"]["note"] = "Existing model/map data retained; no network data fetched."
        LIVE_JSON.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        SOURCE_STATUS_JSON.write_text(json.dumps(source_status, indent=2, ensure_ascii=False), encoding="utf-8")
        OUTPUT_HTML.write_text(html, encoding="utf-8")
        return source_status

    # Primary live weather source.
    try:
        for b in batch(zones, BATCH_SIZE):
            live_by_zone.update(fetch_open_meteo_batch(b, session))
        source_status["sources"]["open_meteo_ecmwf"] = {
            "enabled": True,
            "available": bool(live_by_zone),
            "records": len(live_by_zone),
            "url": OPEN_METEO_ECMWF,
        }
    except Exception as exc:
        source_status["sources"]["open_meteo_ecmwf"] = {"enabled": True, "available": False, "error": str(exc), "url": OPEN_METEO_ECMWF}

    # GloFAS discharge context.
    try:
        for b in batch(zones, BATCH_SIZE):
            flood_by_zone.update(fetch_open_meteo_flood_batch(b, session))
        source_status["sources"]["open_meteo_flood_glofas"] = {
            "enabled": True, "available": bool(flood_by_zone), "records": len(flood_by_zone), "url": OPEN_METEO_FLOOD
        }
    except Exception as exc:
        source_status["sources"]["open_meteo_flood_glofas"] = {"enabled": True, "available": False, "error": str(exc), "url": OPEN_METEO_FLOOD}

    # IMD context.
    imd_ctx, imd_status = fetch_imd_context(session)
    source_status["sources"]["imd"] = imd_status | {"url": "https://api.imd.gov.in/public/api_reference.html"}

    # NASA POWER climatology is queried once per district centroid.
    power_by_district: Dict[Tuple[str,str], Optional[float]] = {}
    power_enabled = os.getenv("NASA_POWER_ENABLE", "0") == "1"
    if power_enabled:
        centroids = district_centroids(payload)
        for key, (lat, lon) in centroids.items():
            power_by_district[key] = nasa_power_climatology(lat, lon, session)
            time.sleep(float(os.getenv("NASA_POWER_SLEEP", "0.05")))
        source_status["sources"]["nasa_power"] = {
            "enabled": True,
            "available": any(v is not None for v in power_by_district.values()),
            "records": sum(v is not None for v in power_by_district.values()),
            "url": NASA_POWER_CLIM,
        }
    else:
        source_status["sources"]["nasa_power"] = {"enabled": False, "available": False, "url": NASA_POWER_CLIM}

    # Optional external-data adapters. They are separate because GPM/SMAP/ERA5/Sentinel-1 are gridded/raster sources
    # and must be spatially sampled/regridded before they can safely replace a CatBoost input.
    if os.getenv("ENABLE_OPTIONAL_SOURCES", "0") == "1":
        bounds = payload.get("stateBounds", {})
        lats = [v["minLat"] for v in bounds.values()] + [v["maxLat"] for v in bounds.values()]
        lons = [v["minLon"] for v in bounds.values()] + [v["maxLon"] for v in bounds.values()]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        source_status["sources"]["optional_external_sources"] = build_optional_status(bbox)
    else:
        source_status["sources"]["optional_external_sources"] = {
            "enabled": False,
            "details": ["Set ENABLE_OPTIONAL_SOURCES=1 to run Earthdata/Copernicus/Sentinel-1/static-source adapters."],
        }

    # Optional FourCastNet supplementary forecast call. It is not injected into the CatBoost feature vector
    # because the public NVIDIA API accepts four built-in global inputs and returns variables not represented in the trained 34-feature schema.
    source_status["sources"]["nvidia_fourcastnet"] = fetch_fourcastnet_status(session) | {"url": NVIDIA_FOURCASTNET}

    static_defaults = district_static_defaults()
    model = load_model()
    payload = score_zones(payload, live_by_zone, flood_by_zone, imd_ctx, power_by_district, static_defaults, model)
    payload["apiSources"] = source_status["sources"]
    payload["modelLiveInputPolicy"] = {
        "dynamic_features": "Open-Meteo ECMWF + IMD/NASA POWER where available",
        "static_features": "retained from supplied training dataset/zone geometry to keep the trained feature distribution unchanged",
        "supplementary_sources": "GloFAS discharge and FourCastNet are displayed/contextual; not injected into the 34-feature CatBoost vector",
    }

    LIVE_JSON.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    SOURCE_STATUS_JSON.write_text(json.dumps(source_status, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_HTML.write_text(replace_flood_data(html, payload), encoding="utf-8")
    return source_status


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Refresh live flood-risk data and rebuild the existing HTML map without changing its UI format.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    status = refresh(force=args.force)
    print(json.dumps(status, indent=2))
