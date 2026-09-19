"""Optional external-data adapters for the flood-risk project.

These sources are intentionally separated from the 34-feature CatBoost scoring path.
The supplied model was trained on a synthetic feature distribution. Replacing static
terrain/land-use values with a different source without retraining can create feature
distribution shift. The adapters therefore fetch/archive/validate external products and
are ready for a future retraining pipeline.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

BASE = Path(__file__).resolve().parent
EXT = BASE / "data" / "external"
EXT.mkdir(parents=True, exist_ok=True)


def earthdata_search(short_name: str, bbox: tuple[float, float, float, float], count: int = 5) -> Dict[str, Any]:
    try:
        import earthaccess
        auth = earthaccess.login(strategy="environment", persist=False)
        if not auth:
            return {"available": False, "reason": "NASA Earthdata credentials not available."}
        results = earthaccess.search_data(short_name=short_name, bounding_box=bbox, count=count)
        return {
            "available": True,
            "short_name": short_name,
            "count": len(results),
            "results": [str(r) for r in results[:count]],
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc), "short_name": short_name}


def gpm_imerg_search(bbox: tuple[float, float, float, float]) -> Dict[str, Any]:
    short_name = os.getenv("GPM_SHORT_NAME", "GPM_3IMERGHHE_07")
    return earthdata_search(short_name, bbox, count=int(os.getenv("GPM_SEARCH_COUNT", "3")))


def smap_search(bbox: tuple[float, float, float, float]) -> Dict[str, Any]:
    short_name = os.getenv("SMAP_SHORT_NAME", "SPL4SMAU")
    return earthdata_search(short_name, bbox, count=int(os.getenv("SMAP_SEARCH_COUNT", "3")))


def copernicus_sentinel1_search(bbox: tuple[float, float, float, float]) -> Dict[str, Any]:
    url = "https://catalogue.dataspace.copernicus.eu/stac/search"
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(os.getenv("SENTINEL1_LOOKBACK_DAYS", "3")))
    payload = {
        "bbox": list(bbox),
        "datetime": f"{start.isoformat().replace('+00:00','Z')}/{now.isoformat().replace('+00:00','Z')}",
        "collections": ["sentinel-1-grd"],
        "limit": int(os.getenv("SENTINEL1_LIMIT", "20")),
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        (EXT / "sentinel1_latest_search.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return {"available": True, "url": url, "features": len(data.get("features", [])), "file": str(EXT / "sentinel1_latest_search.json")}
    except Exception as exc:
        return {"available": False, "url": url, "reason": str(exc)}


def retrieve_era5_latest(bbox: tuple[float, float, float, float]) -> Dict[str, Any]:
    if os.getenv("ERA5_ENABLE", "0") != "1":
        return {"enabled": False, "available": False, "reason": "ERA5_ENABLE=0"}
    try:
        import cdsapi
        client = cdsapi.Client()
        north, west, south, east = bbox[3], bbox[0], bbox[1], bbox[2]
        now = datetime.now(timezone.utc) - timedelta(hours=6)
        target = EXT / "era5_latest.nc"
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
                    "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation"
                ],
                "year": now.strftime("%Y"),
                "month": now.strftime("%m"),
                "day": now.strftime("%d"),
                "time": [f"{now.strftime('%H')}:00"],
                "area": [north, west, south, east],
                "format": "netcdf",
            },
            str(target),
        )
        return {"enabled": True, "available": True, "file": str(target), "url": "https://cds.climate.copernicus.eu/"}
    except Exception as exc:
        return {"enabled": True, "available": False, "reason": str(exc), "url": "https://cds.climate.copernicus.eu/"}


def hydrosheds_static_status() -> Dict[str, Any]:
    path = os.getenv("HYDROSHEDS_DEM_PATH", "").strip()
    return {
        "enabled": bool(path),
        "available": bool(path and Path(path).exists()),
        "path": path or None,
        "url": "https://www.hydrosheds.org/products",
        "note": "HydroSHEDS is a static GIS source; use it to regenerate elevation/flow/HAND/catchment features when retraining."
    }


def osm_static_status() -> Dict[str, Any]:
    path = os.getenv("OSM_EXTRACT_PATH", "").strip()
    return {
        "enabled": bool(path),
        "available": bool(path and Path(path).exists()),
        "path": path or None,
        "url": "https://www.openstreetmap.org/",
        "note": "Use a regional extract for road/building density; do not query the public API once per map zone."
    }


def jrc_surface_water_status() -> Dict[str, Any]:
    return {
        "enabled": os.getenv("JRC_GSW_ENABLE", "0") == "1",
        "available": os.getenv("JRC_GSW_ENABLE", "0") == "1",
        "url": "https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater",
        "note": "Historical surface-water occurrence; not a live flood label."
    }


def bhuvan_wms_status() -> Dict[str, Any]:
    url = os.getenv("BHUVAN_WMS_URL", "").strip()
    return {
        "enabled": bool(url),
        "available": bool(url),
        "url": url or "https://bhuvan-app1.nrsc.gov.in/",
        "note": "Bhuvan flood-hazard WMS is treated as contextual map data unless a validated layer-specific service URL is configured."
    }


def build_optional_status(bbox: tuple[float, float, float, float]) -> Dict[str, Any]:
    return {
        "nasa_gpm_imerg": gpm_imerg_search(bbox),
        "nasa_smap": smap_search(bbox),
        "copernicus_sentinel1": copernicus_sentinel1_search(bbox),
        "copernicus_era5": retrieve_era5_latest(bbox),
        "hydrosheds": hydrosheds_static_status(),
        "openstreetmap": osm_static_status(),
        "jrc_global_surface_water": jrc_surface_water_status(),
        "bhuvan_nrsc": bhuvan_wms_status(),
    }
