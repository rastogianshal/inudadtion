from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from live_pipeline import refresh, LIVE_JSON, SOURCE_STATUS_JSON, OUTPUT_HTML

app = FastAPI(title="Flood Risk Watch API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_refresh():
    # Live-by-default for the deployed app; failures keep the last generated map data.
    if os.getenv("AUTO_REFRESH_LIVE", "1") == "1" and os.getenv("LIVE_OFFLINE", "0") != "1":
        try:
            refresh(force=False)
        except Exception:
            pass

@app.get("/", response_class=FileResponse)
def home():
    if not OUTPUT_HTML.exists():
        refresh(force=True)
    return FileResponse(OUTPUT_HTML, media_type="text/html")

@app.get("/api/data")
def data():
    if not LIVE_JSON.exists():
        refresh(force=True)
    return JSONResponse(__import__("json").loads(LIVE_JSON.read_text(encoding="utf-8")))

@app.get("/api/status")
def status():
    if not SOURCE_STATUS_JSON.exists():
        refresh(force=False)
    return JSONResponse(__import__("json").loads(SOURCE_STATUS_JSON.read_text(encoding="utf-8")))

@app.post("/api/refresh")
def refresh_data():
    try:
        return JSONResponse(refresh(force=True))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
