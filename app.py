from dotenv import load_dotenv
load_dotenv()
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
LIVE_JSON = BASE / "live_flood_data.json"

st.set_page_config(page_title="Flood Risk AI", page_icon="🌊", layout="wide")


def load_live():
    if not LIVE_JSON.exists():
        return {"states": {}}
    return json.loads(LIVE_JSON.read_text(encoding="utf-8"))


def risk(p):
    if p < 0.33:
        return "LOW", "🟢"
    if p < 0.66:
        return "MEDIUM", "🟡"
    return "HIGH", "🔴"


data = load_live()
rows = []
for state, districts in data.get("states", {}).items():
    for d in districts:
        for z in d.get("zones", []):
            rows.append({
                "state": state,
                "district": d["district"],
                "region_id": z.get("id"),
                "latitude": z.get("lat"),
                "longitude": z.get("lon"),
                "flood_probability": z.get("p", 0.0),
                "risk": risk(float(z.get("p", 0.0)))[0],
                **{k: z.get(k) for k in [
                    "rain_24h_mm", "rain_72h_mm", "rain_7d_mm", "rain_3h_mm", "rain_6h_mm",
                    "rain_12h_mm", "soil_moisture_0_10", "rainfall_anomaly_7d", "runoff_mm",
                    "elevation_m", "hand_m", "distance_to_river_m", "drainage_density_km_per_km2",
                    "slope_deg", "builtup_fraction", "impervious_fraction", "flow_accumulation",
                    "catchment_area_km2", "road_density", "building_density", "surface_water_occurrence",
                    "river_discharge_m3s", "river_discharge_trend", "imd_rainfall_today_mm", "imd_warning"
                ]}
            })

df = pd.DataFrame(rows)
st.title("🌊 AI Flood Risk Prediction")
st.caption("Assam • Bihar • Odisha | State → District → Spatial Region")
st.warning("The trained CatBoost model is unchanged. Live weather/test features are supplied by the configured APIs; training data is synthetic prototype data.")

if df.empty:
    st.error("No live data file is available. Run: python refresh_live_data.py")
    st.stop()

state = st.sidebar.selectbox("1. Select State", sorted(df.state.unique()))
sdf = df[df.state == state].copy()
sp = sdf.flood_probability.mean()
sr, icon = risk(sp)

c1, c2, c3 = st.columns(3)
c1.metric("State", state)
c2.metric("Flood chance", f"{sp:.1%}")
c3.metric("Risk", f"{icon} {sr}")
if sr == "HIGH":
    st.error(f"🔴 HIGH risk — {sp:.1%}")
elif sr == "MEDIUM":
    st.warning(f"🟡 MEDIUM risk — {sp:.1%}")
else:
    st.success(f"🟢 LOW risk — {sp:.1%}")

district = st.selectbox("2. Select District", sorted(sdf.district.unique()))
dd = sdf[sdf.district == district].copy()
dp = dd

district_p = dp.flood_probability.mean()
dr, di = risk(district_p)
a, b, c = st.columns(3)
a.metric("District", district)
b.metric("Flood chance", f"{district_p:.1%}")
c.metric("Risk", f"{di} {dr}")

rs = dp.groupby("region_id", as_index=False).agg({
    "flood_probability":"mean", "latitude":"mean", "longitude":"mean",
    "rain_24h_mm":"mean", "rain_72h_mm":"mean", "rain_7d_mm":"mean",
    "soil_moisture_0_10":"mean", "drainage_density_km_per_km2":"mean",
    "distance_to_river_m":"mean", "elevation_m":"mean", "slope_deg":"mean",
    "runoff_mm":"mean", "builtup_fraction":"mean", "impervious_fraction":"mean"
})
rs["risk"] = rs.flood_probability.map(lambda p: risk(p)[0])

st.subheader("Regions Inside District")
st.dataframe(rs.sort_values("flood_probability", ascending=False), use_container_width=True, hide_index=True)

region_id = st.selectbox("3. Select Region", sorted(rs.region_id.dropna().astype(int)))
row = rs[rs.region_id == region_id].iloc[0]
rp = float(row.flood_probability)
rr, ri = risk(rp)
if rr == "HIGH": st.error(f"🔴 HIGH RISK — {rp:.1%}")
elif rr == "MEDIUM": st.warning(f"🟡 MEDIUM RISK — {rp:.1%}")
else: st.success(f"🟢 LOW RISK — {rp:.1%}")

c = st.columns(4)
c[0].metric("Flood chance", f"{rp:.1%}")
c[1].metric("Rain 24h", f"{row.rain_24h_mm:.1f} mm")
c[2].metric("Rain 72h", f"{row.rain_72h_mm:.1f} mm")
c[3].metric("Rain 7d", f"{row.rain_7d_mm:.1f} mm")

c = st.columns(4)
c[0].metric("Soil moisture", f"{row.soil_moisture_0_10:.3f}")
c[1].metric("Drainage density", f"{row.drainage_density_km_per_km2:.3f} km/km²")
c[2].metric("River distance", f"{row.distance_to_river_m:.0f} m")
c[3].metric("Elevation", f"{row.elevation_m:.1f} m")

st.caption(f"Last live refresh: {data.get('lastLiveRefresh', 'unknown')}")
