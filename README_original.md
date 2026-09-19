# Assam + Bihar + Odisha Flood Risk AI

## Trained model
CatBoostClassifier on the uploaded 500,000-row / 43-column dataset.

Validation metrics:
- ROC-AUC: 0.9637
- F1: 0.8586
- Accuracy: 0.8921
- Precision: 0.8567
- Recall: 0.8605

## Critical warning
The uploaded file is explicitly marked SYNTHETIC_PROTOTYPE / SYNTHETIC_NOT_OBSERVATION.
This model is a working ML/software prototype, NOT an operational flood-warning model.

## Run
Place all files in one directory, install requirements, then:
streamlit run app.py

## Risk colors
Green < 33%, Yellow 33%–<66%, Red >=66%.

## Region
Region IDs are spatial KMeans clusters inside each district, not official administrative units.

## Live APIs
The bottom of app.py contains commented integration templates for IMD, NASA GPM,
NASA SMAP, ERA5/Copernicus, Sentinel-1, HydroSHEDS, OSM and JRC Global Surface Water.
