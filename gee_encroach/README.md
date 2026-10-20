# GEE encroachment pipeline (Option A)

Detects buildings encroaching on roads using Google Open Buildings v3 (50 cm-derived footprints) and grades them by severity. No satellite imagery is downloaded.

## One-time setup (Colab or local)
1. **Cloud project**: go to https://console.cloud.google.com, create (or pick) a project, enable the *Earth Engine API*, and register it for Earth Engine at https://code.earthengine.google.com/register (choose non-commercial/research if applicable).
2. Put the project ID in [config.py](config.py) (`PROJECT`).
3. `pip install earthengine-api geopandas shapely pandas numpy`
4. First run triggers `ee.Authenticate()` (browser sign-in). In Colab it is the same.

## Run
```bash
python -m gee_encroach.prepare_roads   # local: 1 km sections -> output_gee/road_sections.geojson
python -m gee_encroach.pipeline        # starts 2 Earth Engine export tasks to Drive
# wait for tasks (https://code.earthengine.google.com/tasks), download encroach_buildings_csv.csv into output_gee/
python -m gee_encroach.score           # severity levels + section report + review sample
```

## Outputs (`output_gee/`)
| File | Content |
|---|---|
| `buildings_scored.csv` | zone (A<15 m, B 15-30, C 30-60), distance to centerline, new-construction flag, severity score/level |
| `section_report.csv` | per 1 km section counts, areas, density, risk score, priority |
| `review_sample.csv` | stratified sample to label in QGIS, fill `is_encroachment` |

## Limits
- Distance is measured to the nearest **building edge**. The 15/30/60 m bands are placeholders; replace them with the real Right-of-Way width.
- Severity and risk scores are rule-based. A trained model needs the labels from `review_sample.csv`.
- Temporal data is Sentinel-2 derived (about 4 m effective), so it is noisy for single houses. Thresholds `NEW_PRE_MAX` and `NEW_POST_MIN` must be checked visually.
- Open Buildings v3 reflects roughly 2023 imagery.
