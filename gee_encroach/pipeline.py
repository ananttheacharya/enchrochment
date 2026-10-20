"""Step 2: run the encroachment extraction on Google Earth Engine.

For every Open Buildings v3 polygon near the roads this computes
  - the nearest road section and the distance from the road centerline to the building edge
  - the encroachment zone (A/B/C)
  - mean 'building_presence' per year 2016-2023 from the Open Buildings 2.5D Temporal dataset
and exports one CSV (no geometry, plus centroid lon/lat) and one GeoJSON-ready table to Drive.

    python -m gee_encroach.pipeline            # authenticates if needed, starts export tasks
"""
import os
import json
import ee

from .config import GeeConfig

OB_POLYGONS = "GOOGLE/Research/open-buildings/v3/polygons"
OB_TEMPORAL = "GOOGLE/Research/open-buildings-temporal/v1"


def init(cfg: GeeConfig):
    try:
        ee.Initialize(project=cfg.PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=cfg.PROJECT)


def load_sections(cfg: GeeConfig) -> ee.FeatureCollection:
    path = os.path.join(cfg.OUT_DIR, "road_sections.geojson")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run `python -m gee_encroach.prepare_roads` first")
    with open(path) as f:
        gj = json.load(f)
    return ee.FeatureCollection(gj)


def build_buildings(cfg: GeeConfig, sections: ee.FeatureCollection) -> ee.FeatureCollection:
    zone_c = cfg.ZONE_C_M
    region = sections.geometry().buffer(zone_c + 10)
    b = (ee.FeatureCollection(OB_POLYGONS)
         .filterBounds(region)
         .filter(ee.Filter.gte("confidence", cfg.MIN_CONF))
         .filter(ee.Filter.gte("area_in_meters", cfg.MIN_AREA_M2)))

    # nearest road section within ZONE_C_M; measureKey stores that distance (metres)
    near = ee.Filter.withinDistance(distance=zone_c, leftField=".geo", rightField=".geo", maxError=1)
    joined = ee.Join.saveBest(matchKey="sec", measureKey="dist_m").apply(b, sections, near)

    a, bb = cfg.ZONE_A_M, cfg.ZONE_B_M

    def tag(f):
        f = ee.Feature(f)
        d = ee.Number(f.get("dist_m"))
        zone = ee.Algorithms.If(d.lt(a), "A", ee.Algorithms.If(d.lt(bb), "B", "C"))
        c = f.geometry().centroid(1).coordinates()
        return (f.set({"sec_id": ee.Feature(f.get("sec")).get("sec_id"), "zone": zone,
                       "lon": c.get(0), "lat": c.get(1)})
                .select(["sec_id", "zone", "dist_m", "confidence", "area_in_meters", "lon", "lat"]))

    return ee.FeatureCollection(joined).map(tag)


def yearly_presence(cfg: GeeConfig) -> ee.Image:
    col = ee.ImageCollection(OB_TEMPORAL)
    bands = []
    for y in cfg.YEARS:
        img = (col.filterDate(f"{y}-01-01", f"{y + 1}-01-01").select("building_presence")
               .mosaic().rename(f"pres_{y}"))
        bands.append(img)
    return ee.Image.cat(bands)


def check_temporal_coverage(cfg: GeeConfig, sections: ee.FeatureCollection):
    n = (ee.ImageCollection(OB_TEMPORAL).filterBounds(sections.geometry().bounds())
         .filterDate(f"{cfg.YEARS[-1]}-01-01", f"{cfg.YEARS[-1] + 1}-01-01").size().getInfo())
    print(f"Temporal tiles covering the roads in {cfg.YEARS[-1]}: {n}")
    if n == 0:
        print("WARNING: no Open Buildings Temporal coverage here; temporal columns will be empty. "
              "Use the Sentinel-2 fallback (see README).")
    return n


def main():
    cfg = GeeConfig()
    init(cfg)
    sections = load_sections(cfg)
    print("sections:", sections.size().getInfo())
    check_temporal_coverage(cfg, sections)

    blds = build_buildings(cfg, sections)
    print("buildings within zone C:", blds.size().getInfo())

    # temporal presence per building polygon (4 m effective resolution)
    enriched = yearly_presence(cfg).reduceRegions(collection=blds, reducer=ee.Reducer.mean(), scale=4)

    cols = ["sec_id", "zone", "dist_m", "confidence", "area_in_meters", "lon", "lat"] + \
           [f"pres_{y}" for y in cfg.YEARS]
    t1 = ee.batch.Export.table.toDrive(collection=enriched, description="encroach_buildings_csv",
                                       folder=cfg.DRIVE_FOLDER, fileFormat="CSV", selectors=cols)
    t2 = ee.batch.Export.table.toDrive(collection=enriched, description="encroach_buildings_geojson",
                                       folder=cfg.DRIVE_FOLDER, fileFormat="GeoJSON")
    t1.start(); t2.start()
    print("Started export tasks. Monitor at https://code.earthengine.google.com/tasks")
    print(f"Results land in Drive/{cfg.DRIVE_FOLDER}. Download encroach_buildings_csv.csv into {cfg.OUT_DIR}.")


if __name__ == "__main__":
    main()
