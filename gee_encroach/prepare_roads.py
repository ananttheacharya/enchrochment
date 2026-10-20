"""Step 1 (local, no Earth Engine needed): cut the road network into sections.

Sections are made in a metric UTM CRS so lengths are real metres, then saved as
WGS84 GeoJSON for upload to Earth Engine (small enough to send inline, no asset upload needed).

    python -m gee_encroach.prepare_roads
"""
import os
import json
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import substring, linemerge

from .config import GeeConfig


def make_sections(cfg: GeeConfig) -> gpd.GeoDataFrame:
    roads = gpd.read_file(cfg.ROAD_SHP)
    roads = roads[roads.geometry.notna() & ~roads.is_empty]
    utm = roads.estimate_utm_crs()
    roads = roads.to_crs(utm)
    print(f"{len(roads)} road features | UTM CRS {utm.to_string()}")

    rows, k = [], 0
    for ridx, g in zip(roads.index, roads.geometry):
        merged = linemerge(g) if g.geom_type == "MultiLineString" else g
        lines = [merged] if isinstance(merged, LineString) else list(merged.geoms)
        for line in lines:
            n = int(-(-line.length // cfg.SEGMENT_M))
            for i in range(n):
                a, b = i * cfg.SEGMENT_M, min((i + 1) * cfg.SEGMENT_M, line.length)
                if b - a < 5:
                    continue
                rows.append({"sec_id": f"SEC_{k:04d}", "len_m": round(b - a, 1), "road_idx": int(ridx),
                             "geometry": substring(line, a, b)})
                k += 1
    sec = gpd.GeoDataFrame(rows, crs=utm)
    print(f"{len(sec)} sections | total {sec.len_m.sum() / 1000:.1f} km")
    return sec


def main():
    cfg = GeeConfig()
    sec = make_sections(cfg).to_crs(4326)
    sec["geometry"] = sec.geometry.simplify(1e-5)       # ~1 m, keeps the upload small
    out = os.path.join(cfg.OUT_DIR, "road_sections.geojson")
    sec.to_file(out, driver="GeoJSON")
    size_mb = os.path.getsize(out) / 1e6
    print(f"Wrote {out} ({size_mb:.2f} MB)")
    if size_mb > 8:
        print("WARNING: >8 MB - Earth Engine's inline request limit is 10 MB. Upload as an asset instead.")


if __name__ == "__main__":
    main()
