"""Encroachment Analysis & Reporting for Every Road Section.

Splits continuous road networks into uniform segments (e.g., 1000m), computes buffer
zones (15m, 30m, 60m), vectorizes predicted building footprints, computes exact distances
to centerline, classifies encroachment severity, and performs temporal change detection
(Nov 2025 -> Mar 2026) in the swath overlap.
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import substring
import rasterio
from rasterio.features import shapes
from scipy.ndimage import label, binary_closing


def segment_road_network(roads_gdf, segment_length=1000.0):
    """Cut continuous line geometries into equal-length segments with unique IDs."""
    segments = []
    seg_idx = 0
    
    for row_idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)
        for line in lines:
            line_len = line.length
            if line_len <= 0:
                continue
            
            n_parts = int(np.ceil(line_len / segment_length))
            for i in range(n_parts):
                d_start = i * segment_length
                d_end = min((i + 1) * segment_length, line_len)
                if d_end - d_start < 5.0:  # skip micro-slivers < 5m
                    continue
                    
                sub = substring(line, d_start, d_end)
                attr = dict(row)
                attr.pop("geometry", None)
                attr["sec_id"] = f"SEC_{seg_idx:04d}"
                attr["sec_len_m"] = round(float(sub.length), 1)
                attr["orig_line_idx"] = row_idx
                segments.append({"geometry": sub, **attr})
                seg_idx += 1
                
    gdf_sec = gpd.GeoDataFrame(segments, crs=roads_gdf.crs)
    print(f"Divided road network into {len(gdf_sec):,} sections of length ~{segment_length:.0f}m")
    return gdf_sec


def polygonize_predicted_buildings(scene, min_area_m2=20.0):
    """Vectorize classification raster into polygon footprints with footprint geometry cleanup."""
    cfg = scene.cfg
    clf_path = cfg.P("classification.tif")
    prob_path = cfg.P("building_probability.tif")
    
    with rasterio.open(clf_path) as src:
        clf_arr = src.read(1)
        transform = src.transform
        crs = src.crs
        
    with rasterio.open(prob_path) as psrc:
        prob_arr = psrc.read(1)
        
    # Mask of building pixels
    mask = (clf_arr == 1)
    # Small morphological closure to heal small holes
    mask = binary_closing(mask, structure=np.ones((2, 2)))
    
    records = []
    bld_idx = 0
    for geom_dict, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if val != 1:
            continue
        poly = shape(geom_dict)
        area_m2 = poly.area
        if area_m2 < min_area_m2:
            continue
            
        records.append({
            "bld_id": f"BLD_{bld_idx:05d}",
            "area_m2": round(float(area_m2), 1),
            "geometry": poly,
        })
        bld_idx += 1
        
    bld_gdf = gpd.GeoDataFrame(records, crs=crs)
    print(f"Vectorized {len(bld_gdf):,} predicted building footprints (>= {min_area_m2} m²)")
    return bld_gdf


def generate_encroachment_report_all_sections(scene, segment_length=1000.0, buffer_zones=(15, 30, 60)):
    """Generate exhaustive section-by-section encroachment report and GIS layers."""
    cfg = scene.cfg
    t0 = time.time()
    
    roads_gdf = scene.roads
    if len(roads_gdf) == 0:
        raise RuntimeError("No road network found in scene.roads. Please verify ROAD_SHP.")
        
    # 1. Segment roads
    sec_gdf = segment_road_network(roads_gdf, segment_length=segment_length)
    
    # 2. Vectorize building predictions
    bld_gdf = polygonize_predicted_buildings(scene)
    if len(bld_gdf) == 0:
        print("Warning: No buildings were predicted.")
        return None, None
        
    # 3. Spatial index matching
    print("Computing proximity of buildings to road sections...")
    bld_sindex = bld_gdf.sindex
    sec_sindex = sec_gdf.sindex
    
    # Buffer limits
    z1, z2, z3 = buffer_zones
    max_buf = z3
    
    # Enrich buildings with closest road section & distance
    bld_centroids = bld_gdf.geometry.centroid
    closest_sec = []
    dist_to_road = []
    
    for idx, cent in enumerate(bld_centroids):
        # Find candidate sections within max_buf + 20m
        cand_idx = list(sec_sindex.query(cent.buffer(max_buf + 20), predicate="intersects"))
        if not cand_idx:
            closest_sec.append("NONE")
            dist_to_road.append(9999.0)
            continue
        cand_geoms = sec_gdf.geometry.iloc[cand_idx]
        dists = cand_geoms.distance(cent)
        min_i = np.argmin(dists)
        closest_sec.append(sec_gdf["sec_id"].iloc[cand_idx[min_i]])
        dist_to_road.append(float(dists.iloc[min_i]))
        
    bld_gdf["sec_id"] = closest_sec
    bld_gdf["dist_centerline_m"] = np.round(dist_to_road, 1)
    
    # Zone classification
    bld_gdf["encroach_zone"] = "Outside Buffer"
    bld_gdf.loc[bld_gdf["dist_centerline_m"] <= z3, "encroach_zone"] = f"Zone C ({z2}-{z3}m)"
    bld_gdf.loc[bld_gdf["dist_centerline_m"] <= z2, "encroach_zone"] = f"Zone B ({z1}-{z2}m)"
    bld_gdf.loc[bld_gdf["dist_centerline_m"] <= z1, "encroach_zone"] = f"Zone A (<{z1}m Critical)"
    
    # 4. Check for New Construction between Nov 2025 and Mar 2026 where both dates overlap
    print("Performing multi-date change verification in swath overlap...")
    prob_ds = rasterio.open(cfg.P("building_probability.tif"))
    
    change_status = []
    for idx, poly in enumerate(bld_gdf.geometry):
        cx, cy = poly.centroid.x, poly.centroid.y
        row_px, col_px = rasterio.transform.rowcol(scene.MTR, cx, cy)
        # Check if coordinates inside scene bounds
        if 0 <= row_px < scene.H and 0 <= col_px < scene.W:
            # Inspect overlap
            t = scene.make_tile(row_px, col_px, 1, 1, halo=2)
            t = scene.load_imagery(t)
            if t.both.any():
                # Both dates available
                g1, r1, n1 = t.img[:, t.core[0], t.core[1]].ravel()
                g2, r2, n2 = t.alt[:, t.core[0], t.core[1]].ravel()
                nd1 = (n1 - r1) / (n1 + r1 + 1e-6)
                nd2 = (n2 - r2) / (n2 + r2 + 1e-6)
                b1 = (g1 + r1 + n1) / 3
                b2 = (g2 + r2 + n2) / 3
                # Significant brightening + vegetation clearing from Nov 2025 to Mar 2026
                if (b2 - b1) > 0.08 and (nd1 - nd2) > 0.15:
                    change_status.append("New Construction (Nov 2025 -> Mar 2026)")
                else:
                    change_status.append("Pre-existing Structure")
            else:
                change_status.append("Single Date Available")
        else:
            change_status.append("Out of Bounds")
            
    bld_gdf["temporal_status"] = change_status
    prob_ds.close()
    
    # Save enriched building polygons
    bld_encroaching = bld_gdf[bld_gdf["dist_centerline_m"] <= max_buf].copy()
    bld_encroaching.to_file(cfg.P("encroaching_buildings.gpkg"), driver="GPKG")
    print(f"Saved {len(bld_encroaching):,} encroaching buildings to {cfg.P('encroaching_buildings.gpkg')}")
    
    # 5. Aggregate Report by Road Section
    sec_stats = []
    for s_idx, sec in sec_gdf.iterrows():
        sid = sec["sec_id"]
        sec_blds = bld_encroaching[bld_encroaching["sec_id"] == sid]
        
        n_tot = len(sec_blds)
        tot_area = sec_blds["area_m2"].sum() if n_tot > 0 else 0.0
        
        n_z1 = (sec_blds["dist_centerline_m"] <= z1).sum()
        area_z1 = sec_blds.loc[sec_blds["dist_centerline_m"] <= z1, "area_m2"].sum()
        
        n_z2 = ((sec_blds["dist_centerline_m"] > z1) & (sec_blds["dist_centerline_m"] <= z2)).sum()
        area_z2 = sec_blds.loc[(sec_blds["dist_centerline_m"] > z1) & (sec_blds["dist_centerline_m"] <= z2), "area_m2"].sum()
        
        n_z3 = ((sec_blds["dist_centerline_m"] > z2) & (sec_blds["dist_centerline_m"] <= z3)).sum()
        area_z3 = sec_blds.loc[(sec_blds["dist_centerline_m"] > z2) & (sec_blds["dist_centerline_m"] <= z3), "area_m2"].sum()
        
        n_new = (sec_blds["temporal_status"] == "New Construction (Nov 2025 -> Mar 2026)").sum()
        density_km = (n_tot / (sec["sec_len_m"] / 1000.0)) if sec["sec_len_m"] > 0 else 0.0
        
        # Action Priority
        if n_z1 > 0:
            priority = "HIGH (Immediate Action - RoW Infringement)"
        elif n_z2 > 0 or n_new > 0:
            priority = "MEDIUM (Active Encroachment)"
        elif n_z3 > 0:
            priority = "LOW (Buffer Zone Only)"
        else:
            priority = "CLEAR (No Encroachment)"
            
        sec_stats.append({
            "section_id": sid,
            "length_m": sec["sec_len_m"],
            "total_encroaching_buildings": int(n_tot),
            "total_encroached_area_m2": round(float(tot_area), 1),
            f"buildings_critical_lt_{z1}m": int(n_z1),
            f"area_critical_lt_{z1}m": round(float(area_z1), 1),
            f"buildings_zone_{z1}_{z2}m": int(n_z2),
            f"area_zone_{z1}_{z2}m": round(float(area_z2), 1),
            f"buildings_zone_{z2}_{z3}m": int(n_z3),
            f"area_zone_{z2}_{z3}m": round(float(area_z3), 1),
            "new_constructions_nov_to_mar": int(n_new),
            "encroachment_density_per_km": round(float(density_km), 2),
            "action_priority": priority,
        })
        
    df_report = pd.DataFrame(sec_stats)
    csv_path = cfg.P("encroachment_report_all_sections.csv")
    df_report.to_csv(csv_path, index=False)
    
    # Join stats to GeoDataFrame and export
    sec_gdf = sec_gdf.merge(df_report, left_on="sec_id", right_on="section_id")
    sec_gdf.to_file(cfg.P("road_sections_evaluated.gpkg"), driver="GPKG")
    
    print("\n" + "=" * 80)
    print("ENCROACHMENT REPORT FOR ALL ROAD SECTIONS GENERATED:")
    print("=" * 80)
    print(f"Total Road Sections: {len(df_report):,}")
    print(f"Total Road Length:   {df_report['length_m'].sum() / 1000.0:.1f} km")
    print(f"Total Encroaching Structures: {df_report['total_encroaching_buildings'].sum():,}")
    print(f"Critical (<{z1}m) Infringements: {df_report[f'buildings_critical_lt_{z1}m'].sum():,}")
    print(f"New Constructions (Nov->Mar):   {df_report['new_constructions_nov_to_mar'].sum():,}")
    print(f"\nSaved Report CSV:  {csv_path}")
    print(f"Saved GIS Layers:  {cfg.P('road_sections_evaluated.gpkg')}")
    print(f"                   {cfg.P('encroaching_buildings.gpkg')}")
    print(f"Completed in {time.time() - t0:.1f}s")
    return df_report, bld_encroaching
