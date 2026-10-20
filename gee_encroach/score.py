"""Step 3 (local): turn the Earth Engine export into severity levels and a per-section report.

    python -m gee_encroach.score [path/to/encroach_buildings_csv.csv]

Outputs (in OUT_DIR):
  buildings_scored.csv     per building: zone, new-construction flag, severity score, severity level
  section_report.csv       per 1 km section: counts/areas by zone, new construction, density, risk score, priority
  review_sample.csv        stratified sample to verify/label in QGIS (needed before any supervised model)

NOTE: severity scores are transparent rules, NOT a trained model. They are the baseline that a supervised
model should beat once you have reviewed labels (see review_sample.csv).
"""
import os
import sys
import numpy as np
import pandas as pd

from .config import GeeConfig

ZONE_W = {"A": 1.0, "B": 0.6, "C": 0.25}


def add_temporal_flag(df: pd.DataFrame, cfg: GeeConfig) -> pd.DataFrame:
    pre = df[[f"pres_{y}" for y in cfg.PRE_YEARS]].mean(axis=1)
    post = df[[f"pres_{y}" for y in cfg.POST_YEARS]].mean(axis=1)
    df["pres_pre"], df["pres_post"] = pre.round(3), post.round(3)
    df["new_construction"] = (pre < cfg.NEW_PRE_MAX) & (post > cfg.NEW_POST_MIN)
    df["temporal_known"] = pre.notna() & post.notna()
    return df


def score_buildings(df: pd.DataFrame) -> pd.DataFrame:
    """Severity in [0,1]: closeness dominates, then footprint size, detection confidence, and recency."""
    zone_w = df["zone"].map(ZONE_W).astype(float)
    # within a zone, nearer = worse (linear over the zone's own width is avoided to keep it simple)
    size = np.clip(np.log1p(df["area_in_meters"]) / np.log1p(300.0), 0, 1)
    conf = np.clip((df["confidence"] - 0.65) / 0.35, 0, 1)
    new = df["new_construction"].astype(float)
    s = 0.60 * zone_w + 0.15 * size + 0.10 * conf + 0.15 * new
    df["severity_score"] = s.round(3)
    df["severity_level"] = pd.cut(s, [-1, 0.30, 0.55, 0.75, 2],
                                  labels=["LOW", "MODERATE", "HIGH", "CRITICAL"]).astype(str)
    return df


def section_report(df: pd.DataFrame, sections_len: pd.DataFrame | None = None) -> pd.DataFrame:
    g = df.groupby("sec_id")
    rep = pd.DataFrame({
        "n_buildings": g.size(),
        "area_m2": g["area_in_meters"].sum().round(1),
        "n_zone_A": g.apply(lambda x: (x.zone == "A").sum()),
        "n_zone_B": g.apply(lambda x: (x.zone == "B").sum()),
        "n_zone_C": g.apply(lambda x: (x.zone == "C").sum()),
        "area_zone_A_m2": g.apply(lambda x: x.loc[x.zone == "A", "area_in_meters"].sum()).round(1),
        "n_new_construction": g["new_construction"].sum(),
        "min_dist_m": g["dist_m"].min().round(1),
        "mean_severity": g["severity_score"].mean().round(3),
        "max_severity": g["severity_score"].max().round(3),
    }).reset_index()
    if sections_len is not None:
        rep = sections_len.merge(rep, on="sec_id", how="left").fillna(
            {c: 0 for c in rep.columns if c != "sec_id"})
        rep["buildings_per_km"] = (rep.n_buildings / (rep.len_m / 1000)).round(2)
        rep["zoneA_per_km"] = (rep.n_zone_A / (rep.len_m / 1000)).round(2)
    # section risk: weighted blend of critical-zone density, total severity and growth, rank-normalised
    r = lambda s: s.rank(pct=True) if s.nunique() > 1 else s * 0
    key_a = rep.get("zoneA_per_km", rep.n_zone_A)
    rep["risk_score"] = (0.5 * r(key_a) + 0.3 * r(rep.mean_severity) + 0.2 * r(rep.n_new_construction)).round(3)
    rep["priority"] = np.select(
        [rep.n_zone_A > 0, (rep.n_zone_B > 0) | (rep.n_new_construction > 0), rep.n_zone_C > 0],
        ["HIGH", "MEDIUM", "LOW"], default="CLEAR")
    return rep.sort_values(["priority", "risk_score"], ascending=[True, False]).reset_index(drop=True)


def review_sample(df: pd.DataFrame, per_stratum: int = 60, seed: int = 42) -> pd.DataFrame:
    parts = [d.sample(min(len(d), per_stratum), random_state=seed)
             for _, d in df.groupby(["zone", "new_construction"])]
    out = pd.concat(parts)
    out["is_encroachment"] = ""     # <- fill 1/0 after viewing lon/lat on high-resolution imagery
    out["notes"] = ""
    return out


def main():
    cfg = GeeConfig()
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(cfg.OUT_DIR, "encroach_buildings_csv.csv")
    df = pd.read_csv(src)
    print(f"{len(df):,} buildings loaded")
    df = score_buildings(add_temporal_flag(df, cfg))
    df.to_csv(os.path.join(cfg.OUT_DIR, "buildings_scored.csv"), index=False)

    sec_path = os.path.join(cfg.OUT_DIR, "road_sections.geojson")
    lens = None
    if os.path.exists(sec_path):
        import geopandas as gpd
        lens = pd.DataFrame(gpd.read_file(sec_path)[["sec_id", "len_m"]])
    rep = section_report(df, lens)
    rep.to_csv(os.path.join(cfg.OUT_DIR, "section_report.csv"), index=False)
    review_sample(df).to_csv(os.path.join(cfg.OUT_DIR, "review_sample.csv"), index=False)

    print(df.groupby(["zone", "severity_level"]).size().unstack(fill_value=0))
    print(rep.priority.value_counts().to_string())
    print(f"New-construction flagged: {int(df.new_construction.sum()):,} "
          f"(temporal data present for {df.temporal_known.mean():.0%})")


if __name__ == "__main__":
    main()
