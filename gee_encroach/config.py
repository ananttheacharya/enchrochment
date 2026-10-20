"""Configuration for the Google Earth Engine (Option A) encroachment pipeline."""
import os
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class GeeConfig:
    # ---- Earth Engine ----
    PROJECT: str = "complete-trees-301313"       # Cloud project registered for Earth Engine
    DRIVE_FOLDER: str = "encroachment_gee"        # Drive folder that receives the exports

    # ---- inputs (local) ----
    ROAD_SHP: str = os.path.join(REPO, "Road_line", "road_complete.shp")
    OUT_DIR: str = os.path.join(REPO, "output_gee")
    SEGMENT_M: float = 1000.0                     # road section length

    # ---- Open Buildings v3 filters ----
    MIN_CONF: float = 0.70                        # Google recommends 0.65-0.75 as a precision/recall balance
    MIN_AREA_M2: float = 20.0                     # drop sheds / noise

    # ---- encroachment bands (metres from road centerline to nearest building edge) ----
    ZONE_A_M: float = 15.0                        # critical
    ZONE_B_M: float = 30.0                        # severe
    ZONE_C_M: float = 60.0                        # buffer / monitor

    # ---- temporal (Open Buildings 2.5D Temporal v1, 2016-2023) ----
    YEARS: tuple = (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)
    PRE_YEARS: tuple = (2016, 2017, 2018)         # "baseline" years
    POST_YEARS: tuple = (2022, 2023)              # "recent" years
    NEW_PRE_MAX: float = 0.15                     # presence below this in baseline ...
    NEW_POST_MIN: float = 0.35                    # ... and above this recently => new construction (TUNE visually)

    def __post_init__(self):
        os.makedirs(self.OUT_DIR, exist_ok=True)
