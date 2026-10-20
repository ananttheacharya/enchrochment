"""Central configuration for the Encroachment pipeline.

Every notebook does:
    from encroach.config import Config
    cfg = Config()                      # defaults, auto-detected paths
    cfg = Config(TILE=512, SEED=1)      # or override anything
Only paths / a few switches normally need changing.
"""
import os
import glob
from dataclasses import dataclass, field, asdict


def default_root():
    for p in ("/content/drive/MyDrive/Encroachment",
              "/content/drive/My Drive/Encroachment",
              r"G:\My Drive\Encroachment"):
        if os.path.isdir(p):
            return p
    return os.getcwd()


def discover_swaths(root):
    """Return {swath_name: folder_with_BAND2.tif}, ordered by name (swath1 first = priority)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "swath*"))):
        hits = glob.glob(os.path.join(d, "**", "BAND2.tif"), recursive=True)
        if hits:
            out[os.path.basename(d)] = os.path.dirname(hits[0])
    return out


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    ROOT: str = field(default_factory=default_root)
    SWATHS: dict = None            # auto: {name: folder}; first entry has priority in the overlap
    AOI_SHP: str = None            # auto: ROOT/layers/aoi_ring_15_200m.shp
    BUILD_SHP: str = None          # auto: ROOT/layers/buildings_ring_15_200m.shp
    ROAD_SHP: str = None           # auto: ROOT/Road_line/road_complete.shp
    OUT_DIR: str = None            # auto: ROOT/output_v3
    PATCH_DIR: str = None          # auto: /content/patches on Colab (fast local disk), else OUT_DIR/patches
    VALIDATION_POINTS: str = None  # optional point shapefile, integer field 1 = building, 0 = not
    VALIDATION_FIELD: str = "class"

    # ------------------------------------------------------------ tiling
    TILE: int = 1024
    HALO: int = 16                 # context for hand-made features (largest filter radius ~12 px)
    DL_HALO: int = 64              # context for CNN inference

    # ------------------------------------------------------ radiometry
    TOA: bool = True               # DN -> top-of-atmosphere reflectance using BAND_META.txt
    NORMALISE: str = "pif"         # "pif" (robust linear on stable pixels), "quantile", or None
    COREG: bool = True             # sub-pixel phase-correlation of swath 2 onto swath 1
    COREG_MAX_PX: float = 3.0      # reject co-registration estimates larger than this

    # -------------------------------------------- footprint alignment
    AUTO_SHIFT: bool = True
    MAX_SHIFT_PX: int = 4
    SHIFT_STEP_PX: float = 0.25    # sub-pixel refinement step
    SHIFT_MIN_GAIN: float = 0.03

    # ------------------------------------------------------------ labels
    BUILD_MIN_COVER: float = 0.50
    BG_BUFFER_PX: int = 2
    HARD_NEG_SHARE: float = 0.40

    # --------------------------------------------------- split / sample
    BLOCK_M: int = 500
    TEST_FRACTION: float = 0.30
    VAL_FRACTION: float = 0.15
    N_TRAIN_POS: int = 40000
    NEG_RATIO: int = 4
    N_MIX: int = 20000             # partly covered pixels, used only by the coverage regressor
    N_EVAL_PER_CLASS: int = 5000   # per class, for validation and test sample points

    # -------------------------------------------------------------- GBM
    OPTUNA_TRIALS: int = 30
    CLEAN_LABELS: bool = True
    CLEAN_THR: float = 0.92        # drop train negatives whose out-of-fold P(building) >= this

    # ---------------------------------------------------- post-process
    HYSTERESIS: bool = True
    HYST_LOW_FRAC: float = 0.6     # low threshold = HYST_LOW_FRAC * threshold

    # --------------------------------------------------------------- DL
    PATCH: int = 256
    PATCH_STRIDE: int = 128
    MIN_VALID_FRAC: float = 0.05
    ARCH: str = "Unet"             # any segmentation_models_pytorch arch: Unet, UnetPlusPlus, FPN, ...
    ENCODER: str = "resnet34"
    ENCODER_WEIGHTS: str = "imagenet"
    UPSAMPLE: int = 1              # 2 = run the network on a 2x bicubic-upsampled input
    EPOCHS: int = 80
    BATCH: int = 16
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    POS_PATCH_SHARE: float = 0.7   # share of sampled patches that contain buildings
    PATIENCE: int = 15
    IGNORE_MIXED: bool = True      # mask head ignores 0 < cover < BUILD_MIN_COVER pixels
    TVERSKY_ALPHA: float = 0.3     # FP weight
    TVERSKY_BETA: float = 0.7      # FN weight (higher = favour recall)
    W_COVER: float = 0.5
    W_BOUNDARY: float = 0.2
    TTA: bool = True
    INFER_WINDOW: int = 512
    NUM_WORKERS: int = 2

    # -------------------------------------------------- encroachment
    DIST_BANDS_M: tuple = (15, 30, 60, 100, 200)
    SEGMENT_M: int = 1000

    SEED: int = 42

    def __post_init__(self):
        r = self.ROOT
        if self.SWATHS is None:
            self.SWATHS = discover_swaths(r)
        if self.AOI_SHP is None:
            p_layers = os.path.join(r, "layers", "aoi_ring_15_200m.shp")
            p_data = os.path.join(r, "data", "aoi_ring_15_200m.shp")
            self.AOI_SHP = p_layers if os.path.exists(p_layers) else p_data
        if self.BUILD_SHP is None:
            p_layers = os.path.join(r, "layers", "buildings_ring_15_200m.shp")
            p_data = os.path.join(r, "data", "buildings_ring_15_200m.shp")
            self.BUILD_SHP = p_layers if os.path.exists(p_layers) else p_data
        if self.ROAD_SHP is None:
            self.ROAD_SHP = os.path.join(r, "Road_line", "road_complete.shp")
        if self.OUT_DIR is None:
            self.OUT_DIR = os.path.join(r, "output_v3")
        if self.PATCH_DIR is None:
            self.PATCH_DIR = "/content/patches" if os.path.isdir("/content") else os.path.join(self.OUT_DIR, "patches")
        os.makedirs(self.OUT_DIR, exist_ok=True)

    def P(self, name):
        return os.path.join(self.OUT_DIR, name)

    def check(self):
        """Print what was found and raise if something essential is missing."""
        print("ROOT     :", self.ROOT)
        for n, f in self.SWATHS.items():
            print(f"swath    : {n} -> {f}")
        missing = []
        for k in ("AOI_SHP", "BUILD_SHP", "ROAD_SHP"):
            p = getattr(self, k)
            ok = os.path.exists(p)
            print(f"{k:9s}: {'OK     ' if ok else 'MISSING'} {p}")
            if not ok and k != "ROAD_SHP":
                missing.append(p)
        print("OUT_DIR  :", self.OUT_DIR)
        if not self.SWATHS:
            missing.append("no swath folders with BAND2.tif under ROOT")
        if missing:
            raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing) +
                                    "\nUpload them or pass the paths, e.g. Config(AOI_SHP=..., BUILD_SHP=...)")
        return self

    def to_dict(self):
        return asdict(self)
