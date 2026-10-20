"""Per-pixel features for the GBM and input channels for the CNN.

All functions take a tile loaded by Scene.load_imagery (haloed) and return arrays on the haloed grid.
Reflectance inputs are TOA (0-1); if TOA is off they are DN, which only changes scales.
"""
import numpy as np
from scipy.ndimage import uniform_filter, grey_opening, sobel, gaussian_filter

EPS = 1e-6


def local_std(a, k):
    m = uniform_filter(a, k); m2 = uniform_filter(a * a, k)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def _line(length, angle):
    """Boolean linear structuring element of `length` px at 0/45/90/135 degrees."""
    se = np.zeros((length, length), bool)
    c = length // 2
    if angle == 0:
        se[c, :] = True
    elif angle == 90:
        se[:, c] = True
    elif angle == 45:
        se[np.arange(length)[::-1], np.arange(length)] = True
    else:
        se[np.arange(length), np.arange(length)] = True
    return se


def mbi(b, scales=(3, 7, 11, 15), angles=(0, 45, 90, 135)):
    """Morphological Building Index (Huang & Zhang 2011), simplified: mean over directions of the
    differential white top-hat profile. Bright compact objects (roofs) score high, long bright
    objects (roads) are suppressed because some direction fits inside them."""
    acc = np.zeros_like(b)
    for ang in angles:
        prev = None
        for s in scales:
            th = b - grey_opening(b, footprint=_line(s, ang))
            if prev is not None:
                acc += np.abs(th - prev)
            prev = th
    return acc / (len(angles) * (len(scales) - 1))


def pantex(b, win=9):
    """PanTex-like built-up texture: minimum over 4 directions of local mean squared difference
    to the neighbour (a cheap stand-in for GLCM contrast)."""
    out = None
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        d = b - np.roll(b, (dy, dx), (0, 1))
        c = uniform_filter(d * d, win)
        out = c if out is None else np.minimum(out, c)
    return out


def base_indices(img):
    g, r, n = img
    b = (g + r + n) / 3
    nd = (n - r) / (n + r + EPS)
    return g, r, n, b, nd


FEATURES = [
    "green", "red", "nir", "ndvi", "brightness", "g_r_ratio", "nir_r_ratio",
    "bright_contrast11", "bright_contrast21", "bright_contrast41", "nir_contrast21", "ndvi_contrast21",
    "bright_std3", "bright_std7", "nir_std5", "red_std5", "ndvi_mean7", "bright_mean7",
    "tophat3", "tophat7", "tophat11", "dog1_3", "edge", "mbi", "pantex",
    "road_dist", "alt_ndvi", "alt_bright", "d_ndvi", "d_bright", "swath",
]


def make_features(t, scene):
    g, r, n, b, nd = base_indices(t.img)
    sb = uniform_filter(b, 3)
    edge = np.hypot(sobel(sb, 0), sobel(sb, 1))
    if np.isnan(t.alt).all():
        ab = and_ = np.full_like(b, np.nan)
    else:
        ag, ar, an = t.alt
        ab = (ag + ar + an) / 3
        and_ = (an - ar) / (an + ar + EPS)
    F = [g, r, n, nd, b, g / (r + EPS), n / (r + EPS),
         b - uniform_filter(b, 11), b - uniform_filter(b, 21), b - uniform_filter(b, 41),
         n - uniform_filter(n, 21), nd - uniform_filter(nd, 21),
         local_std(b, 3), local_std(b, 7), local_std(n, 5), local_std(r, 5),
         uniform_filter(nd, 7), uniform_filter(b, 7),
         b - grey_opening(b, size=(3, 3)), b - grey_opening(b, size=(7, 7)), b - grey_opening(b, size=(11, 11)),
         gaussian_filter(b, 1) - gaussian_filter(b, 3), edge, mbi(b), pantex(b),
         scene.road_dist(t), and_, ab, nd - and_, b - ab, t.sw.astype(np.float32)]
    return np.stack(F).astype(np.float32)


# ------------------------------------------------------------------------------------------- CNN
DL_CHANNELS = ["green", "red", "nir", "ndvi", "mbi", "road_dist", "date"]


def dl_channels(t, scene):
    """Inputs for the CNN on the haloed tile (C, hh, ww). `date` = 0 for swath 1, 1 for swath 2."""
    g, r, n, b, nd = base_indices(t.img)
    date = np.where(t.sw == 2, 1.0, 0.0)
    X = np.stack([g, r, n, nd, mbi(b), scene.road_dist(t) / 300.0, date]).astype(np.float32)
    X[:, t.sw == 0] = 0
    return X
