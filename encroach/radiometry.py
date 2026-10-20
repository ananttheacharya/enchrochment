"""Radiometry: LISS-4 DN -> TOA reflectance, and relative normalisation of swath 2 to swath 1."""
import os
import glob
import datetime as dt
import numpy as np

# Mean exo-atmospheric solar irradiance for Resourcesat LISS-4 bands (mW / cm2 / um).
# Published values for RS-2 LISS-IV; RS-2A carries the same sensor design. Small errors here only
# scale a band by a constant, and the PIF normalisation below removes any residual between dates.
ESUN = {2: 181.89, 3: 156.96, 4: 110.48}


def read_meta(folder):
    """Parse BAND_META.txt (key= value per line) into a dict of strings."""
    hits = glob.glob(os.path.join(folder, "**", "BAND_META.txt"), recursive=True)
    if not hits:
        return {}
    meta = {}
    with open(hits[0], encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def _date(meta):
    s = meta.get("DateOfPass", "")
    try:
        return dt.datetime.strptime(s, "%d-%b-%Y").date()
    except ValueError:
        return None


def earth_sun_distance(date):
    doy = date.timetuple().tm_yday
    return 1 - 0.01672 * np.cos(np.deg2rad(0.9856 * (doy - 4)))


def toa_coeffs(meta, bands=(2, 3, 4)):
    """Return [(gain, offset)] so that reflectance = gain * DN + offset, or None if metadata is incomplete.

    L   = Lmin + (Lmax - Lmin) * DN / (2^bits - 1)          [mW / cm2 / sr / um]
    rho = pi * L * d^2 / (ESUN * cos(sun_zenith))
    """
    try:
        bits = int(meta.get("BitsPerPixel", 10))
        dmax = 2 ** bits - 1
        elev = float(meta["SunElevationAtCenter"])
        d = earth_sun_distance(_date(meta))
        cz = np.cos(np.deg2rad(90.0 - elev))
        out = []
        for b in bands:
            lmin, lmax = float(meta[f"B{b}_Lmin"]), float(meta[f"B{b}_Lmax"])
            k = np.pi * d * d / (ESUN[b] * cz)
            out.append((k * (lmax - lmin) / dmax, k * lmin))
        return out
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def describe(meta):
    return (f"{meta.get('SatID', '?')} {meta.get('Sensor', '?')} {meta.get('DateOfPass', '?')} | "
            f"sun elev {meta.get('SunElevationAtCenter', '?')} | bits {meta.get('BitsPerPixel', '?')}")


# --------------------------------------------------------------------------- relative normalisation
def fit_pif(A, B, keep=0.5, n_iter=4, trim=2.5):
    """Robust per-band linear map B -> A on pseudo-invariant pixels.

    A, B: (3, N) reflectance of the same pixels on the reference / target date (Green, Red, NIR).
    Pixels whose NDVI and brightness rank changed least between dates are treated as invariant;
    then a least-squares line is refit while trimming residual outliers.
    Returns [(slope, intercept)] per band.
    """
    nd_a = (A[2] - A[1]) / (A[2] + A[1] + 1e-6)
    nd_b = (B[2] - B[1]) / (B[2] + B[1] + 1e-6)
    rk = lambda x: np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
    change = np.abs(rk(nd_a) - rk(nd_b)) + np.abs(rk(A.mean(0)) - rk(B.mean(0)))
    sel = change <= np.quantile(change, keep)
    coefs = []
    for i in range(A.shape[0]):
        a, b = A[i][sel], B[i][sel]
        m = np.ones(len(a), bool)
        s, c = 1.0, 0.0
        for _ in range(n_iter):
            s, c = np.polyfit(b[m], a[m], 1)
            r = a - (s * b + c)
            mad = np.median(np.abs(r[m] - np.median(r[m]))) * 1.4826 + 1e-9
            m = np.abs(r - np.median(r[m])) <= trim * mad
        coefs.append((float(s), float(c)))
    return coefs


def fit_quantile(A, B):
    q = np.linspace(0, 100, 101)
    out = []
    for i in range(A.shape[0]):
        sq = np.percentile(B[i], q) + np.arange(len(q)) * 1e-9
        out.append((sq.tolist(), np.percentile(A[i], q).tolist()))
    return out


def apply_norm(img, valid, norm):
    """norm = {"kind": "pif"|"quantile", "coefs": ...}; modifies img (3, h, w) in place on valid pixels."""
    if not norm:
        return img
    for i in range(img.shape[0]):
        v = img[i][valid]
        if norm["kind"] == "pif":
            s, c = norm["coefs"][i]
            img[i][valid] = s * v + c
        else:
            sq, rq = norm["coefs"][i]
            img[i][valid] = np.interp(v, sq, rq)
    return img
