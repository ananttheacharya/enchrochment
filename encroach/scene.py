"""Scene: working grid, AOI, tiles, swath readers (TOA + normalised + co-registered), footprints and roads.

Nothing here builds full-size arrays; everything works per tile.
State that is estimated once (co-registration offsets, normalisation, footprint shifts) is saved to
OUT_DIR/scene_state.json so later notebooks reuse it instead of re-estimating.
"""
import os
import re
import glob
import json
import warnings
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
from rasterio import features
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from scipy.ndimage import uniform_filter, gaussian_filter, distance_transform_edt
from shapely import clip_by_rect
from shapely.geometry import box
from shapely.validation import make_valid

from . import radiometry as rad

warnings.filterwarnings("ignore")


def find_band(folder, n):
    tifs = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
            if f.lower().endswith((".tif", ".tiff"))]
    pat = re.compile(rf"(^|[^a-z0-9])(band|b)_?0?{n}([^0-9]|$)", re.I)
    hits = [f for f in tifs if pat.search(os.path.splitext(os.path.basename(f))[0])]
    if len(hits) != 1:
        raise RuntimeError(f"Could not identify band {n} in {folder}. TIFs found:\n" + "\n".join(tifs))
    return hits[0]


def _union(gs):
    return gs.union_all() if hasattr(gs, "union_all") else gs.unary_union


class Scene:
    def __init__(self, cfg, load_state=True):
        self.cfg = cfg
        self.names = list(cfg.SWATHS)
        self.bands = {n: [find_band(f, b) for b in (2, 3, 4)] for n, f in cfg.SWATHS.items()}
        self.meta = {n: rad.read_meta(f) for n, f in cfg.SWATHS.items()}
        self.toa = {n: (rad.toa_coeffs(self.meta[n]) if cfg.TOA else None) for n in self.names}
        for n in self.names:
            print(f"{n}: {rad.describe(self.meta[n])} | TOA {'on' if self.toa[n] else 'off (DN used)'}")

        with rasterio.open(self.bands[self.names[0]][0]) as s:
            self.CRS, self.RES, T0 = s.crs, s.res[0], s.transform
        assert self.CRS.is_projected and T0.e < 0 and abs(T0.b) < 1e-9, "Expected a north-up projected image"

        aoi = gpd.read_file(cfg.AOI_SHP).to_crs(self.CRS)
        aoi["geometry"] = aoi.geometry.apply(make_valid)
        aoi = aoi[~aoi.is_empty]
        self.aoi = aoi
        self.AOI_UNION = _union(aoi.geometry)
        minx, miny, maxx, maxy = aoi.total_bounds
        RES = self.RES
        col0 = int(np.floor((minx - T0.c) / RES)); col1 = int(np.ceil((maxx - T0.c) / RES))
        row0 = int(np.floor((T0.f - maxy) / RES)); row1 = int(np.ceil((T0.f - miny) / RES))
        self.MTR = Affine(RES, 0, T0.c + col0 * RES, 0, -RES, T0.f - row0 * RES)
        self.W, self.H = col1 - col0, row1 - row0
        print(f"Grid: {self.W} x {self.H} px | AOI {aoi.area.sum() / 1e4:.0f} ha")

        self.blk = max(1, int(round(cfg.BLOCK_M / RES)))
        self.ncb = self.W // self.blk + 1
        self.NB = (self.H // self.blk + 1) * self.ncb

        # estimated state
        self.offsets = {n: (0.0, 0.0) for n in self.names}       # (dy, dx) px applied to each swath
        self.norm = {n: None for n in self.names}                 # relative normalisation to swath 1
        self.shifts = {k + 1: (0.0, 0.0) for k in range(len(self.names))}  # footprint shift (east, north) m
        if load_state and os.path.exists(self.state_path):
            self.load_state()
        self._open()

        self.TILES = []
        T = cfg.TILE
        for r0 in range(0, self.H, T):
            for c0 in range(0, self.W, T):
                h, w = min(T, self.H - r0), min(T, self.W - c0)
                x0, y1 = self.MTR * (c0, r0); x1, y0 = self.MTR * (c0 + w, r0 + h)
                if self.AOI_UNION.intersects(box(x0, y0, x1, y1)):
                    self.TILES.append((r0, c0, h, w))
        print(f"{len(self.TILES)} tiles touch the AOI")
        self._bld = None
        self._roads = None

    # ------------------------------------------------------------------ state
    @property
    def state_path(self):
        return self.cfg.P("scene_state.json")

    def save_state(self):
        st = dict(offsets=self.offsets, norm=self.norm, shifts={str(k): v for k, v in self.shifts.items()})
        with open(self.state_path, "w") as f:
            json.dump(st, f, indent=1)
        print("saved", self.state_path)

    def load_state(self):
        with open(self.state_path) as f:
            st = json.load(f)
        self.offsets = {n: tuple(st["offsets"].get(n, (0.0, 0.0))) for n in self.names}
        self.norm = {n: st["norm"].get(n) for n in self.names}
        self.shifts = {int(k): tuple(v) for k, v in st["shifts"].items()}
        print("loaded state:", {"offsets": self.offsets, "shifts": self.shifts,
                                "norm": {n: (v or {}).get("kind") for n, v in self.norm.items()}})

    # ---------------------------------------------------------------- readers
    def _open(self):
        for sw in getattr(self, "SW", []):
            for v in sw.vrts:
                v.close()
            for s in sw.srcs:
                s.close()
        self.SW = []
        for k, n in enumerate(self.names):
            srcs = [rasterio.open(p) for p in self.bands[n]]
            dy, dx = self.offsets[n]
            vrts = []
            for s in srcs:
                kw = dict(crs=self.CRS, transform=self.MTR, width=self.W, height=self.H,
                          src_nodata=0, nodata=0,
                          resampling=Resampling.nearest if (k == 0 and dy == 0 and dx == 0) else Resampling.bilinear)
                if dy or dx:
                    kw["src_transform"] = Affine.translation(dx * self.RES, -dy * self.RES) * s.transform
                vrts.append(WarpedVRT(s, **kw))
            self.SW.append(NS(name=n, srcs=srcs, vrts=vrts))

    def set_offset(self, name, dy, dx):
        self.offsets[name] = (float(dy), float(dx))
        self._open()

    def read_dn(self, k, win):
        return np.stack([v.read(1, window=win).astype("float32") for v in self.SW[k].vrts])

    def read_sw(self, k, win, normalise=True):
        """Reflectance (3, h, w) and validity of swath k. Invalid pixels are 0."""
        a = self.read_dn(k, win)
        v = (a > 0).all(0)
        co = self.toa[self.names[k]]
        if co:
            for i, (g, o) in enumerate(co):
                a[i] = np.where(v, a[i] * g + o, 0)
        else:
            a[:, ~v] = 0
        if normalise and self.norm[self.names[k]]:
            rad.apply_norm(a, v, self.norm[self.names[k]])
        return a, v

    # ------------------------------------------------------------------ tiles
    def make_tile(self, r0, c0, h, w, halo=None):
        halo = self.cfg.HALO if halo is None else halo
        rh0, ch0 = max(0, r0 - halo), max(0, c0 - halo)
        rh1, ch1 = min(self.H, r0 + h + halo), min(self.W, c0 + w + halo)
        t = NS(r0=r0, c0=c0, h=h, w=w, rh0=rh0, ch0=ch0, hh=rh1 - rh0, ww=ch1 - ch0)
        t.core = (slice(r0 - rh0, r0 - rh0 + h), slice(c0 - ch0, c0 - ch0 + w))
        t.win = Window(ch0, rh0, t.ww, t.hh)
        t.cwin = Window(c0, r0, w, h)
        return t

    def tiles(self, halo=None):
        for r0, c0, h, w in self.TILES:
            yield self.make_tile(r0, c0, h, w, halo)

    def tile_transform(self, t):
        return self.MTR * Affine.translation(t.ch0, t.rh0)

    def aoi_mask(self, t):
        x0, y1 = self.MTR * (t.ch0, t.rh0); x1, y0 = self.MTR * (t.ch0 + t.ww, t.rh0 + t.hh)
        g = clip_by_rect(self.AOI_UNION, x0, y0, x1, y1)
        if g.is_empty:
            return np.zeros((t.hh, t.ww), bool)
        return features.rasterize([(g, 1)], out_shape=(t.hh, t.ww), transform=self.tile_transform(t),
                                  fill=0, dtype="uint8").astype(bool)

    def load_imagery(self, t, prefer=0):
        """Mosaic with swath `prefer` on top. Sets t.img, t.sw (1-based id, 0 = none), t.valid,
        t.alt (the other date where both exist, else NaN) and t.both."""
        k0 = prefer
        a, va = self.read_sw(k0, t.win)
        img, sw = a.copy(), np.where(va, k0 + 1, 0).astype(np.uint8)
        alt = np.full_like(a, np.nan)
        both = np.zeros_like(va)
        for k in range(len(self.SW)):
            if k == k0:
                continue
            b, vb = self.read_sw(k, t.win)
            fill = vb & (sw == 0)
            img[:, fill] = b[:, fill]
            sw[fill] = k + 1
            bb = vb & va
            alt[:, bb] = b[:, bb]
            both |= bb
        t.img, t.sw, t.alt, t.both = img, sw, alt, both
        t.aoi = self.aoi_mask(t)
        t.valid = t.aoi & (sw > 0)
        return t

    def block_ids(self, t):
        rows = t.r0 + np.arange(t.h); cols = t.c0 + np.arange(t.w)
        return ((rows // self.blk)[:, None] * self.ncb + (cols // self.blk)[None, :]).astype(np.int32)

    # ------------------------------------------------------------- footprints
    @property
    def bld(self):
        if self._bld is None:
            b = gpd.read_file(self.cfg.BUILD_SHP).to_crs(self.CRS)
            b = b[b.geometry.notna() & ~b.is_empty].copy()
            b["geometry"] = b.geometry.apply(make_valid)
            self._bld = b
            self._bsidx = b.sindex
            print(f"{len(b):,} footprints | {b.area.sum() / 1e4:.1f} ha | median {b.area.median():.0f} m2 | "
                  f"{100 * (b.area < self.RES ** 2 / 2).mean():.0f}% smaller than half a pixel")
        return self._bld

    def cov_window(self, t, shift=(0.0, 0.0), f=4):
        """Fraction of each pixel covered by footprints moved by `shift` (east, north) metres."""
        bld = self.bld
        dx, dy = shift
        tr = Affine.translation(-dx, -dy) * self.tile_transform(t)
        x0, y1 = tr * (0, 0); x1, y0 = tr * (t.ww, t.hh)
        hit = self._bsidx.query(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), predicate="intersects")
        if len(hit) == 0:
            return np.zeros((t.hh, t.ww), np.float32)
        fine = features.rasterize([(g, 1) for g in bld.geometry.iloc[hit]], out_shape=(t.hh * f, t.ww * f),
                                  transform=tr * Affine.scale(1 / f), fill=0, dtype="uint8")
        return fine.reshape(t.hh, f, t.ww, f).mean(axis=(1, 3)).astype(np.float32)

    def tile_cov(self, t):
        """Coverage using the footprint shift of whichever swath is shown at each pixel."""
        sw = np.zeros((t.hh, t.ww), np.uint8)
        for k in range(len(self.SW)):
            sw[(self.read_dn(k, t.win)[0] > 0) & (sw == 0)] = k + 1
        shifts = {k: self.shifts.get(k, (0.0, 0.0)) for k in range(1, len(self.SW) + 1)}
        present = [k for k in shifts if (sw == k).any()]
        if len(set(shifts[k] for k in present)) <= 1:
            return self.cov_window(t, shifts[present[0]] if present else shifts[1])
        out = np.zeros((t.hh, t.ww), np.float32)
        for k in present:
            out = np.where(sw == k, self.cov_window(t, shifts[k]), out)
        return out

    # ------------------------------------------------------------------ roads
    @property
    def roads(self):
        if self._roads is None:
            if not os.path.exists(self.cfg.ROAD_SHP):
                self._roads = gpd.GeoDataFrame(geometry=[], crs=self.CRS)
            else:
                r = gpd.read_file(self.cfg.ROAD_SHP).to_crs(self.CRS)
                r = r[r.geometry.notna() & ~r.is_empty].copy()
                self._roads = r.explode(index_parts=False).reset_index(drop=True)
            self._rsidx = self._roads.sindex if len(self._roads) else None
            print(f"roads: {len(self._roads)} line parts, {self._roads.length.sum() / 1000:.1f} km")
        return self._roads

    def road_dist(self, t, max_m=300.0):
        """Distance (m) from each pixel of the haloed tile to the nearest road line, capped at max_m."""
        roads = self.roads
        if self._rsidx is None:
            return np.full((t.hh, t.ww), max_m, np.float32)
        m = int(np.ceil(max_m / self.RES))
        hh, ww = t.hh + 2 * m, t.ww + 2 * m
        tr = self.MTR * Affine.translation(t.ch0 - m, t.rh0 - m)
        x0, y1 = tr * (0, 0); x1, y0 = tr * (ww, hh)
        hit = self._rsidx.query(box(x0, y0, x1, y1), predicate="intersects")
        if len(hit) == 0:
            return np.full((t.hh, t.ww), max_m, np.float32)
        ras = features.rasterize([(g, 1) for g in roads.geometry.iloc[hit]], out_shape=(hh, ww), transform=tr,
                                 fill=0, dtype="uint8", all_touched=True)
        d = distance_transform_edt(ras == 0) * self.RES
        return np.minimum(d, max_m)[m:m + t.hh, m:m + t.ww].astype(np.float32)

    # ====================================================== one-off estimation
    def _overlap_tiles(self, n=8, min_frac=0.9):
        """Tiles where swath 1 and swath 2 both cover at least min_frac of the tile (any land, not only AOI)."""
        scores = []
        for t in self.tiles(halo=0):
            a = self.read_dn(0, t.win)[0] > 0
            b = self.read_dn(1, t.win)[0] > 0
            scores.append(((a & b).mean(), t))
        scores.sort(key=lambda x: -x[0])
        return [t for s, t in scores[:n] if s >= min_frac]

    def coregister(self):
        """Sub-pixel phase correlation of swath 2 onto swath 1 (high-passed brightness)."""
        from skimage.registration import phase_cross_correlation
        if len(self.SW) < 2:
            return
        n2 = self.names[1]
        tiles = self._overlap_tiles()
        if not tiles:
            print("co-registration: overlap too small, skipped"); return
        est = []
        for t in tiles:
            a, va = self.read_sw(0, t.win, normalise=False)
            b, vb = self.read_sw(1, t.win, normalise=False)
            v = va & vb
            hp = lambda x: x - gaussian_filter(x, 4)
            A = hp(np.where(v, a.mean(0), a.mean(0)[v].mean()))
            B = hp(np.where(v, b.mean(0), b.mean(0)[v].mean()))
            sh, err, _ = phase_cross_correlation(A, B, upsample_factor=20, normalization=None)
            est.append((*sh, err))
        est = np.array(est)
        dy, dx = np.median(est[:, 0]), np.median(est[:, 1])
        spread = np.median(np.abs(est[:, :2] - [dy, dx]), axis=0)
        print("co-registration per tile (dy, dx px):\n", np.round(est[:, :2], 2))
        if max(abs(dy), abs(dx)) > self.cfg.COREG_MAX_PX or spread.max() > 0.5:
            print(f"-> estimate unstable (median {dy:+.2f},{dx:+.2f}, spread {spread.round(2)}) - not applied")
            return
        ody, odx = self.offsets[n2]
        self.set_offset(n2, ody + dy, odx + dx)
        print(f"-> swath 2 shifted by dy={dy:+.2f}, dx={dx:+.2f} px (total {self.offsets[n2]})")

    def fit_normalisation(self, rng, n_per_tile=30000):
        if len(self.SW) < 2 or not self.cfg.NORMALISE:
            return
        A, B = [], []
        for t in self.tiles(halo=0):
            a, va = self.read_sw(0, t.win, normalise=False)
            b, vb = self.read_sw(1, t.win, normalise=False)
            idx = np.flatnonzero((va & vb).ravel())
            if len(idx) == 0:
                continue
            idx = rng.choice(idx, min(len(idx), n_per_tile), replace=False)
            A.append(a.reshape(3, -1)[:, idx]); B.append(b.reshape(3, -1)[:, idx])
        if not A or sum(x.shape[1] for x in A) < 5000:
            print("normalisation: overlap too small, skipped"); return
        A, B = np.hstack(A), np.hstack(B)
        kind = self.cfg.NORMALISE
        coefs = rad.fit_pif(A, B) if kind == "pif" else rad.fit_quantile(A, B)
        self.norm[self.names[1]] = {"kind": kind, "coefs": coefs}
        Bn = rad.apply_norm(B[:, :, None].copy(), np.ones((B.shape[1], 1), bool), self.norm[self.names[1]])[:, :, 0]
        print(f"normalisation ({kind}) on {A.shape[1]:,} overlap px")
        print("  swath1 mean:", A.mean(1).round(4), "| swath2 before:", B.mean(1).round(4),
              "| after:", Bn.mean(1).round(4))
        if kind == "pif":
            print("  slope/intercept:", [(round(s, 3), round(c, 4)) for s, c in coefs])

    def _pick_tiles(self, k, n=6):
        bld = self.bld
        cen = bld.geometry.centroid
        cr = ((self.MTR.f - cen.y.values) / self.RES).astype(int) // self.cfg.TILE
        cc = ((cen.x.values - self.MTR.c) / self.RES).astype(int) // self.cfg.TILE
        cnt = pd.Series(list(zip(cr, cc))).value_counts()
        starts = {(a, b) for a, b, _, _ in self.TILES}
        out = []
        for (tr_, tc_), _ in cnt.items():
            r0, c0 = tr_ * self.cfg.TILE, tc_ * self.cfg.TILE
            if (r0, c0) not in starts:
                continue
            t = self.make_tile(r0, c0, min(self.cfg.TILE, self.H - r0), min(self.cfg.TILE, self.W - c0))
            am = self.aoi_mask(t)
            v = (self.read_dn(k, t.win)[0] > 0) & am
            if am.sum() > 0 and v.sum() / am.sum() > 0.4:
                out.append(t)
            if len(out) >= n:
                break
        return out

    def estimate_footprint_shift(self, k):
        """Integer search +/- MAX_SHIFT_PX, then sub-pixel refinement around the best integer shift.
        Score = how much brighter / less vegetated footprint pixels are than non-footprint pixels."""
        cfg, RES, M = self.cfg, self.RES, self.cfg.MAX_SHIFT_PX
        tiles = self._pick_tiles(k)
        if not tiles:
            print(f"swath {k + 1}: no suitable tiles for the alignment check"); return
        prep = []
        for t in tiles:
            a, v = self.read_sw(k, t.win)
            ok = v & self.aoi_mask(t)
            b = a.mean(0); nd = (a[2] - a[1]) / (a[2] + a[1] + 1e-6)
            z = lambda x: (x - x[v].mean()) / (x[v].std() + 1e-6)
            pm = z(b - uniform_filter(b, 21)) - z(nd - uniform_filter(nd, 21))
            prep.append((t, ok, pm))

        inner = (slice(M + 2, -M - 2), slice(M + 2, -M - 2))

        def score_cov(covs):
            s, used = 0.0, 0
            for (t, ok, pm), cov in zip(prep, covs):
                c, o, p = cov[inner], ok[inner], pm[inner]
                mk, bk = (c >= 0.5) & o, (c == 0) & o
                if mk.sum() >= 30 and bk.sum() >= 30:
                    s += p[mk].mean() - p[bk].mean(); used += 1
            return s / max(used, 1)

        def score(shift_m, f):
            return score_cov([self.cov_window(t, shift_m, f=f) for t, _, _ in prep])

        # integer stage: rasterize once per tile, then roll
        cov0 = [self.cov_window(t, (0.0, 0.0), f=2) for t, _, _ in prep]
        grid = {}
        for dy in range(-M, M + 1):
            for dx in range(-M, M + 1):
                grid[(dy, dx)] = score_cov([np.roll(c, (dy, dx), (0, 1)) for c in cov0])
        (by, bx), best = max(grid.items(), key=lambda kv: kv[1])
        # sub-pixel stage around the integer optimum
        st = cfg.SHIFT_STEP_PX
        fine = {}
        for fy in np.arange(by - 0.75, by + 0.75 + 1e-9, st):
            for fx in np.arange(bx - 0.75, bx + 0.75 + 1e-9, st):
                fine[(round(float(fy), 3), round(float(fx), 3))] = score((fx * RES, -fy * RES), 4)
        (fy, fx), fbest = max(fine.items(), key=lambda kv: kv[1])
        s0f = score((0.0, 0.0), 4)
        print(f"swath {k + 1}: score at 0 = {s0f:.3f}; integer best {best:.3f} at dx={bx:+d}, dy={by:+d}; "
              f"sub-pixel best {fbest:.3f} at dx={fx:+.2f}, dy={fy:+.2f} px")
        if (fx, fy) != (0, 0) and fbest - s0f >= cfg.SHIFT_MIN_GAIN:
            self.shifts[k + 1] = (float(fx * RES), float(-fy * RES))
            print(f"  -> footprints shifted {self.shifts[k + 1][0]:+.2f} m east, {self.shifts[k + 1][1]:+.2f} m north")
        else:
            self.shifts[k + 1] = (0.0, 0.0)
            print("  -> no significant shift")

    def prepare(self, rng):
        """Run all one-off estimations in the right order and save them."""
        if self.cfg.COREG:
            self.coregister()
        self.fit_normalisation(rng)
        if self.cfg.AUTO_SHIFT:
            for k in range(len(self.SW)):
                self.estimate_footprint_shift(k)
        self.save_state()
