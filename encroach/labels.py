"""Pass 1 (footprint coverage raster + block statistics) and the spatial train / validation / test split."""
import time
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **k: x


def profile_u8(scene, **kw):
    p = dict(driver="GTiff", height=scene.H, width=scene.W, count=1, dtype="uint8", crs=scene.CRS,
             transform=scene.MTR, tiled=True, blockxsize=512, blockysize=512, compress="deflate", nodata=255)
    p.update(kw)
    return p


def coverage_pass(scene):
    """Write footprint_coverage.tif (0-100 %) and return per-block pixel counts."""
    cfg = scene.cfg
    keys = ("valid", "ref", "neg", "nn", "mix")
    cnt = {k: np.zeros(scene.NB, np.int64) for k in keys}
    t0 = time.time()
    with rasterio.open(cfg.P("footprint_coverage.tif"), "w", **profile_u8(scene)) as dst:
        for t in tqdm(list(scene.tiles()), desc="pass 1 (coverage)"):
            t.aoi = scene.aoi_mask(t)
            cov = scene.tile_cov(t)
            dst.write(np.round(cov[t.core] * 100).astype(np.uint8), 1, window=t.cwin)
            near = binary_dilation(cov > 0, iterations=cfg.BG_BUFFER_PX)[t.core]
            a = t.aoi[t.core]; cc = cov[t.core]; bid = scene.block_ids(t)
            for key, m in (("valid", a), ("ref", a & (cc >= cfg.BUILD_MIN_COVER)), ("neg", a & ~near),
                           ("nn", a & near & (cc == 0)), ("mix", a & (cc > 0) & (cc < cfg.BUILD_MIN_COVER))):
                cnt[key] += np.bincount(bid[m], minlength=scene.NB)
    print(f"pass 1 done in {time.time() - t0:.0f}s | AOI px {cnt['valid'].sum():,} | building px "
          f"{cnt['ref'].sum():,} ({100 * cnt['ref'].sum() / max(cnt['valid'].sum(), 1):.2f}%)")
    np.savez(cfg.P("block_counts.npz"), **cnt)
    return cnt


def spatial_split(scene, cnt, rng):
    """0 = unused, 1 = train, 2 = validation, 3 = test. Whole blocks go to one set."""
    cfg = scene.cfg
    nv, nr = cnt["valid"], cnt["ref"]
    cand = np.flatnonzero(nv > 0)
    tot_v, tot_r = nv.sum(), nr.sum()

    def fill(order, frac):
        take, acc = [], 0
        for b in order:
            take.append(b); acc += nv[b]
            if acc >= frac * tot_v:
                break
        return np.array(take)

    for _ in range(2000):
        test = fill(rng.permutation(cand), cfg.TEST_FRACTION)
        val = fill(rng.permutation(np.setdiff1d(cand, test)), cfg.VAL_FRACTION)
        te, va = nr[test].sum(), nr[val].sum()
        if te >= max(300, 0.15 * tot_r) and va >= max(300, 0.07 * tot_r) and tot_r - te - va >= 0.3 * tot_r:
            blk = np.zeros(scene.NB, np.uint8)
            blk[cand] = 1; blk[val] = 2; blk[test] = 3
            break
    else:
        raise RuntimeError("Could not split blocks with enough buildings in each part. Reduce BLOCK_M.")
    for s, nm in ((1, "train"), (2, "validation"), (3, "test")):
        m = blk == s
        print(f"{nm:10s}: {m.sum():4d} blocks | {nv[m].sum():>10,} px | {nr[m].sum():>8,} building px")
    np.save(cfg.P("block_split.npy"), blk)
    return blk


def load_split(cfg):
    return np.load(cfg.P("block_split.npy")), dict(np.load(cfg.P("block_counts.npz")))
