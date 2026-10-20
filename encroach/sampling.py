"""Pixel sampling for tabular models (GBM classifier and coverage regressor).

Extracts balanced training samples, hard negatives near buildings, mixed pixels for
continuous coverage regression, and held-out validation/test sets with spatial block IDs.
"""
import time
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation
from .features import make_features

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x


def sample_pixels(scene, split, cnt, rng):
    """Sample train/val/test pixels across the spatial blocks.

    Returns:
        dict containing:
            Xtr, ytr, cov_tr, bid_tr
            Xva, yva, cov_va, bid_va
            Xte, yte, cov_te, bid_te
            Xmix, cov_mix (for training coverage regressor)
    """
    cfg = scene.cfg
    cov_ds = rasterio.open(cfg.P("footprint_coverage.tif"))
    
    ps = lambda arr, s: arr[split == s].sum()
    pool = {
        "tr_pos": ps(cnt["ref"], 1),
        "tr_neg": ps(cnt["neg"], 1),
        "tr_nn":  ps(cnt["nn"], 1),
        "tr_mix": ps(cnt.get("mix", np.zeros_like(cnt["ref"])), 1),
        "va_pos": ps(cnt["ref"], 2),
        "va_neg": ps(cnt["neg"], 2) + ps(cnt["nn"], 2),
        "te_pos": ps(cnt["ref"], 3),
        "te_neg": ps(cnt["neg"], 3) + ps(cnt["nn"], 3),
    }
    
    n_tr_pos = min(cfg.N_TRAIN_POS, pool["tr_pos"])
    n_neg_tr = n_tr_pos * cfg.NEG_RATIO
    n_mix_tr = min(cfg.N_MIX, max(pool["tr_mix"], 1))
    
    prob_sel = {
        "tr_pos": n_tr_pos / max(pool["tr_pos"], 1),
        "tr_neg": min(1.0, n_neg_tr * (1.0 - cfg.HARD_NEG_SHARE) / max(pool["tr_neg"], 1)),
        "tr_nn":  min(1.0, n_neg_tr * cfg.HARD_NEG_SHARE / max(pool["tr_nn"], 1)),
        "tr_mix": min(1.0, n_mix_tr / max(pool["tr_mix"], 1)),
        "va_pos": min(1.0, cfg.N_EVAL_PER_CLASS / max(pool["va_pos"], 1)),
        "va_neg": min(1.0, cfg.N_EVAL_PER_CLASS / max(pool["va_neg"], 1)),
        "te_pos": min(1.0, cfg.N_EVAL_PER_CLASS / max(pool["te_pos"], 1)),
        "te_neg": min(1.0, cfg.N_EVAL_PER_CLASS / max(pool["te_neg"], 1)),
    }
    
    samples = []  # tuples of (split_id, label, coverage, block_id, feature_row)
    t_start = time.time()
    
    for t in tqdm(list(scene.tiles()), desc="sampling pixels"):
        t = scene.load_imagery(t)
        cov = cov_ds.read(1, window=t.win).astype(np.float32) / 100.0
        near = binary_dilation(cov > 0, iterations=cfg.BG_BUFFER_PX)[t.core]
        cc = cov[t.core]
        v = t.valid[t.core]
        bids = scene.block_ids(t)
        sets = split[bids]
        
        pos = v & (cc >= cfg.BUILD_MIN_COVER)
        far = v & ~near
        nn = v & near & (cc == 0)
        none = v & (cc == 0)
        mix = v & (cc > 0) & (cc < cfg.BUILD_MIN_COVER)
        
        u = rng.random((t.h, t.w))
        sel = {
            ("tr_pos", 1, 1): pos & (sets == 1) & (u < prob_sel["tr_pos"]),
            ("tr_neg", 1, 0): far & (sets == 1) & (u < prob_sel["tr_neg"]),
            ("tr_nn",  1, 0): nn  & (sets == 1) & (u < prob_sel["tr_nn"]),
            ("tr_mix", 1, 2): mix & (sets == 1) & (u < prob_sel["tr_mix"]),
            ("va_pos", 2, 1): pos & (sets == 2) & (u < prob_sel["va_pos"]),
            ("va_neg", 2, 0): none & (sets == 2) & (u < prob_sel["va_neg"]),
            ("te_pos", 3, 1): pos & (sets == 3) & (u < prob_sel["te_pos"]),
            ("te_neg", 3, 0): none & (sets == 3) & (u < prob_sel["te_neg"]),
        }
        
        if not any(m.any() for m in sel.values()):
            continue
            
        F = make_features(t, scene)
        for (_, split_id, lbl), m in sel.items():
            rr, cc_ = np.nonzero(m)
            if len(rr) == 0:
                continue
            X = F[:, rr + t.core[0].start, cc_ + t.core[1].start].T
            cov_vals = cc[rr, cc_]
            b_vals = bids[rr, cc_]
            y_vals = np.full(len(rr), lbl, dtype=np.int32)
            s_vals = np.full(len(rr), split_id, dtype=np.int32)
            samples.append((s_vals, y_vals, cov_vals, b_vals, X))
            
    cov_ds.close()
    print(f"pixel sampling completed in {time.time() - t_start:.1f}s")
    
    S_s = np.concatenate([s[0] for s in samples])
    S_y = np.concatenate([s[1] for s in samples])
    S_cov = np.concatenate([s[2] for s in samples])
    S_b = np.concatenate([s[3] for s in samples])
    S_X = np.vstack([s[4] for s in samples])
    
    # Train binary classification set (exclude mix class 2)
    m_tr = (S_s == 1) & (S_y != 2)
    m_va = (S_s == 2)
    m_te = (S_s == 3)
    m_mix = (S_s == 1)
    
    data = {
        "Xtr": S_X[m_tr], "ytr": S_y[m_tr], "cov_tr": S_cov[m_tr], "bid_tr": S_b[m_tr],
        "Xva": S_X[m_va], "yva": S_y[m_va], "cov_va": S_cov[m_va], "bid_va": S_b[m_va],
        "Xte": S_X[m_te], "yte": S_y[m_te], "cov_te": S_cov[m_te], "bid_te": S_b[m_te],
        "Xmix": S_X[m_mix], "cov_mix": S_cov[m_mix],
    }
    
    np.savez_compressed(cfg.P("sampled_tabular_data.npz"), **data)
    print(f"Train: {len(data['ytr']):,} samples ({data['ytr'].sum():,} building)")
    print(f"Val:   {len(data['yva']):,} samples ({data['yva'].sum():,} building)")
    print(f"Test:  {len(data['yte']):,} samples ({data['yte'].sum():,} building)")
    print(f"Mix/Reg: {len(data['cov_mix']):,} continuous coverage samples")
    return data
