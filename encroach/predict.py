"""Full-tile inference engine with 8-way TTA, Gaussian window blending, and ensembling."""
import os
import time
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
import torch
import joblib
from .features import dl_channels, make_features
from .gbm import HAS_LGB

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x


def _make_gaussian_weight(size):
    """2D Gaussian window to smoothly blend overlapping inference tiles."""
    y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    d = np.sqrt(x * x + y * y)
    sigma = 0.5
    w = np.exp(-(d * d) / (2.0 * sigma * sigma))
    return w.astype(np.float32)


def predict_patch_dl(model, x_tensor, device, tta=True):
    """Predict single batch with optional 8-fold test-time augmentation (D4)."""
    model.eval()
    with torch.no_grad():
        x = x_tensor.to(device)
        if not tta:
            out = model(x)
            prob = torch.sigmoid(out["mask"]).squeeze(1).cpu().numpy()
            cov = out["cov"].squeeze(1).cpu().numpy()
            return prob, cov
            
        probs = []
        covs = []
        # 8 transforms: 4 rotations x 2 flips
        for flip in [False, True]:
            x_f = torch.flip(x, dims=[3]) if flip else x
            for rot in range(4):
                x_r = torch.rot90(x_f, k=rot, dims=[2, 3])
                out = model(x_r)
                p = torch.sigmoid(out["mask"])
                c = out["cov"]
                # Invert rotation & flip
                p_inv = torch.rot90(p, k=(4 - rot) % 4, dims=[2, 3])
                c_inv = torch.rot90(c, k=(4 - rot) % 4, dims=[2, 3])
                if flip:
                    p_inv = torch.flip(p_inv, dims=[3])
                    c_inv = torch.flip(c_inv, dims=[3])
                probs.append(p_inv.squeeze(1).cpu().numpy())
                covs.append(c_inv.squeeze(1).cpu().numpy())
                
        return np.mean(probs, axis=0), np.mean(covs, axis=0)


def predict_full_scene(scene, dl_model, device, gbm_model=None, ensemble_weight=0.6):
    """Run tiled inference over the AOI and save probability and coverage rasters."""
    cfg = scene.cfg
    P_SIZE = cfg.PATCH
    STRIDE = P_SIZE // 2
    G_WEIGHT = _make_gaussian_weight(P_SIZE)
    
    prof_f = dict(
        driver="GTiff", height=scene.H, width=scene.W, count=1, dtype="float32",
        crs=scene.CRS, transform=scene.MTR, tiled=True, blockxsize=512, blockysize=512,
        compress="deflate", nodata=-1.0, predictor=3
    )
    
    t0 = time.time()
    with rasterio.open(cfg.P("building_probability.tif"), "w", **prof_f) as dp, \
         rasterio.open(cfg.P("building_coverage.tif"), "w", **prof_f) as dc:
         
        for t in tqdm(list(scene.tiles(halo=cfg.DL_HALO)), desc="full-scene inference"):
            t = scene.load_imagery(t)
            v = t.valid[t.core]
            if not v.any():
                dp.write(np.full((t.h, t.w), -1.0, np.float32), 1, window=t.cwin)
                dc.write(np.full((t.h, t.w), -1.0, np.float32), 1, window=t.cwin)
                continue
                
            X_dl = dl_channels(t, scene)  # (C, hh, ww)
            hh, ww = t.hh, t.ww
            
            prob_acc = np.zeros((hh, ww), np.float32)
            cov_acc = np.zeros((hh, ww), np.float32)
            w_acc = np.zeros((hh, ww), np.float32)
            
            # Sliding window with 50% overlap and Gaussian blend
            for r in range(0, hh - P_SIZE + 1, STRIDE):
                for c in range(0, ww - P_SIZE + 1, STRIDE):
                    sub_x = torch.from_numpy(X_dl[:, r:r + P_SIZE, c:c + P_SIZE]).unsqueeze(0).float()
                    p_sub, c_sub = predict_patch_dl(dl_model, sub_x, device, tta=cfg.TTA)
                    
                    prob_acc[r:r + P_SIZE, c:c + P_SIZE] += p_sub[0] * G_WEIGHT
                    cov_acc[r:r + P_SIZE, c:c + P_SIZE] += c_sub[0] * G_WEIGHT
                    w_acc[r:r + P_SIZE, c:c + P_SIZE] += G_WEIGHT
                    
            nz = w_acc > 0
            prob_acc[nz] /= w_acc[nz]
            cov_acc[nz] /= w_acc[nz]
            
            prob_core = prob_acc[t.core]
            cov_core = cov_acc[t.core]
            
            # Optional Ensembling with LightGBM
            if gbm_model is not None:
                F = make_features(t, scene)
                Fc = F[:, t.core[0], t.core[1]]
                gbm_prob = np.zeros((t.h, t.w), np.float32)
                gbm_prob[v] = gbm_model.predict_proba(Fc[:, v].T)[:, 1]
                prob_core = ensemble_weight * prob_core + (1.0 - ensemble_weight) * gbm_prob
                
            final_prob = np.where(v, prob_core, -1.0)
            final_cov = np.where(v, cov_core, -1.0)
            
            dp.write(final_prob.astype(np.float32), 1, window=t.cwin)
            dc.write(final_cov.astype(np.float32), 1, window=t.cwin)
            
    print(f"Full-scene prediction finished in {time.time() - t0:.1f}s")
