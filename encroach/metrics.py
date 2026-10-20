"""Comprehensive accuracy metrics and validation assessment."""
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, label
from sklearn.metrics import confusion_matrix

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x


def calculate_metrics(tp, fp, fn, tn, title=""):
    n = tp + fp + fn + tn
    oa = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    kappa = (oa - pe) / max(1.0 - pe, 1e-9) if pe < 1.0 else np.nan
    pa_b = tp / max(tp + fn, 1)      # Producer's accuracy / Recall
    ua_b = tp / max(tp + fp, 1)      # User's accuracy / Precision
    pa_n = tn / max(tn + fp, 1)
    ua_n = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    
    res = {
        "set": title, "N": int(n), "OA": round(oa, 4), "kappa": round(kappa, 4),
        "PA_building": round(pa_b, 4), "UA_building": round(ua_b, 4),
        "PA_nonbuilding": round(pa_n, 4), "UA_nonbuilding": round(ua_n, 4),
        "F1_building": round(f1, 4), "IoU_building": round(iou, 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    return res


def evaluate_test_blocks(scene, threshold, hysteresis=False, low_frac=0.6):
    """Assess strict, mixed-excluded, 1-pixel tolerant, and per-building detection on held-out test blocks."""
    cfg = scene.cfg
    split = np.load(cfg.P("block_split.npy"))
    prob_ds = rasterio.open(cfg.P("building_probability.tif"))
    cov_ds = rasterio.open(cfg.P("footprint_coverage.tif"))
    
    C = dict(tp=0, fp=0, fn=0, tn=0, tp_x=0, fp_x=0, fn_x=0, tn_x=0,
             rel_prec_num=0, rel_prec_den=0, rel_rec_num=0, rel_rec_den=0)
    OBJ_BINS = [2, 4, 9]
    obj_tot = np.zeros(4, np.int64)
    obj_hit = np.zeros(4, np.int64)
    S8 = np.ones((3, 3), bool)
    
    prof_c = dict(
        driver="GTiff", height=scene.H, width=scene.W, count=1, dtype="uint8",
        crs=scene.CRS, transform=scene.MTR, tiled=True, blockxsize=512, blockysize=512,
        compress="deflate", nodata=255
    )
    
    with rasterio.open(cfg.P("classification.tif"), "w", **prof_c) as dc:
        for t in tqdm(list(scene.tiles()), desc="evaluating test blocks"):
            ph = prob_ds.read(1, window=t.win)
            ch = cov_ds.read(1, window=t.win).astype(np.float32) / 100.0
            
            if hysteresis:
                # Seed with high threshold, expand to low threshold
                seed = ph >= threshold
                allow = ph >= (threshold * low_frac)
                pred_h, _ = label(allow, structure=S8)
                seeds_in_components = np.unique(pred_h[seed])
                pred_h = np.isin(pred_h, seeds_in_components[seeds_in_components > 0])
            else:
                pred_h = ph >= threshold
                
            ref_h = ch >= cfg.BUILD_MIN_COVER
            pc = ph[t.core]
            valid = pc >= 0
            pred = pred_h[t.core]
            ref = ref_h[t.core]
            cov = ch[t.core]
            
            dc.write(np.where(valid, pred.astype(np.uint8), 255), 1, window=t.cwin)
            
            test = valid & (split[scene.block_ids(t)] == 3)
            if not test.any():
                continue
                
            C["tp"] += int((pred & ref & test).sum())
            C["fp"] += int((pred & ~ref & test).sum())
            C["fn"] += int((~pred & ref & test).sum())
            C["tn"] += int((~pred & ~ref & test).sum())
            
            keep = test & ~((cov > 0) & (cov < cfg.BUILD_MIN_COVER))
            C["tp_x"] += int((pred & ref & keep).sum())
            C["fp_x"] += int((pred & ~ref & keep).sum())
            C["fn_x"] += int((~pred & ref & keep).sum())
            C["tn_x"] += int((~pred & ~ref & keep).sum())
            
            pred_d = binary_dilation(pred_h, S8)[t.core]
            ref_d = binary_dilation(ref_h, S8)[t.core]
            C["rel_prec_num"] += int((pred & ref_d & test).sum())
            C["rel_prec_den"] += int((pred & test).sum())
            C["rel_rec_num"] += int((ref & pred_d & test).sum())
            C["rel_rec_den"] += int((ref & test).sum())
            
            lab, nl = label(cov > 0, structure=S8)
            if nl:
                idx = np.arange(1, nl + 1)
                in_test = ndi.maximum(test.astype(np.uint8), lab, idx) > 0
                hit = ndi.maximum(pred_d.astype(np.uint8), lab, idx) > 0
                area = np.bincount(lab.ravel())[1:]
                cls_ = np.digitize(area, OBJ_BINS)
                for c_ in range(4):
                    m = in_test & (cls_ == c_)
                    obj_tot[c_] += m.sum()
                    obj_hit[c_] += (m & hit).sum()
                    
    prob_ds.close()
    cov_ds.close()
    
    rows = []
    rows.append(calculate_metrics(C["tp"], C["fp"], C["fn"], C["tn"], "Wall-to-wall Test Blocks (Strict)"))
    rows.append(calculate_metrics(C["tp_x"], C["fp_x"], C["fn_x"], C["tn_x"], "Mixed Pixels Excluded"))
    
    rp = C["rel_prec_num"] / max(C["rel_prec_den"], 1)
    rr = C["rel_rec_num"] / max(C["rel_rec_den"], 1)
    rf1 = 2 * rp * rr / max(rp + rr, 1e-9)
    rows.append({
        "set": "1-pixel Tolerant (Spatial buffer)",
        "UA_building": round(rp, 4), "PA_building": round(rr, 4), "F1_building": round(rf1, 4),
    })
    
    det_rate = obj_hit.sum() / max(obj_tot.sum(), 1)
    rows.append({
        "set": "Per-building Object Detection (All sizes)",
        "PA_building": round(det_rate, 4), "N": int(obj_tot.sum()),
    })
    
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(cfg.P("test_block_accuracy_report.csv"), index=False)
    
    det_bins = pd.DataFrame({
        "object_size": ["1 px", "2-3 px", "4-8 px", ">=9 px"],
        "total_reference": obj_tot,
        "detected": obj_hit,
        "detection_rate": np.round(obj_hit / np.maximum(obj_tot, 1), 3),
    })
    det_bins.to_csv(cfg.P("per_building_detection_bins.csv"), index=False)
    
    print("\n" + "=" * 70)
    print("TEST BLOCK ACCURACY SUMMARY (Target ~0.85):")
    print("=" * 70)
    print(df_metrics.to_string())
    print("\nObject Detection by Size:")
    print(det_bins.to_string())
    return df_metrics, det_bins
