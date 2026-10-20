"""Deep Learning dataset and patch extraction.

Exports sliding-window 256x256 image patches with multi-task targets:
- Image: C channels (Green, Red, NIR, NDVI, MBI, Road Distance, Swath ID)
- Coverage: Continuous fraction of pixel covered by building footprint [0.0 - 1.0]
- Mask: Binary building classification (coverage >= BUILD_MIN_COVER)
- Boundary: Footprint boundary edges for boundary-aware loss
- Ignore: Pixels outside the AOI or invalid swath data
"""
import os
import glob
import time
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .features import dl_channels

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x


def extract_patches(scene, split, out_dir):
    """Slice the scene into overlapping 256x256 multi-task patches and save to disk."""
    cfg = scene.cfg
    os.makedirs(out_dir, exist_ok=True)
    P_SIZE = cfg.PATCH
    STRIDE = cfg.PATCH_STRIDE
    cov_ds = rasterio.open(cfg.P("footprint_coverage.tif"))
    
    saved = {"train": 0, "val": 0, "test": 0}
    t0 = time.time()
    
    for t in tqdm(list(scene.tiles(halo=P_SIZE // 2)), desc="extracting patches"):
        t = scene.load_imagery(t)
        cov_full = cov_ds.read(1, window=t.win).astype(np.float32) / 100.0
        X_full = dl_channels(t, scene)  # shape (C, hh, ww)
        bids_full = scene.block_ids(scene.make_tile(t.r0, t.c0, t.h, t.w, halo=P_SIZE // 2))
        
        # Slide window across the core area
        for r_rel in range(0, t.h, STRIDE):
            for c_rel in range(0, t.w, STRIDE):
                # Coordinates in haloed tile
                r_start = t.core[0].start + r_rel
                c_start = t.core[1].start + c_rel
                
                # Check bounds
                if r_start + P_SIZE > t.hh or c_start + P_SIZE > t.ww:
                    continue
                    
                sub_v = t.valid[r_start:r_start + P_SIZE, c_start:c_start + P_SIZE]
                if sub_v.mean() < cfg.MIN_VALID_FRAC:
                    continue
                    
                sub_X = X_full[:, r_start:r_start + P_SIZE, c_start:c_start + P_SIZE]
                sub_cov = cov_full[r_start:r_start + P_SIZE, c_start:c_start + P_SIZE]
                sub_mask = (sub_cov >= cfg.BUILD_MIN_COVER).astype(np.uint8)
                
                # Compute boundary mask: dilation - erosion of footprints
                bin_bld = sub_cov > 0
                boundary = (binary_dilation(bin_bld, iterations=1) ^ binary_erosion(bin_bld, iterations=1)).astype(np.uint8)
                
                # Ignore mask: invalid imagery OR outside AOI
                ignore = (~sub_v).astype(np.uint8)
                if cfg.IGNORE_MIXED:
                    # optionally ignore ambiguous 0 < cov < 0.5 pixels for classification loss
                    ignore_cls = ignore | ((sub_cov > 0) & (sub_cov < cfg.BUILD_MIN_COVER)).astype(np.uint8)
                else:
                    ignore_cls = ignore
                    
                # Determine spatial split from block IDs
                b_center = bids_full[r_start + P_SIZE // 2, c_start + P_SIZE // 2]
                split_id = split[b_center]
                if split_id == 0:
                    continue
                    
                split_name = {1: "train", 2: "val", 3: "test"}[split_id]
                has_bld = int(sub_mask.sum() > 0)
                
                patch_id = f"p_{split_name}_{t.r0 + r_rel}_{t.c0 + c_rel}_bld{has_bld}.npz"
                fpath = os.path.join(out_dir, patch_id)
                
                np.savez_compressed(
                    fpath,
                    x=sub_X,
                    cov=sub_cov,
                    mask=sub_mask,
                    boundary=boundary,
                    ignore=ignore,
                    ignore_cls=ignore_cls,
                    split=split_id,
                    has_bld=has_bld,
                )
                saved[split_name] += 1

    cov_ds.close()
    print(f"Patches extracted in {time.time() - t0:.1f}s: {saved}")
    return saved


class EncroachmentPatchDataset(Dataset):
    """Loads extracted patches from disk with online augmentation."""
    def __init__(self, patch_dir, split_id, augment=False, upsample_2x=False):
        self.split_id = split_id
        self.augment = augment
        self.upsample_2x = upsample_2x
        split_name = {1: "train", 2: "val", 3: "test"}[split_id]
        self.files = sorted(glob.glob(os.path.join(patch_dir, f"p_{split_name}_*.npz")))
        self.has_bld = np.array(["_bld1" in f for f in self.files], dtype=int)
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = data["x"]                 # (C, H, W) float32
        cov = data["cov"]             # (H, W) float32
        mask = data["mask"]           # (H, W) uint8
        boundary = data["boundary"]   # (H, W) uint8
        ignore = data["ignore"]       # (H, W) uint8
        ignore_cls = data["ignore_cls"]
        
        # Spatial Augmentations (D4 dihedral group)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2)
                cov = np.flip(cov, axis=1)
                mask = np.flip(mask, axis=1)
                boundary = np.flip(boundary, axis=1)
                ignore = np.flip(ignore, axis=1)
                ignore_cls = np.flip(ignore_cls, axis=1)
            # Random vertical flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1)
                cov = np.flip(cov, axis=0)
                mask = np.flip(mask, axis=0)
                boundary = np.flip(boundary, axis=0)
                ignore = np.flip(ignore, axis=0)
                ignore_cls = np.flip(ignore_cls, axis=0)
            # Random 90-degree rotations
            k = np.random.randint(0, 4)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2))
                cov = np.rot90(cov, k, axes=(0, 1))
                mask = np.rot90(mask, k, axes=(0, 1))
                boundary = np.rot90(boundary, k, axes=(0, 1))
                ignore = np.rot90(ignore, k, axes=(0, 1))
                ignore_cls = np.rot90(ignore_cls, k, axes=(0, 1))
            # Photometric jitter on optical bands (first 3 channels: Green, Red, NIR)
            gain = np.random.uniform(0.90, 1.10, size=(3, 1, 1)).astype(np.float32)
            bias = np.random.uniform(-0.02, 0.02, size=(3, 1, 1)).astype(np.float32)
            x[:3] = np.clip(x[:3] * gain + bias, 0.0, 1.0)
            
        t_x = torch.from_numpy(np.ascontiguousarray(x)).float()
        t_cov = torch.from_numpy(np.ascontiguousarray(cov)).float().unsqueeze(0)
        t_mask = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        t_bnd = torch.from_numpy(np.ascontiguousarray(boundary)).float().unsqueeze(0)
        t_ign = torch.from_numpy(np.ascontiguousarray(ignore)).bool().unsqueeze(0)
        t_ign_cls = torch.from_numpy(np.ascontiguousarray(ignore_cls)).bool().unsqueeze(0)
        
        if self.upsample_2x:
            # 2x super-resolution training mode
            t_x = torch.nn.functional.interpolate(t_x.unsqueeze(0), scale_factor=2, mode="bicubic", align_corners=False).squeeze(0)
            t_cov = torch.nn.functional.interpolate(t_cov.unsqueeze(0), scale_factor=2, mode="bilinear", align_corners=False).squeeze(0)
            t_mask = torch.nn.functional.interpolate(t_mask.unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0)
            t_bnd = torch.nn.functional.interpolate(t_bnd.unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0)
            t_ign = torch.nn.functional.interpolate(t_ign.float().unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0).bool()
            t_ign_cls = torch.nn.functional.interpolate(t_ign_cls.float().unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0).bool()
            
        return {
            "image": t_x,
            "coverage": t_cov,
            "mask": t_mask,
            "boundary": t_bnd,
            "ignore": t_ign,
            "ignore_cls": t_ign_cls,
        }


def make_dataloaders(patch_dir, batch_size=16, pos_share=0.7, upsample_2x=False, num_workers=2):
    """Build train and validation dataloaders with positive-patch oversampling."""
    ds_tr = EncroachmentPatchDataset(patch_dir, split_id=1, augment=True, upsample_2x=upsample_2x)
    ds_va = EncroachmentPatchDataset(patch_dir, split_id=2, augment=False, upsample_2x=upsample_2x)
    
    # Balanced sampling weights
    weights = np.where(ds_tr.has_bld == 1, pos_share / max(ds_tr.has_bld.sum(), 1), (1.0 - pos_share) / max((ds_tr.has_bld == 0).sum(), 1))
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds_tr), replacement=True)
    
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader_tr, loader_va
