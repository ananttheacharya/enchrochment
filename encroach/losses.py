"""Loss functions for multi-task building segmentation and coverage regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Tversky Loss for extreme class imbalance and small object detection (Salehi et al.).
    alpha=0.3, beta=0.7 penalizes False Negatives (missed buildings) more than False Positives.
    """
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets, valid_mask=None):
        probs = torch.sigmoid(logits)
        if valid_mask is not None:
            probs = probs[valid_mask]
            targets = targets[valid_mask]
            
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky


class MultiTaskLoss(nn.Module):
    """Combines:
    1. Tversky + BCE for Building Mask (with ignore mask)
    2. Smooth L1 (Huber) for Continuous Footprint Coverage Fraction
    3. BCE for Footprint Boundary Edges
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tversky = TverskyLoss(alpha=cfg.TVERSKY_ALPHA, beta=cfg.TVERSKY_BETA)
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.1)
        self.bce_bnd = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, batch):
        logits_mask = preds["mask"]
        pred_cov = preds["cov"]
        logits_bnd = preds["bnd"]
        
        true_mask = batch["mask"]
        true_cov = batch["coverage"]
        true_bnd = batch["boundary"]
        
        ign_cls = batch["ignore_cls"]
        ign_all = batch["ignore"]
        
        valid_cls = (~ign_cls)
        valid_all = (~ign_all)
        
        # 1. Mask Loss (Tversky + BCE)
        loss_tversky = self.tversky(logits_mask, true_mask, valid_cls)
        bce_map = self.bce_cls(logits_mask, true_mask)
        loss_bce = (bce_map * valid_cls.float()).sum() / (valid_cls.float().sum() + 1e-7)
        loss_mask = 0.5 * loss_tversky + 0.5 * loss_bce
        
        # 2. Coverage Regression Loss (Smooth L1)
        cov_l1 = self.smooth_l1(pred_cov, true_cov)
        loss_cov = (cov_l1 * valid_all.float()).sum() / (valid_all.float().sum() + 1e-7)
        
        # 3. Boundary Edge Loss
        bnd_bce = self.bce_bnd(logits_bnd, true_bnd)
        loss_bnd = (bnd_bce * valid_all.float()).sum() / (valid_all.float().sum() + 1e-7)
        
        total_loss = loss_mask + self.cfg.W_COVER * loss_cov + self.cfg.W_BOUNDARY * loss_bnd
        
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_mask": loss_mask.item(),
            "loss_cov": loss_cov.item(),
            "loss_bnd": loss_bnd.item(),
        }
