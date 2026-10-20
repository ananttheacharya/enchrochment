# LISS-4 Building & Encroachment Detection Pipeline

An end-to-end remote sensing and deep learning pipeline for high-precision building footprint classification and road encroachment reporting from IRS Resourcesat-2 LISS-4 multispectral satellite imagery.

---

## 📌 Overview

- **Sensors & Data**: IRS Resourcesat-2 LISS-4 multispectral bands (Green: B2, Red: B3, NIR: B4 at ~5m resolution) across multi-temporal swaths.
- **Goal**: High-accuracy building segmentation and buffer-based encroachment detection along road networks.
- **Models**:
  - LightGBM baseline with spectral indices (NDVI, SAVI, NDWI, NDBI), textural features (GLCM, variance, gradient), and shadow morphology.
  - Deep Learning: Multi-scale U-Net / Attention architectures with focal-dice loss for dense footprint extraction.

---

## 🗂️ Repository Structure

```text
├── encroach/                         # Core modular Python package
│   ├── config.py                     # Central configuration & dynamic path resolution
│   ├── radiometry.py                 # DN to TOA reflectance & PIF normalization
│   ├── scene.py                      # Multi-swath mosaic and tile handling
│   ├── labels.py                     # Footprint alignment & rasterization
│   ├── features.py                   # Multi-spectral & spatial feature extraction
│   ├── sampling.py                   # Hard-negative and spatial-balanced sampling
│   ├── gbm.py                        # LightGBM pipeline
│   ├── data.py                       # PyTorch dataset & augmentations
│   ├── models.py                     # U-Net & deep segmentation architectures
│   ├── losses.py                     # Combined BCE, Dice, and Focal loss functions
│   ├── predict.py                    # Tiled inference with overlap smoothing
│   ├── metrics.py                    # Precision, Recall, F1, and IoU evaluation
│   └── encroachment.py               # Buffer-based road encroachment reporting
├── 01_preprocess_and_gbm_v3.ipynb    # Step 1: Preprocessing & LightGBM Baseline
├── 02_make_patches.ipynb             # Step 2: Patch extraction for Deep Learning
├── 03_train_unet.ipynb               # Step 3: PyTorch U-Net training (GPU)
├── 04_predict_and_encroachment_report.ipynb # Step 4: Full-scene inference & encroachment analytics
├── layers/                           # Vector reference layers (AOI and Building shapefiles)
├── Road_line/                        # Road network centerlines
├── COLAB_SETUP_GUIDE.md              # Complete guide for running on Google Colab Pro
├── requirements.txt                  # Python dependencies
└── .gitignore
```

---

## 🚀 Quickstart

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Inspect or adapt settings in `encroach/config.py`:
   ```python
   from encroach.config import Config
   cfg = Config()
   ```

3. **Running the Pipeline**:
   - Follow the numbered notebooks sequentially from `01` to `04`.
   - For Google Colab Pro GPU execution, refer to [COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md).
