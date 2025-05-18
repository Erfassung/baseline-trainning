# 🧠 NDVI-Based Crop Classification Pipeline

This repository contains a full pipeline for training a crop classification model using NDVI-based image patches. It includes patch extraction, label filtering based on purity, metadata generation, and final model training and evaluation with PDF reporting.

---

## 📁 Project Structure

```
project_ms/
├── images/                        # Raw NDVI GeoTIFF files (multiple tiles)
├── label.tif                      # Rasterized label map (plot_IDs from shapefile)
├── md_FieldSHP.shp/.dbf/.shx      # Plot polygon shapefile with crop metadata
├── output/                        # Output patches and metadata
├── cut_and_check_patches.py      # Patch slicing tool with purity check
├── prepare_training_data.py      # Merge all metadata.csv files into one
├── train_model_from_merged.py    # Model training and PDF report generation
```

---

## 🚀 Step-by-Step Usage

### 1. Patch Extraction from NDVI Images

```bash
python cut_and_check_patches.py \
  --image E:/Erfassung/project_ms/images/your_tile.tif \
  --label E:/Erfassung/project_ms/label.tif \
  --output E:/Erfassung/project_ms/output \
  --patch_height 242 --patch_width 272 \
  --stride_y 121 --stride_x 136
```

Repeat this for each NDVI tile. Each run generates:

* `/output/images/*.tif`
* `/output/labels/*.tif`
* `/output/metadata.csv`

Only patches with dominant labels and at least 90% purity are retained.

---

### 2. Merge Metadata

```bash
python prepare_training_data.py
```

* Finds all metadata.csv files
* Merges them into one: `merged_metadata.csv`
* Generates a crop class bar plot: `label_distribution.png`

---

### 3. Train Model and Generate PDF Report

```bash
python train_model_from_merged.py
```

This script will:

* Read `merged_metadata.csv`
* Extract NDVI statistics for each patch
* Use manually defined plot\_ID → crop mapping
* Train a RandomForest classifier
* Output:

  * `crop_class_distribution.png`
  * `classification_report_summary.pdf`
  * Console classification report + confusion matrix

---

## 📝 PDF Report (What to Share)

The PDF contains:

* Bar chart of class distribution
* Detailed precision/recall/F1-score per crop
* Confusion matrix in readable format

Useful for documentation and team presentations.

---

## 💡 Notes

* Crop label mapping is hardcoded (edit `id_to_crop` in `train_model_from_merged.py`)
* You can extend this to include Genotype, Area, etc. from the shapefile.
* If needed, change the patch size or stride for finer spatial sampling.

---

## 🙋 For Questions

Contact: \[your name / email] or leave a GitHub issue.

---

Happy patching & modeling 🌱
