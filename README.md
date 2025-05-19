# 📦 UAV Crop Classification using Patch-Based Random Forest

A machine learning pipeline for classifying crop types from UAV multispectral imagery using patch-level analysis and Random Forest classifier.

## 🚀 Features Implemented

* ✅ Patch extraction from orthomosaic TIFFs using polygon-labeled shapefile
* ✅ Supports **multi-size patch testing**: `128x128`, `256x256`, and non-square `242x272`
* ✅ Filters mixed-label patches to ensure clean supervision
* ✅ Feature extraction per patch:

  * Mean / Std / Max of 10 spectral bands
  * NDVI & GNDVI vegetation indices
* ✅ Class balancing with **SMOTE** (only when safe)
* ✅ RandomForestClassifier with `class_weight="balanced"`
* ✅ Train-test split with stratification (80/20)
* ✅ Accuracy reporting + `classification_report`
* ✅ Automatic confusion matrix PNG export per patch size
* ✅ Accuracy comparison bar chart: `patch_size_accuracy_bar.png`

---

## 🧾 Required Inputs

* **TIFF images**: UAV-captured multispectral reflectance orthophotos
* **Shapefile** (`.shp`, `.shx`, `.dbf`, etc.): contains crop-type polygons with field `crop`
* **Folder structure**:

  ```
  Project/
  ├── ms data/
  │   ├── UAV3-MS/          # Contains .tif images
  │   └── metadata/
  │       └── md_FieldSHP/  # Contains shapefile components
  ```

---

## 📂 Output

After running the script, you will get:

* `confusion_matrix_<WxH>.png` for each patch size
* `patch_predictions_<WxH>.csv` with per-patch prediction results
* `patch_size_accuracy_bar.png` comparing different patch sizes

---

## 🧠 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python patch_rf_classifier_multi.py
```

---

## 🔬 Next Suggestions (Optional Enhancements)

* Export each patch as `.png` for manual review
* Add GLCM-based texture features (contrast, homogeneity)
* Add LightGBM or CNN/Vision Transformer comparison
* Add time-aware modeling if image timestamps are available

---

## 📌 Authors / Group

* ✍️ Code: Marius (with ChatGPT assist)
* 📷 Data: UAV team
* 🧠 Model logic: ML research group

---

## 📄 License

MIT or Creative Commons (customizable)

Ready to push to GitHub and share with your group! 🚀

