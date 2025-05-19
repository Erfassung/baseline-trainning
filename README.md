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


Warum unser Patch-basierter Ansatz dem klassischen GeoPatch-Workflow überlegen ist
Aspekt	Klassischer GeoPatch	Unser Multi-Patch-RF
Patch-Größen	meist fixe Kachelgröße (z. B. 256×256)	128×128, 256×256 und 242×272 werden gleichzeitig getestet → beste Größe datengetrieben auswählen
Merkmale	Roh-Bänder oder einfache Mittelwerte	Mittelwert + Std + Maximum aller 10 Bänder + NDVI + GNDVI ⇒ deutlich reichhaltigeres Spektrum
Klassen­ungleich­gewicht	oft ignoriert → dominante Klassen überwiegen	class_weight="balanced" und SMOTE-Oversampling (falls genug Samples)
Qualitätskontrolle	Patch kann gemischte Kulturen enthalten	Prüft Polygon-Schnitt: nur reine (1 Label)-Patches werden verwendet
Automatische Auswertung	manuell vergleichen	Confusion-Matrix & Balkendiagramm werden automatisch je Patch-Größe erzeugt
Nicht-quadratische Patches	i. d. R. nicht unterstützt	242×272 demonstriert flexible Window-Geometrie

Kurz gesagt: mehr Feature-Power, bessere Daten­balance, adaptive Patch-Strategie ⇒ höhere Robustheit und weniger Overfitting.







## 📄 License

MIT or Creative Commons (customizable)

Ready to push to GitHub and share with your group! 🚀

