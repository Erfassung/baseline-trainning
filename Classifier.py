# Einheitliches Skript: Unterstützt den Vergleich der Klassifizierung von quadratischen und nicht-quadratischen Patches (128x128, 256x256, 242x272)

# ========= Bibliotheken importieren =========
# Importieren von Bibliotheken für die Verarbeitung von Rasterdaten, Geodaten und maschinellem Lernen
import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import rcParams

# Konfigurieren Sie matplotlib für die Unterstützung mehrerer Sprachen
rcParams['font.sans-serif'] = ['SimHei']  # Schriftart auf SimHei setzen
rcParams['axes.unicode_minus'] = False  # Problem mit Minuszeichen beheben

# ========= Parameterdefinition =========
# Definition der Eingabedaten und Patch-Verarbeitungsparameter
image_dir = r"E:\Project1\ms data\UAV3-MS"  # Verzeichnis der multispektralen Bilder
shapefile_path = r"E:\Project1\ms data\metadata\md_FieldSHP\md_FieldSHP.shp"  # Pfad zur Shapefile-Datei
label_column = "crop"  # Spaltenname mit Klassifizierungsinformationen in der Shapefile
patch_configs = [(128, 128), (256, 256), (242, 272)]  # Verschiedene Patch-Größen zum Vergleich

# ========= Shapefile-Daten laden =========
# Laden Sie geografische Daten aus der Shapefile und überprüfen Sie die Anzahl der Einträge
gdf = gpd.read_file(shapefile_path)
print(f"[INFO] {len(gdf)} Polygone aus der Shapefile geladen.")

# ========= Rasterdaten laden =========
# Suchen und laden Sie multispektrale Bilder aus dem angegebenen Verzeichnis
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
if not image_paths:
    raise FileNotFoundError("Keine TIF-Dateien gefunden!")

# Überprüfen Sie die Projektion der Rasterdaten und passen Sie die Projektion der Geodaten bei Bedarf an
with rasterio.open(image_paths[0]) as src:
    image_crs = src.crs  # Koordinatenreferenzsystem (CRS) der Rasterdaten
    band_count = src.count  # Anzahl der Bänder im Raster
    if gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)
        print("[INFO] Geodaten wurden auf das CRS der Rasterdaten projiziert.")

# ========= Hauptschleife: Verarbeitung verschiedener Patch-Größen =========
# Iterieren Sie über die definierten Patch-Größen und extrahieren Sie Merkmale
results_summary = []

# Fügen Sie Code hinzu, um das Bild jedes Patches in der Hauptschleife zu speichern
for patch_width, patch_height in patch_configs:
    print(f"\n========== Verarbeitung der Patch-Größe = {patch_width}x{patch_height} ==========")
    X, y, patch_names = [], [], []

    # Ändern Sie den Schritt auf die Hälfte der Patch-Größe
    step_x = patch_width // 2
    step_y = patch_height // 2

    # Iterieren Sie über alle Rasterbilder
    for path in image_paths:
        with rasterio.open(path) as src:
            img_name = os.path.basename(path)  # Name der aktuellen Rasterdatei
            width, height = src.width, src.height  # Breite und Höhe des Rasters

            # Teilen Sie das Raster basierend auf der Patch-Größe in Blöcke
            for i in range(0, width, step_x):
                for j in range(0, height, step_y):
                    if i + patch_width > width or j + patch_height > height:
                        continue  # Überspringen, wenn der Patch die Rastergrenzen überschreitet
                    window = Window(i, j, patch_width, patch_height)  # Definieren Sie das Fenster des Patches
                    transform = src.window_transform(window)  # Holen Sie sich die Transformationsinformationen des Fensters
                    patch_geom = box(*rasterio.windows.bounds(window, transform))  # Geometrie des Patches
                    patch_gdf = gdf[gdf.intersects(patch_geom)]  # Filtern Sie Geodaten, die den Patch schneiden

                    if patch_gdf.empty:
                        continue  # Überspringen, wenn keine Geodaten den Patch schneiden
                    crops = patch_gdf[label_column].dropna().unique()  # Extrahieren Sie Klassifizierungsinformationen
                    if len(crops) != 1:
                        continue  # Überspringen, wenn der Patch mehrere Klassifizierungen enthält

                    crop_label = crops[0]  # Weisen Sie dem Patch ein Klassifizierungslabel zu
                    patch_data = src.read(window=window)  # Lesen Sie die Rasterdaten des Patches
                    if patch_data.shape[1] == 0 or patch_data.shape[2] == 0:
                        continue  # Überspringen, wenn der Patch leer ist

                    # Aktivieren Sie die Funktion zum Speichern von Patch-Bildern
                    patch_path = f"patch_outputs/{patch_width}x{patch_height}/{img_name}_x{i}_y{j}.png"
                    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                    plt.figure(figsize=(5, 5))
                    plt.imshow(patch_data[0], cmap='gray')  # Zeigen Sie das erste Band an
                    plt.axis('off')
                    plt.title(f"Label: {crop_label}")
                    plt.tight_layout()
                    plt.savefig(patch_path)
                    plt.close()

                    print(f"[INFO] Patch gespeichert als {patch_path}")

                    # Normalisieren Sie alle Bänder
                    for b in range(patch_data.shape[0]):
                        band = patch_data[b]
                        patch_data[b] = (band - band.min()) / (band.max() - band.min() + 1e-6)  # Vermeiden Sie Division durch Null

                    # Berechnen Sie die Merkmale des Patches
                    means = [np.mean(band) for band in patch_data]  # Mittelwert jedes Bandes
                    stds = [np.std(band) for band in patch_data]  # Standardabweichung jedes Bandes
                    maxs = [np.max(band) for band in patch_data]  # Maximalwert jedes Bandes

                    # Berechnen Sie Vegetationsindizes (z. B. NDVI und GNDVI)
                    ndvi = (patch_data[7] - patch_data[3]) / (patch_data[7] + patch_data[3] + 1e-6)
                    gndvi = (patch_data[7] - patch_data[2]) / (patch_data[7] + patch_data[2] + 1e-6)
                    ndvi_mean = np.mean(ndvi)  # Mittelwert des NDVI
                    gndvi_mean = np.mean(gndvi)  # Mittelwert des GNDVI

                    features = means + stds + maxs + [ndvi_mean, gndvi_mean]  # Zusammenfassung aller Merkmale
                    if np.any(np.isnan(features)):
                        continue  # Überspringen, wenn Merkmale NaN-Werte enthalten

                    X.append(features)  # Fügen Sie Merkmale zur Liste hinzu
                    y.append(crop_label)  # Fügen Sie Klassifizierungslabel zur Liste hinzu
                    patch_name = f"{img_name}_x{i}_y{j}"  # Generieren Sie einen Namen für den Patch
                    patch_names.append(patch_name)

    print(f"[INFO] Anzahl der extrahierten Patches: {len(X)}")
    if len(X) < 50:
        print("[WARN] Zu wenige Proben, überspringen Sie diese Größe\n")
        continue

    # Konvertieren Sie die Listen von Merkmalen und Klassifizierungen in Arrays
    X = np.array(X)
    y = np.array(y)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Kodieren Sie Klassifizierungslabel in numerische Werte

    # Teilen Sie die Daten in Trainings- und Testdatensätze auf
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Entfernen Sie NaN-Werte aus den Trainingsdaten
    mask = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Überprüfen Sie die Klassifizierungsverteilung und verwenden Sie bei Bedarf SMOTE zur Balance
    class_counts = Counter(y_train)
    min_class_size = min(class_counts.values())

    if min_class_size > 5:
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        print("[INFO] SMOTE wurde zur Klassifikationsbalance verwendet")
    else:
        X_train_bal, y_train_bal = X_train, y_train
        print("[WARN] Zu wenige Proben, SMOTE übersprungen")

    # Standardisieren Sie den Trainingssatz
    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)

    # Standardisieren Sie den Testsatz basierend auf den Statistiken des Trainingssatzes
    X_test = scaler.transform(X_test)

    # Trainieren Sie den Random-Forest-Klassifikator
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_bal, y_train_bal)

    # Bewerten Sie das Modell auf dem Testdatensatz
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[EVAL] Testgenauigkeit: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Erstellen und Speichern der Konfusionsmatrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Vorhergesagte Werte')
    plt.ylabel('Tatsächliche Werte')
    plt.title(f'Konfusionsmatrix ({patch_width}x{patch_height})')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{patch_width}x{patch_height}.png")
    print(f"[INFO] Konfusionsmatrix wurde als confusion_matrix_{patch_width}x{patch_height}.png gespeichert")

    results_summary.append(((patch_width, patch_height), acc))

# Zeichnen eines Balkendiagramms der Genauigkeit
patch_sizes = [f"{w}x{h}" for (w, h), acc in results_summary]
accuracies = [acc for (w, h), acc in results_summary]

plt.figure(figsize=(10, 6))
plt.bar(patch_sizes, accuracies, color='skyblue')
plt.xlabel("Patch-Größe", fontsize=14)
plt.ylabel("Testgenauigkeit", fontsize=14)
plt.title("Testgenauigkeit für verschiedene Patch-Größen", fontsize=16)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("accuracy_bar_chart.png")
plt.show()
print("[INFO] Genauigkeitsbalkendiagramm wurde als accuracy_bar_chart.png gespeichert")

# Zeichnen der Feature-Importance-Grafik
feature_importances = clf.feature_importances_
feature_names = [f"Feature {i+1}" for i in range(len(feature_importances))]

plt.figure(figsize=(12, 6))
plt.barh(feature_names, feature_importances, color='lightgreen')
plt.xlabel("Feature-Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.title("Analyse der Feature-Importance", fontsize=16)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("[INFO] Feature-Importance-Grafik wurde als feature_importance.png gespeichert")

# Zeichnen der Heatmap des Klassifizierungsberichts
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap des Klassifizierungsberichts", fontsize=16)
plt.tight_layout()
plt.savefig("classification_report_heatmap.png")
plt.show()
print("[INFO] Heatmap des Klassifizierungsberichts wurde als classification_report_heatmap.png gespeichert")

# ========= Zusammenfassung der Ergebnisse =========
# Ausgabe der Genauigkeit für jede Patch-Größe
print("\n📊 Vergleich von Patch-Größe und Genauigkeit")
for (w, h), acc in results_summary:
    print(f"Größe {w}x{h} -> Genauigkeit: {acc:.3f}")
