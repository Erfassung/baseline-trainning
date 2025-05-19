# filepath: e:\Project1\atch Rf Classifier Rect.py
# Einheitliches Skript: Unterstützt den Vergleich von quadratischen und nicht-quadratischen Patches (128x128, 256x256, 242x272)

# ========= Bibliotheken importieren =========
# Importieren der notwendigen Bibliotheken für die Verarbeitung von Rasterdaten, Geodaten und maschinellem Lernen
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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from imblearn.over_sampling import SMOTE
from collections import Counter

# ========= Parameter definieren =========
# Definieren der Eingabedaten und Parameter für die Patch-Verarbeitung
image_dir = r"E:\Project1\ms data\UAV3-MS"  # Verzeichnis mit multispektralen Bildern
shapefile_path = r"E:\Project1\ms data\metadata\md_FieldSHP\md_FieldSHP.shp"  # Pfad zur Shapefile-Datei
label_column = "crop"  # Spalte im Shapefile, die die Klasseninformationen enthält
patch_configs = [(128, 128), (256, 256), (242, 272)]  # Verschiedene Patch-Größen für den Vergleich

# ========= Shapefile-Daten laden =========
# Laden der Geodaten aus der Shapefile-Datei und Überprüfung der Anzahl der Einträge
# Hinweis: Die Geodaten enthalten die Polygone, die die Klasseninformationen repräsentieren.
gdf = gpd.read_file(shapefile_path)
print(f"[INFO] {len(gdf)} Polygone aus der Shapefile geladen.")

# ========= Rasterdaten laden =========
# Suchen und Laden der multispektralen Bilder aus dem angegebenen Verzeichnis
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
if not image_paths:
    raise FileNotFoundError("Keine TIF-Dateien gefunden!")

# Überprüfen der Projektion der Rasterdaten und ggf. Anpassen der Geodaten
with rasterio.open(image_paths[0]) as src:
    image_crs = src.crs  # Koordinatenreferenzsystem (CRS) der Rasterdaten
    band_count = src.count  # Anzahl der Bänder im Raster
    if gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)
        print("[INFO] Geodaten wurden auf das CRS der Rasterdaten projiziert.")

# ========= Hauptschleife: Verarbeitung verschiedener Patch-Größen =========
# Iteration über die definierten Patch-Größen und Extraktion der Merkmale
results_summary = []

for patch_width, patch_height in patch_configs:
    print(f"\n========== Verarbeitung von Patches mit Größe {patch_width}x{patch_height} ==========")
    X, y, patch_names = [], [], []

    # Iteration über alle Rasterbilder
    for path in image_paths:
        with rasterio.open(path) as src:
            img_name = os.path.basename(path)  # Name der aktuellen Rasterdatei
            width, height = src.width, src.height  # Dimensionen des Rasters

            # Iteration über das Raster in Schritten der Patch-Größe
            for i in range(0, width, patch_width):
                for j in range(0, height, patch_height):
                    if i + patch_width > width or j + patch_height > height:
                        continue  # Überspringen, wenn der Patch außerhalb der Rastergrenzen liegt
                    window = Window(i, j, patch_width, patch_height)  # Definieren des Fensters für den Patch
                    transform = src.window_transform(window)  # Transformation für das Fenster
                    patch_geom = box(*rasterio.windows.bounds(window, transform))  # Geometrie des Patches
                    patch_gdf = gdf[gdf.intersects(patch_geom)]  # Filtern der Geodaten, die den Patch schneiden

                    if patch_gdf.empty:
                        continue  # Überspringen, wenn keine Geodaten den Patch schneiden
                    crops = patch_gdf[label_column].dropna().unique()  # Extrahieren der Klasseninformationen
                    if len(crops) != 1:
                        continue  # Überspringen, wenn der Patch mehrere Klassen enthält

                    crop_label = crops[0]  # Klassenzuweisung für den Patch
                    patch_data = src.read(window=window)  # Lesen der Rasterdaten für den Patch
                    if patch_data.shape[1] == 0 or patch_data.shape[2] == 0:
                        continue  # Überspringen, wenn der Patch leer ist

                    # Berechnung der Merkmale für den Patch
                    means = [np.mean(band) for band in patch_data]  # Mittelwerte der Bänder
                    stds = [np.std(band) for band in patch_data]  # Standardabweichungen der Bänder
                    maxs = [np.max(band) for band in patch_data]  # Maximalwerte der Bänder

                    # Berechnung von Vegetationsindizes (z. B. NDVI, GNDVI)
                    ndvi = (patch_data[7] - patch_data[3]) / (patch_data[7] + patch_data[3] + 1e-6)
                    gndvi = (patch_data[7] - patch_data[2]) / (patch_data[7] + patch_data[2] + 1e-6)
                    ndvi_mean = np.mean(ndvi)  # Mittelwert des NDVI
                    gndvi_mean = np.mean(gndvi)  # Mittelwert des GNDVI

                    features = means + stds + maxs + [ndvi_mean, gndvi_mean]  # Zusammenstellen der Merkmale
                    if np.any(np.isnan(features)):
                        continue  # Überspringen, wenn NaN-Werte vorhanden sind

                    X.append(features)  # Hinzufügen der Merkmale zur Liste
                    y.append(crop_label)  # Hinzufügen der Klasse zur Liste
                    patch_name = f"{img_name}_x{i}_y{j}"  # Generieren eines Namens für den Patch
                    patch_names.append(patch_name)

    print(f"[INFO] Anzahl der extrahierten Patches: {len(X)}")
    if len(X) < 50:
        print("[WARN] Zu wenige Patches, Überspringen dieser Größe\n")
        continue

    # Konvertieren der Merkmals- und Klassenlisten in Arrays
    X = np.array(X)
    y = np.array(y)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Kodieren der Klassen in numerische Werte

    # Aufteilen der Daten in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Entfernen von NaN-Werten aus den Trainingsdaten
    mask = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Überprüfen der Klassenverteilung und ggf. Ausgleich mit SMOTE
    class_counts = Counter(y_train)
    min_class_size = min(class_counts.values())

    if min_class_size > 5:
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        print("[INFO] SMOTE wurde zur Ausbalancierung der Klassen verwendet")
    else:
        X_train_bal, y_train_bal = X_train, y_train
        print("[WARN] Zu wenige Daten, SMOTE wurde übersprungen")

    # Training des Random-Forest-Klassifikators
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_bal, y_train_bal)

    # Bewertung des Modells auf den Testdaten
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[EVAL] Genauigkeit auf dem Testdatensatz: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Erstellen und Speichern der Konfusionsmatrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Vorhergesagt')
    plt.ylabel('Wahr')
    plt.title(f'Konfusionsmatrix ({patch_width}x{patch_height})')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{patch_width}x{patch_height}.png")
    print(f"[INFO] Konfusionsmatrix gespeichert als confusion_matrix_{patch_width}x{patch_height}.png")

    results_summary.append(((patch_width, patch_height), acc))

# ========= Zusammenfassung der Ergebnisse =========
# Ausgabe der Genauigkeit für jede Patch-Größe
import pandas as pd

print("\n📊 Vergleich der Patch-Größen und Genauigkeiten")
df_result = pd.DataFrame(results_summary, columns=["PatchSize", "Accuracy"])
for (w, h), acc in results_summary:
    print(f"Größe {w}x{h} -> Genauigkeit: {acc:.3f}")

# ========= Erstellen eines Balkendiagramms =========
# Visualisierung der Genauigkeit für jede Patch-Größe
plt.figure(figsize=(8, 5))
labels = [f"{w}x{h}" for (w, h) in df_result["PatchSize"]]
plt.bar(labels, df_result["Accuracy"], color='skyblue')
plt.ylabel("Genauigkeit")
plt.title("Patch-Größe vs. Genauigkeit")
plt.ylim(0, 1)
for i, acc in enumerate(df_result["Accuracy"]):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
plt.tight_layout()
plt.savefig("patch_size_accuracy_bar.png")
print("[INFO] Balkendiagramm gespeichert als patch_size_accuracy_bar.png")
