import os
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === Manually defined plot_ID to crop name mapping ===
id_to_crop = {
    162: "Summerwheat", 163: "Summerwheat", 164: "Summerwheat", 165: "Summerwheat", 166: "Summerwheat",
    167: "Summerwheat", 168: "Summerwheat", 169: "Summerwheat", 170: "Summerwheat",
    171: "Soybean", 172: "Soybean", 173: "Soybean", 174: "Soybean", 175: "Soybean",
    176: "Soybean", 177: "Soybean", 178: "Soybean", 179: "Soybean", 180: "Soybean",
    181: "Potato", 182: "Potato", 183: "Potato", 184: "Potato", 185: "Potato",
    186: "Potato", 187: "Potato", 188: "Potato", 189: "Potato", 190: "Potato",
    191: "Sugar Beet", 192: "Sugar Beet", 193: "Sugar Beet", 194: "Sugar Beet",
    195: "Sugar Beet", 196: "Sugar Beet", 197: "Sugar Beet", 198: "Sugar Beet",
    199: "Sugar Beet", 200: "Sugar Beet", 201: "Sugar Corn", 202: "Sugar Corn",
    203: "Sugar Corn", 204: "Sugar Corn", 205: "Sugar Corn", 206: "Sugar Corn",
    207: "Sugar Corn", 208: "Sugar Corn", 209: "Sugar Corn", 210: "Sugar Corn",
    211: "Mixture", 212: "Mixture", 213: "Mixture", 214: "Mixture", 215: "Mixture",
    216: "Mixture", 217: "Mixture", 218: "Mixture", 219: "Mixture", 220: "Mixture",
    221: "Mixture", 222: "Mixture", 223: "Mixture", 224: "Mixture", 225: "Mixture",
    226: "Mixture", 227: "Mixture", 228: "Mixture", 229: "Mixture", 230: "Mixture",
    231: "Mixture", 232: "Mixture", 233: "Mixture", 234: "Mixture", 235: "Mixture",
    236: "Mixture", 237: "Mixture", 238: "Sugar Beet", 239: "Sugar Beet",
    240: "Sugar Beet", 241: "Sugar Beet"
}

# === Feature extraction function ===
def extract_features(img_path):
    img = imread(img_path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return np.array([
        img.mean(),
        img.std(),
        img.min(),
        img.max(),
        np.percentile(img, 25),
        np.percentile(img, 50),
        np.percentile(img, 75)
    ])

# === Load merged metadata ===
metadata_path = r"E:\Erfassung\project_ms\output\merged_metadata.csv"
data = pd.read_csv(metadata_path)

# === Filter out invalid labels not in the mapping ===
data = data[data["dominant_plot_id"].isin(id_to_crop.keys())]
print(f"✅ Valid samples: {len(data)}")

# === Extract features and labels ===
labels = data["dominant_plot_id"].values
features = np.vstack([extract_features(p) for p in data["image_patch"]])

# === Class distribution visualization ===
counter = Counter([id_to_crop[i] for i in labels])
plt.figure(figsize=(10, 5))
plt.bar(counter.keys(), counter.values(), color="orange")
plt.title("Crop Class Distribution")
plt.xlabel("Crop Type")
plt.ylabel("Number of Patches")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("crop_class_distribution.png")
plt.close()
print("📊 Saved crop_class_distribution.png")

# === Train model ===
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
true_names = [id_to_crop[i] for i in y_test]
pred_names = [id_to_crop[i] for i in y_pred]

report = classification_report(true_names, pred_names, output_dict=True)
report_text = classification_report(true_names, pred_names)
matrix = confusion_matrix(true_names, pred_names)

# === Save report to PDF ===
pdf_path = "classification_report_summary.pdf"
with PdfPages(pdf_path) as pdf:
    # Page 1: Class Distribution
    fig1 = plt.figure(figsize=(10, 5))
    plt.bar(counter.keys(), counter.values(), color="orange")
    plt.title("Crop Class Distribution")
    plt.xlabel("Crop Type")
    plt.ylabel("Number of Patches")
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close()

    # Page 2: Text report
    fig2 = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    plt.title("Classification Report", loc="left")
    lines = report_text.split('\n')
    for i, line in enumerate(lines):
        plt.text(0, 1 - i * 0.04, line, fontsize=10, family="monospace")
    pdf.savefig(fig2)
    plt.close()

print(f"✅ PDF summary saved: {pdf_path}")

# === Final print to console ===
print("\n📊 Classification Report:")
print(report_text)
print("\n🔍 Confusion Matrix:")
print(matrix)
print("\n✅ Model training and evaluation completed.")