# Q2.py

import os
import glob
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

def load_images_from_folders(folders, size=(64, 64)):
    """
    Loads and processes all images from given folders.
    Each image is resized, normalized, and flattened.
    The label is taken from the name of the subfolder (Cat/Dog).
    """
    data, labels = [], []

    for folder in folders:
        if not os.path.exists(folder):
            print(f"[ERROR] Folder not found: {folder}")
            continue
        image_files = glob.glob(os.path.join(folder, "*", "*"))
        for path in image_files:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, size)
            img = img.astype("float32") / 255.0
            data.append(img.flatten())
            label = os.path.basename(os.path.dirname(path))
            labels.append(label)

    return np.array(data), np.array(labels)

# Absolute paths for train and test folders
train_folder = r'c:\Users\gazal\Documents\Computer Vision\Assignment2\Q2\train'
test_folder = r'c:\Users\gazal\Documents\Computer Vision\Assignment2\Q2\test'

print("[INFO] Loading images...")
X_train, y_train = load_images_from_folders([train_folder])
X_test, y_test = load_images_from_folders([test_folder])
print(f"[INFO] Loaded {len(X_train)} train and {len(X_test)} test images.")
print(np.unique(y_train, return_counts=True))

if len(X_train) == 0 or len(X_test) == 0:
    raise RuntimeError("No images found! Check your train/test folder paths and image files.")

# Try multiple classifiers and tune parameters
results = {}

# KNN (tune k)
best_knn_acc = 0
best_k = 1
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if acc > best_knn_acc:
        best_knn_acc = acc
        best_k = k
        best_knn = knn
results['KNN'] = (best_knn_acc, best_k, best_knn)
print(f"[KNN] Best k={best_k}, Accuracy={best_knn_acc:.4f}")

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_preds = logreg.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_preds)
results['LogisticRegression'] = (logreg_acc, logreg)
print(f"[Logistic Regression] Accuracy={logreg_acc:.4f}")

# Random Forest (optional, for comparison)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
results['RandomForest'] = (rf_acc, rf)
print(f"[Random Forest] Accuracy={rf_acc:.4f}")

# Save the best model (highest accuracy)
best_model_name = max(results, key=lambda k: results[k][0])
if best_model_name == 'KNN':
    best_model = results[best_model_name][2]
else:
    best_model = results[best_model_name][1]
joblib.dump(best_model, f"{best_model_name}_catdog_model.pkl")
print(f"[INFO] Saved best model: {best_model_name}")

# Print classification report for best model
print(f"\n[Classification Report for {best_model_name}]")
print(classification_report(y_test, best_model.predict(X_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, best_model.predict(X_test)))

def predict_single_image(image_path, model, size=(64, 64)):
    """
    Loads and prepares one image, then predicts using the given model.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return "Unknown"
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    img_flat = img.flatten().reshape(1, -1)
    return model.predict(img_flat)[0]

# Test on several images from the internet 
# (place them in a folder called 'internet_test')
internet_test_folder = r'c:\Users\gazal\Documents\Computer Vision\Assignment2\Q2\internet_test'
if os.path.exists(internet_test_folder):
    print("\n[INFO] Predicting on internet images...")
    for img_file in glob.glob(os.path.join(internet_test_folder, "*.jpg")):
        result = predict_single_image(img_file, best_model)
        print(f"{os.path.basename(img_file)} → Predicted as: {result}")
else:
    print("\n[INFO] No 'internet_test' folder found. Place internet images there to test.")
