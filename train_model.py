import os
import cv2
import numpy as np
from features import extract_features

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import joblib

X, y = [], []

for label in ["bobcat","coyote","raccoon","deer","empty"]:
    folder = f"dataset/{label}"
    
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        
        if img is None:
            continue
        
        X.append(extract_features(img))
        y.append(label)

X = np.array(X)
y = np.array(y)

# scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

# save model
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved!")