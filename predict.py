import cv2
import joblib
from features import extract_features

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_image(path):
    img = cv2.imread(path)
    feat = extract_features(img)
    feat = scaler.transform([feat])
    
    pred = model.predict(feat)
    print("Prediction:", pred[0])

# test
predict_image("deer.jpg")
