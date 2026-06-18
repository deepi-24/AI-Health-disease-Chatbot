# ==============================
# HEART DISEASE MODEL
# ==============================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = r"D:\soft computing\dataset\HEART.csv"
TARGET_COLUMN = "Heart_Disease"
MODEL_SAVE_PATH = r"D:\soft computing\models\ML\heart_model.pkl"

df = pd.read_csv(DATASET_PATH)

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_SAVE_PATH)
print("Heart model saved")
