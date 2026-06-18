# =====================================
# KIDNEY DISEASE ML MODEL
# (With Categorical Encoding - FIXED)
# =====================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# =====================================
# USER INPUT SECTION
# =====================================

DATASET_PATH = r"D:\soft computing\dataset\KIDNEY.csv"
TARGET_COLUMN = "Kidney_Disease"   # kidney dataset target column
MODEL_SAVE_PATH = r"D:\soft computing\models\ML\kidney_model.pkl"

# =====================================
# STEP 1: LOAD DATASET
# =====================================

print("Loading kidney dataset...")
df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# =====================================
# STEP 2: SPLIT FEATURES & TARGET
# =====================================

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# =====================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =====================================

label_encoders = {}

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Encode target column if needed
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

print("\nCategorical encoding completed.")

# =====================================
# STEP 4: TRAIN-TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# =====================================
# STEP 5: TRAIN MODEL
# =====================================

print("\nTraining Decision Tree model for Kidney Disease...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# =====================================
# STEP 6: EVALUATE MODEL
# =====================================

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# STEP 7: SAVE MODEL AS PICKLE
# =====================================

joblib.dump(model, MODEL_SAVE_PATH)

print("\nKidney disease model saved successfully at:")
print(MODEL_SAVE_PATH)

# =====================================
# END OF SCRIPT
# =====================================
