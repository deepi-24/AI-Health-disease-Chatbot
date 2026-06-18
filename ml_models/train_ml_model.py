# ==============================
# ML Model Training & Pickle Creation
# ==============================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# USER INPUT SECTION (EDIT ONLY THIS)
# ==============================

DATASET_PATH = r"D:\soft computing\dataset\DIABETES.csv" # path to cleaned dataset
TARGET_COLUMN = "Diabetes" # target/label column
MODEL_SAVE_PATH = r"D:\soft computing\models\ML\diabetes_model.pkl"        # output pickle file

# ==============================
# STEP 2: LOAD DATASET
# ==============================

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# ==============================
# STEP 3: SPLIT FEATURES & LABEL
# ==============================

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# ==============================
# STEP 4: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==============================
# STEP 5: TRAIN ML MODEL
# ==============================

print("Training Decision Tree model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ==============================
# STEP 6: MODEL EVALUATION
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# STEP 7: SAVE MODEL AS PICKLE
# ==============================

joblib.dump(model, MODEL_SAVE_PATH)

print("\nModel saved successfully at:", MODEL_SAVE_PATH)

# ==============================
# END OF SCRIPT
# ==============================
