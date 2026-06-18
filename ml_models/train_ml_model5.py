# =====================================
# BLOOD PRESSURE ML MODEL
# (HIGH ACCURACY VERSION)
# =====================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# =====================================
# USER INPUT SECTION
# =====================================

DATASET_PATH = r"D:\soft computing\dataset\blood pressure.csv"
TARGET_COLUMN = "Adrenal_and_thyroid_disorders"
MODEL_SAVE_PATH = r"D:\soft computing\models\ML\blood_pressure.pkl"

# =====================================
# STEP 1: LOAD DATASET
# =====================================

print("Loading blood pressure dataset...")
df = pd.read_csv(DATASET_PATH)
print("Dataset Shape:", df.shape)

# =====================================
# STEP 2: SPLIT FEATURES & TARGET
# =====================================

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

print("Categorical columns:", list(categorical_cols))
print("Numerical columns:", list(numerical_cols))

# =====================================
# STEP 3: PREPROCESSING PIPELINE
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# =====================================
# STEP 4: MODEL PIPELINE (RANDOM FOREST)
# =====================================

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# =====================================
# STEP 5: TRAIN-TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT for medical data
)

# =====================================
# STEP 6: TRAIN MODEL
# =====================================

print("\nTraining high-accuracy blood pressure model...")
model.fit(X_train, y_train)

# =====================================
# STEP 7: EVALUATE MODEL
# =====================================

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# STEP 8: SAVE MODEL
# =====================================

joblib.dump(model, MODEL_SAVE_PATH)
print("\nHigh-accuracy blood pressure model saved at:")
print(MODEL_SAVE_PATH)
