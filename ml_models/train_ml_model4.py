# =====================================
# LIVER DISEASE ML MODEL
# (FINAL FIX – OneHotEncoding Pipeline)
# =====================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# =====================================
# USER INPUT SECTION
# =====================================

DATASET_PATH = r"D:\soft computing\dataset\liver dataset.csv"
TARGET_COLUMN = "Liver_Disease"   # Liver dataset label column
MODEL_SAVE_PATH = r"D:\soft computing\models\ML\liver_disease.pkl"

# =====================================
# STEP 1: LOAD DATASET
# =====================================

print("Loading liver dataset...")
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
# STEP 3: PREPROCESSING
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# =====================================
# STEP 4: PIPELINE (PREPROCESS + MODEL)
# =====================================

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# =====================================
# STEP 5: TRAIN-TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# STEP 6: TRAIN MODEL
# =====================================

print("\nTraining liver disease model...")
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

print("\nLiver disease model saved successfully at:")
print(MODEL_SAVE_PATH)

# =====================================
# END OF SCRIPT
# =====================================
