import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib
import os

# ==============================
# PATHS (MODEL SAVE LOCATION)
# ==============================

MODEL_DIR = r"D:\soft computing\models\SoftComputing"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "asthma_fuzzy_rules.pkl")

# ==============================
# FUZZY INPUT VARIABLES
# ==============================

breathlessness = ctrl.Antecedent(np.arange(0, 11, 1), 'breathlessness')
wheezing = ctrl.Antecedent(np.arange(0, 11, 1), 'wheezing')
cough = ctrl.Antecedent(np.arange(0, 11, 1), 'cough')

# ==============================
# FUZZY OUTPUT VARIABLE
# ==============================

asthma_severity = ctrl.Consequent(np.arange(0, 11, 1), 'asthma_severity')

# ==============================
# MEMBERSHIP FUNCTIONS
# ==============================

breathlessness['low'] = fuzz.trimf(breathlessness.universe, [0, 0, 4])
breathlessness['medium'] = fuzz.trimf(breathlessness.universe, [3, 5, 7])
breathlessness['high'] = fuzz.trimf(breathlessness.universe, [6, 10, 10])

wheezing['low'] = fuzz.trimf(wheezing.universe, [0, 0, 4])
wheezing['medium'] = fuzz.trimf(wheezing.universe, [3, 5, 7])
wheezing['high'] = fuzz.trimf(wheezing.universe, [6, 10, 10])

cough['low'] = fuzz.trimf(cough.universe, [0, 0, 4])
cough['medium'] = fuzz.trimf(cough.universe, [3, 5, 7])
cough['high'] = fuzz.trimf(cough.universe, [6, 10, 10])

asthma_severity['mild'] = fuzz.trimf(asthma_severity.universe, [0, 0, 4])
asthma_severity['moderate'] = fuzz.trimf(asthma_severity.universe, [3, 5, 7])
asthma_severity['severe'] = fuzz.trimf(asthma_severity.universe, [6, 10, 10])

# ==============================
# FUZZY RULES
# ==============================

rule1 = ctrl.Rule(
    breathlessness['high'] & wheezing['high'],
    asthma_severity['severe']
)

rule2 = ctrl.Rule(
    cough['medium'] & wheezing['low'],
    asthma_severity['mild']
)

rule3 = ctrl.Rule(
    breathlessness['medium'] & wheezing['medium'],
    asthma_severity['moderate']
)

rule4 = ctrl.Rule(
    breathlessness['low'] & wheezing['low'] & cough['low'],
    asthma_severity['mild']
)

# ==============================
# CONTROL SYSTEM
# ==============================

asthma_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
asthma_simulation = ctrl.ControlSystemSimulation(asthma_ctrl)

# ==============================
# SAVE FUZZY MODEL
# ==============================

joblib.dump(asthma_simulation, MODEL_PATH)

print("✅ Asthma fuzzy logic model saved successfully!")
print("📁 Saved at:", MODEL_PATH)
