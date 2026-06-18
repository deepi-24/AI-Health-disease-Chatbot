from skfuzzy import control as ctrl
from asthma_rules import create_asthma_rules
from fuzzy_membership import create_fuzzy_variables
import joblib
import os

# ---------------- SAFE MODEL PATH ----------------
MODEL_DIR = r"D:\soft computing\Soft_somputing"
MODEL_PATH = os.path.join(MODEL_DIR, "asthma_fuzzy_rules.pkl")

# ---------------- CREATE FOLDER (ONLY IF NEEDED) ----------------
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ---------------- BUILD & SAVE MODEL ----------------
def build_and_save_model():
    breathlessness, wheezing, cough, asthma_severity = create_fuzzy_variables()
    rules = create_asthma_rules()

    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)

    joblib.dump(simulation, MODEL_PATH)
    print("Asthma fuzzy model saved successfully!")
    print("Saved at:", MODEL_PATH)

# ---------------- PREDICTION FUNCTION ----------------
def predict_asthma(b, w, c):
    model = joblib.load(MODEL_PATH)

    model.input['breathlessness'] = b
    model.input['wheezing'] = w
    model.input['cough'] = c

    model.compute()

    return model.output['asthma_severity']

# ---------------- MAIN ----------------
if __name__ == "__main__":
    build_and_save_model()
