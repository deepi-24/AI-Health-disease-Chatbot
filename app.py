import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
import os

# ================= PAGE =================
st.set_page_config(page_title="AI Public Health Chatbot", layout="wide")
st.title("🩺 AI Public Health Chatbot")
st.caption("This system provides health awareness only – Not a medical diagnosis.")

# ================= CHAT MEMORY =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= PATHS =================
DL_PATHS = {
    "skin": r"D:\soft computing\models\DL\skin_mobilenet.h5",
    "eye": r"D:\soft computing\models\DL\eye_mobilenet.h5",
    "chest": r"D:\soft computing\models\DL\chest_xray_mobilenet_cpu.h5",
    "hair": r"D:\soft computing\models\DL\hair_mobilenet_best.h5",
    "nail": r"D:\soft computing\models\DL\nail_mobilenet",
}

ML_PATHS = {
    "diabetes": r"D:\soft computing\models\ML\diabetes_model.pkl",
    "heart": r"D:\soft computing\models\ML\heart_model.pkl",
    "kidney": r"D:\soft computing\models\ML\kidney_model.pkl",
    "liver": r"D:\soft computing\models\ML\liver_disease.pkl",
    "bp": r"D:\soft computing\models\ML\blood_pressure.pkl",
    "obesity": r"D:\soft computing\models\ML\obesity_model.pkl",
    "stomach": r"D:\soft computing\models\ML\stomach.pkl",
}

SC_PATH = r"D:\soft computing\models\SoftComputing\asthma_fuzzy_rules.pkl"

# ================= LOAD MODELS =================
@st.cache_resource
def load_dl(path):
    # Fixed the InputLayer / batch_shape error by adding compile=False
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_ml(path):
    return joblib.load(path)

# Load DL models
dl_models = {k: load_dl(v) for k, v in DL_PATHS.items() if os.path.exists(v)}

# Load ML models
ml_models = {k: load_ml(v) for k, v in ML_PATHS.items() if os.path.exists(v)}

# Load asthma fuzzy model
asthma_model = joblib.load(SC_PATH) if os.path.exists(SC_PATH) else None

# ================= KEYWORDS =================
KEYWORDS = {
    "headache": ["headache","head pain","migraine","dizziness","heavy head","forehead pain","temple pain","sinus pain","light sensitivity","noise sensitivity","nausea","vomiting","blur vision"],
    "diabetes": ["diabetes","sugar","glucose","blood sugar","frequent urination","thirst","fatigue","weight loss","insulin","sweet urine","high sugar","low sugar","tingling feet"],
    "heart": ["heart","chest pain","cardiac","palpitation","arm pain","breathlessness","high bp","cholesterol","pressure chest","sweating","tight chest"],
    "kidney": ["kidney","urine","foamy urine","blood urine","swelling","dialysis","creatinine","back pain","fluid retention"],
    "liver": ["liver","jaundice","yellow eyes","yellow skin","fatty liver","hepatitis","dark urine","abdominal pain","alcohol damage"],
    "asthma": ["asthma","breathing problem","breathless","wheezing","cough","chest tight","allergy","dust","smoke","inhaler"],
    "vision": ["vision","eye","blurred","retina","red eye","itchy eye","watering","double vision","night blindness","eye pain"],
    "skin": ["skin","rash","itching","red patch","lesion","infection","fungal","acne","psoriasis","dermatitis"],
    "obesity": ["obesity","overweight","fat","bmi","weight gain","belly fat","sedentary","no exercise","high calorie"],
    "stomach": ["stomach","gastric","acidity","ulcer","vomiting","diarrhea","indigestion","abdominal pain","bloating","constipation"],
}

# ================= IMAGE PREDICTION =================
def predict_image(img, image_type):
    model = dl_models.get(image_type)

    if model is None:
        return "❌ Model not available for this image type."

    # Resize
    input_shape = model.input_shape[1:3]
    img = img.resize(input_shape)

    # MobileNet preprocessing
    arr = np.array(img)
    arr = preprocess_input(arr)

    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=-1)

    arr = np.expand_dims(arr, axis=0)

    # Predict
    preds = model.predict(arr, verbose=0)[0]
    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index])

    # Label mapping (MAKE SURE THIS MATCHES TRAINING)
    labels = {
        "chest": ["NORMAL", "PNEUMONIA"],
        "eye": ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"],
        "skin": ["Acne", "Dermatitis", "Eczema", "Fungal", "Normal"],
        "hair": ["Alopecia", "Dandruff", "Normal"],
        "nail": ["Melanoma", "Healthy"]
    }

    current_labels = labels.get(image_type, [])
    result_class = current_labels[class_index] if class_index < len(current_labels) else "Unknown"

    # Final decision
    if result_class.lower() in ["normal", "healthy"]:
        return f"✅ {image_type.capitalize()} appears NORMAL."

    return f"⚠️ Detected: {result_class}. Please consult a doctor."
# ================= MEDICAL TEXT RESPONSE =================
def medical_reply(text):
    t = text.lower().replace(" ", "")

    RESPONSES = {

        "headache": """🧠 Headache Awareness

Possible causes:
- Stress, lack of sleep
- Dehydration
- Eye strain

Common symptoms:
- Head pain, dizziness
- Nausea, light sensitivity

What you can do:
- Drink enough water
- Take proper rest
- Reduce screen time

⚠️ Consult a doctor if severe or frequent.""",

        "diabetes": """🧪 Diabetes Awareness

Possible causes:
- High blood sugar levels
- Insulin resistance

Common symptoms:
- Frequent urination
- Excessive thirst
- Fatigue

What you can do:
- Maintain healthy diet
- Exercise regularly
- Monitor blood sugar

⚠️ Seek medical advice for proper diagnosis.""",

        "heart": """❤️ Heart Disease Awareness

Possible causes:
- High cholesterol
- High blood pressure

Common symptoms:
- Chest pain
- Shortness of breath
- Sweating

What you can do:
- Avoid fatty foods
- Exercise regularly
- Check BP frequently

🚨 Seek immediate help if severe chest pain.""",

        "asthma": """🌬️ Asthma Awareness

Possible causes:
- Allergies, dust, smoke
- Cold air

Common symptoms:
- Cough, wheezing
- Breathlessness

What you can do:
- Avoid triggers
- Use inhaler if prescribed

⚠️ Consult doctor if breathing worsens.""",

        "stomach": """🍽️ Stomach Issues Awareness

Possible causes:
- Poor diet
- Acid imbalance

Common symptoms:
- Bloating, acidity
- Abdominal pain

What you can do:
- Avoid spicy/oily foods
- Eat on time

⚠️ Seek help if pain continues.""",

        "skin": """🧴 Skin Condition Awareness

Possible causes:
- Infection or allergy
- Poor hygiene

Common symptoms:
- Redness, itching
- Rashes or lesions

What you can do:
- Keep skin clean
- Avoid sharing personal items

⚠️ Consult dermatologist if severe.""",

        "vision": """👁️ Eye Health Awareness

Possible causes:
- Screen strain
- Diabetes-related damage

Common symptoms:
- Blurred vision
- Eye pain or redness

What you can do:
- Limit screen time
- Regular eye checkup

⚠️ Consult eye specialist if needed.""",

        "obesity": """⚖️ Obesity Awareness

Possible causes:
- High calorie intake
- Lack of physical activity

Common symptoms:
- Excess body weight
- Fatigue

What you can do:
- Exercise regularly
- Follow healthy diet

⚠️ Maintain BMI under control.""",

        "kidney": """🧂 Kidney Health Awareness

Possible causes:
- High BP, diabetes

Common symptoms:
- Swelling
- Changes in urine

What you can do:
- Stay hydrated
- Monitor BP

⚠️ Seek medical checkup.""",
    }

    KEYWORDS_MAP = {
        "headache": ["headache","headache","headpain","migraine"],
        "diabetes": ["diabetes","sugar"],
        "heart": ["heart","chestpain"],
        "asthma": ["asthma","cough","breathless","wheezing"],
        "stomach": ["stomach","gastric","acidity"],
        "skin": ["skin","rash","itch"],
        "vision": ["eye","vision","blur"],
        "obesity": ["obesity","fat","bmi"],
        "kidney": ["kidney","urine"]
    }

    for disease, words in KEYWORDS_MAP.items():
        if any(w in t for w in words):
            return RESPONSES[disease]

    return """🤖 Please describe your symptoms clearly.

Example:
- I have headache and dizziness
- I feel chest pain
- I have skin rash"""

# ================= SIDEBAR IMAGE ANALYSIS =================
st.sidebar.header("🖼️ Medical Image Analysis")
image_type = st.sidebar.selectbox("Select image type", list(DL_PATHS.keys()))
uploaded = st.sidebar.file_uploader("Upload medical image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.sidebar.image(img, width=200)
    result = predict_image(img, image_type)
    st.sidebar.success(result)

# ================= CHATBOT =================
st.subheader("💬 AI Health Bot")
user_input = st.chat_input("How can I help you?")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", medical_reply(user_input)))

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()