# ================= IMPORTS =================
from chatbot.safety_layer import apply_safety

# Optional LLM support
try:
    from llm.llm_interface import generate_llm_response
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False


# ================= RESPONSE BUILDER =================
def build_basic_response(disease_type, prediction):
    """
    Create structured medical response
    """

    if disease_type == "ASTHMA_SOFT":
        return f"🫁 Asthma severity detected: {prediction}"

    elif disease_type == "DIABETES_ML":
        return f"🩸 Diabetes risk prediction: {prediction}"

    elif disease_type == "BP_ML":
        return f"❤️ Blood pressure level: {prediction}"

    elif disease_type == "SKIN_DL":
        return f"🧴 Skin condition detected: {prediction}"

    elif disease_type == "EYE_DL":
        return f"👁 Eye disease detected: {prediction}"

    elif disease_type == "HAIR_DL":
        return f"💇 Hair condition detected: {prediction}"

    elif disease_type == "NAIL_DL":
        return f"💅 Nail disease detected: {prediction}"

    elif disease_type == "UNKNOWN":
        return "❓ Unable to detect disease from given symptoms."

    return "✅ Prediction completed."


# ================= MAIN GENERATOR =================
def generate_response(disease_type, prediction, user_text=None):
    """
    Final chatbot response pipeline
    """

    # Step 1 → basic response
    response = build_basic_response(disease_type, prediction)

    # Step 2 → enhance using LLM (if available)
    if LLM_AVAILABLE:
        try:
            response = generate_llm_response(disease_type, prediction)
        except Exception:
            pass

    # Step 3 → apply safety layer
    response = apply_safety(response, user_text)

    return response