from chatbot.nlp_processor import extract_symptoms


def route_disease(user_text):

    symptoms = extract_symptoms(user_text)

    if not symptoms:
        return "Unable to detect symptoms."

    # ===== ROUTING =====
    if "fever" in symptoms:
        return "Possible Viral Fever detected."

    elif "cough" in symptoms:
        return "Possible Respiratory Infection."

    elif "chest pain" in symptoms:
        return "Possible Heart-related issue."

    elif "rash" in symptoms or "itching" in symptoms:
        return "Possible Skin Disease."

    elif "diabetes" in symptoms:
        return "Possible Diabetes symptoms."

    else:
        return "General health issue detected."