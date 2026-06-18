# ================= NLP PROCESSOR =================

def preprocess_text(text):
    """
    Clean user text
    """
    text = text.lower().strip()
    return text


def extract_symptoms(user_input):
    """
    Extract medical symptoms from user message
    """

    user_input = preprocess_text(user_input)

    symptom_keywords = {

        "fever": [
            "fever", "high temperature", "temperature",
            "body heat"
        ],

        "cough": [
            "cough", "dry cough", "continuous cough"
        ],

        "breathlessness": [
            "breathless",
            "breathing problem",
            "shortness of breath",
            "difficulty breathing"
        ],

        "wheezing": [
            "wheezing", "whistle breathing"
        ],

        "chest_pain": [
            "chest pain", "heart pain"
        ],

        "skin_issue": [
            "rash", "itching", "skin allergy",
            "red spots", "skin problem"
        ],

        "eye_issue": [
            "eye pain",
            "blur vision",
            "eye irritation",
            "red eye"
        ],

        "hair_issue": [
            "hair fall",
            "hair loss",
            "baldness"
        ],

        "nail_issue": [
            "nail discoloration",
            "yellow nail",
            "nail infection"
        ],

        "diabetes": [
            "diabetes",
            "sugar",
            "high sugar"
        ],

        "bp": [
            "blood pressure",
            "bp",
            "hypertension"
        ]
    }

    detected_symptoms = set()

    # ===== Keyword Matching =====
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in user_input:
                detected_symptoms.add(symptom)
                break

    return list(detected_symptoms)