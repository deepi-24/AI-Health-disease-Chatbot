# ================= SAFETY LAYER =================

def check_emergency(user_text):
    """
    Detect emergency medical situations
    """

    user_text = user_text.lower()

    emergency_keywords = [
        "chest pain",
        "heart attack",
        "cannot breathe",
        "breathing stopped",
        "unconscious",
        "severe bleeding",
        "stroke",
        "fainted"
    ]

    for word in emergency_keywords:
        if word in user_text:
            return True

    return False


def apply_safety(response, user_text=None):
    """
    Apply safety rules and disclaimer
    """

    # ===== Emergency Override =====
    if user_text and check_emergency(user_text):
        return (
            "🚨 EMERGENCY DETECTED!\n"
            "Please seek immediate medical attention or call emergency services.\n"
            "Do NOT rely on chatbot advice in emergencies."
        )

    # ===== Normal Disclaimer =====
    disclaimer = (
        "\n\n⚠️ Disclaimer:\n"
        "This AI Healthcare chatbot provides health awareness only.\n"
        "It is NOT a medical diagnosis.\n"
        "Please consult a qualified doctor."
    )

    return response + disclaimer