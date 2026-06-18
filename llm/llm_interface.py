from llm.prompt_templates import health_prompt

def generate_llm_response(disease_type, prediction):
    """
    Simulated LLM response (Closed-domain healthcare)
    """

    prompt = health_prompt(disease_type, prediction)

    # Simulated LLM output (replace with real LLM later if needed)
    response = f"""
🩺 Health Awareness Result

Condition: {prediction}

This condition is related to {disease_type.lower()}.
It may occur due to lifestyle, genetic, or environmental factors.

Common symptoms may include discomfort, fatigue, or visible changes.
Maintaining a healthy lifestyle and early awareness can help reduce risks.

⚠️ Disclaimer:
This chatbot provides health awareness only.
Please consult a qualified medical professional for diagnosis or treatment.
"""

    return response.strip()
