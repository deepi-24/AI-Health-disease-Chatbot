from chatbot.nlp_processor import extract_symptoms
from chatbot.disease_router import route_disease
from chatbot.response_generator import generate_response


def ai_chat(user_text):

    symptoms = extract_symptoms(user_text)

    disease_type, prediction = route_disease(symptoms)

    reply = generate_response(
        disease_type,
        prediction,
        user_text
    )

    return reply