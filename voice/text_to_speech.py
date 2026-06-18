import pyttsx3

# Initialize once (important for speed)
engine = pyttsx3.init()
engine.setProperty("rate", 170)   # speaking speed
engine.setProperty("volume", 1.0) # max volume


def speak(text: str):
    """
    Convert chatbot reply text to voice.
    """
    if not text:
        return

    print("🤖 Speaking:", text)
    engine.say(text)
    engine.runAndWait()
