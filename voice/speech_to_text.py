import speech_recognition as sr

def listen_from_mic(timeout=5):
    """
    Capture voice from microphone and convert to text.
    Returns recognized text or None.
    """

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Listening... Speak now")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio)
            print("📝 You said:", text)
            return text

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected")
            return None

        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None

        except sr.RequestError:
            print("🌐 Internet error for speech service")
            return None
