import pyttsx3

# Voice
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")
voice_rate = 145
engine.setProperty("rate", voice_rate)


# Speak Function
def speak(text: str, print_text: bool = False):
    if not print_text:
        engine.say(text)
        engine.runAndWait()
    elif print_text:
        engine.say(text)
        print("Program: " + text)
        engine.runAndWait()
