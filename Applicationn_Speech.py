# Les librairies
import speech_recognition as sr
import pyaudio   
import pyttsx3 as ttx
import pywhatkit
import datetime 

# Demennder au micro pour ecouter
listener = sr.Recognizer()
engine = ttx.init()
voice = engine.getProperty('voices')
engine.setProperty('voice', 'french') # La langue

def parle(text) :
    engine.say(text)
    engine.runAndWait()

def ecoute() :
    try :
        with sr.Microphone() as source :
            print("Parlez s'il vous plaît")
            voix = listener.listen(source)
            command = listener.recognize_google(voix, language='fr-FR')

    except :
        pass
    return command

def assistant() :
    command = ecoute()
    print(command)
    if 'bonjour' in command : 
        parle('bonjour comment ça va ?')

    elif 'oui ça va et toi' in command :
        parle('Oui ça va de mon coté, comment je peux vous aider')
    
    elif 'Mettez la chanson de' in command :
        chanteur = command.replace('Mettez la chanson', '')
        print(chanteur)
        pywhatkit.playony(chanteur)

    elif 'heure' in command :
        heure = datetime.datetime.now().strftime('%H:%M')
        parle('Il est'+ heure)

    elif 'ton nom' in command :
        parle("Je m'appelle Moutassim")

while True :
    assistant()
