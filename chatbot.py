import random
import json
import pickle
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Importamos los archivos generados en el código anterior
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Pasamos las palabras de la oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_letters]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)  # Obtiene el índice de la categoría con la probabilidad más alta
    category = classes[max_index]
    return category

# Obtenemos una respuesta aleatoria correspondiente a la categoría
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])  # Respuesta aleatoria
            break
    return result

# Función principal para obtener la respuesta
def respuesta(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res.encode('utf-8').decode('utf-8')

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://soccerruleschatbot.netlify.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/question")
async def get_question(question: str = Query(..., description="La pregunta que deseas responder")):
    return JSONResponse(content={"response": respuesta(question)}, media_type="application/json")