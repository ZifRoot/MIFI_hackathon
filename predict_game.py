import torch
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr
import numpy as np

# Загрузка обученной модели и токенизатора
model = BertForSequenceClassification.from_pretrained('./game_model')
tokenizer = BertTokenizer.from_pretrained('./game_model')

# Функция для предсказания игры
def predict_game(description):
    inputs = tokenizer(description, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Захват голосового ввода
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Говорите...")
    audio = recognizer.listen(source)
    description = recognizer.recognize_google(audio, language="ru-RU")
    print(f"Введено описание: {description}")

# Предсказание игры
predicted_game_index = predict_game(description)
print(f"Предсказанная игра: {predicted_game_index}")
