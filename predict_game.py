import torch
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr
import pandas as pd

# Загрузка обученной модели и токенизатора
model = BertForSequenceClassification.from_pretrained('./game_model')
tokenizer = BertTokenizer.from_pretrained('./game_model')

# Загрузка данных из game.json в DataFrame
data = pd.read_json('./data/game.json')

# Убедитесь, что нужные столбцы существуют
if 'name' not in data.columns or 'description' not in data.columns:
    raise ValueError("В данных отсутствуют столбцы 'gameTitle' или 'description'.")

# Функция для предсказания игры
def predict_top_games(description, top_n=5):
    inputs = tokenizer(description, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    
    # Получаем top_n вероятностей и индексов
    top_probs, top_indices = torch.topk(probabilities, top_n)
    
    # Создаём таблицу с именами игр и вероятностями
    results = []
    for prob, idx in zip(top_probs, top_indices):
        if idx.item() < len(data):
            game_title = data.iloc[idx.item()]['name']
            results.append({"Имя": game_title, "Вероятность": prob.item()})
        else:
            results.append({"Имя": "Неизвестная игра", "Вероятность": prob.item()})
    
    return pd.DataFrame(results)

# Захват голосового ввода
recognizer = sr.Recognizer()

print("Говорите...")
with sr.Microphone() as source:
    audio = recognizer.listen(source)

try:
    # Распознавание текста
    description = recognizer.recognize_google(audio, language="ru-RU")
    print(f"Введено описание: {description}")

    # Предсказание топ-5 игр
    top_games = predict_top_games(description, top_n=5)
    print("Топ-5 предсказанных игр:")
    print(top_games)
except sr.UnknownValueError:
    print("Не удалось распознать речь.")
except sr.RequestError as e:
    print(f"Ошибка сервиса распознавания речи: {e}")
except KeyError:
    print("Предсказанный индекс отсутствует в данных.")
