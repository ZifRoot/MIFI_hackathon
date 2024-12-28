import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных из game.json
with open('./data/game.json', 'r') as f:
    games = json.load(f)

# Маппинг игр на индексы для классификации
game_names = [game["name"] for game in games]
game_descriptions = [game["description"] for game in games]

# Токенизация
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(game_descriptions, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Маппинг названий игр на метки (классы)
labels = np.array([i for i in range(len(games))])

# Разделение на тренировочную и тестовую выборки
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs['input_ids'], labels, test_size=0.2, random_state=42)

# Создание датасетов для тренировки
class GameDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = GameDataset(train_inputs, train_labels)
test_dataset = GameDataset(test_inputs, test_labels)

# Загрузка модели BERT для классификации
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(games))

# Настройка аргументов для тренировки без W&B
training_args = TrainingArguments(
    output_dir='./results',          # директория для сохранения результатов
    num_train_epochs=3,              # количество эпох
    per_device_train_batch_size=8,   # размер пакета для тренировки
    per_device_eval_batch_size=16,   # размер пакета для оценки
    warmup_steps=500,                # количество шагов для разогрева
    weight_decay=0.01,               # регуляризация
    logging_dir='./logs',            # директория для логов
    report_to="none"                 # Отключает интеграцию с wandb
)

# Создание тренера
trainer = Trainer(
    model=model,                         # модель
    args=training_args,                  # аргументы
    train_dataset=train_dataset,         # тренировочный датасет
    eval_dataset=test_dataset,           # тестовый датасет
    compute_metrics=lambda p: {'accuracy': accuracy_score(p.predictions.argmax(axis=-1), p.label_ids)}  # метрика
)

# Тренировка модели
trainer.train()

# Сохранение модели
model.save_pretrained('./game_model')
tokenizer.save_pretrained('./game_model')

print("Модель обучена и сохранена в 'game_model'")
