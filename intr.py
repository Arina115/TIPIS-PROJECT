
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# === Архитектура модели ===
class WineQualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 6)  # 6 классов: 3,4,5,6,7,8
        )
    def forward(self, x):
        return self.layers(x)

# === Загрузка модели и скалера ===
@st.cache_resource
def load_model():
    model = WineQualityNet()
    model.load_state_dict(torch.load('best_wine_model.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load('wine_robust_scaler.pkl')

model = load_model()
scaler = load_scaler()

# === Названия признаков (ТОЧНО в том же порядке, что и в датасете!) ===
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

# === Streamlit интерфейс ===
st.title("🍷 Предсказание качества вина")
st.write("Введите физико-химические параметры вина:")

# Словарь разумных значений по умолчанию (на основе датасета)
defaults = [8.3, 0.53, 0.27, 2.5, 0.08, 15.0, 45.0, 0.996, 3.3, 0.66, 10.0]

user_input = []
for feat, default in zip(FEATURES, defaults):
    val = st.number_input(
        label=feat,
        value=float(default),
        min_value=0.0,
        step=0.1,
        format="%.2f"
    )
    user_input.append(val)

# Кнопка
if st.button("Предсказать качество"):
    try:
        # Подготовка данных
        x = np.array(user_input).reshape(1, -1)
        x_scaled = scaler.transform(x)
        
        # Предсказание
        with torch.no_grad():
            logits = model(torch.tensor(x_scaled, dtype=torch.float32))
            pred_class = torch.argmax(logits, dim=1).item()
            quality = pred_class + 3  # 0→3, 1→4, ..., 5→8

        st.success(f"✅ Качество вина: **{quality}** (по шкале от 3 до 8)")
    except Exception as e:
        st.error(f"Ошибка: {e}")