import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from src.model import EmotionCNN   # если model.py в папке src

# Эмоции
EMOTIONS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# фикс длины как в dataset.py
max_len = 150

# Загружаем модель
model = EmotionCNN(num_classes=8, max_len=max_len).to(device)
model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location=device))
model.eval()

def predict(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # паддинг или обрезка
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    X = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        _, pred = torch.max(outputs, 1)

    return EMOTIONS[pred.item()], mfcc, probs

# ---------------- GUI ----------------
st.title("🎤 Emotion Recognition from Voice")
st.write("Загрузите .wav файл и получите предсказанную эмоцию + MFCC график")

uploaded_file = st.file_uploader("Выберите аудиофайл", type=["wav"])

if uploaded_file is not None:
    # проигрывание
    st.audio(uploaded_file, format="audio/wav")

    # временный файл
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # предсказание
    emotion, mfcc, probs = predict("temp.wav")

    # вывод эмоции
    st.success(f"🧠 Определённая эмоция: **{emotion}**")

    # MFCC график
    fig, ax = plt.subplots(figsize=(8, 4))
    img = ax.imshow(mfcc, aspect="auto", origin="lower")
    ax.set_title(f"MFCC (Predicted: {emotion})")
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

    # Барчарт вероятностей
    st.subheader("📊 Вероятности по всем эмоциям")
    st.bar_chart({EMOTIONS[i]: probs[i] for i in range(len(EMOTIONS))})

    # Убираем временный файл
    os.remove("temp.wav")