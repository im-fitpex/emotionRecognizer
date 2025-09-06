import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from model import EmotionCNN

EMOTIONS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Используем max_len = 150 как в dataset.py
max_len = 150
model = EmotionCNN(num_classes=8, max_len=max_len).to(device)
model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location=device))
model.eval()

def predict(file, plot_mfcc=True):
    # Загружаем аудио
    y, sr = librosa.load(file, sr=16000)
    
    # Вычисляем MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Padding или обрезка
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # Конвертируем в тензор
    X = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,40,max_len]

    # Предсказание
    with torch.no_grad():
        outputs = model(X)
        _, pred = torch.max(outputs, 1)

    # Визуализация MFCC
    if plot_mfcc:
        plt.figure(figsize=(10,4))
        plt.imshow(mfcc, aspect='auto', origin='lower')
        plt.title(f"Predicted Emotion: {EMOTIONS[pred.item()]}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return EMOTIONS[pred.item()]

if __name__ == "__main__":
    # Пример
    audio_file = "test/03-01-05-01-02-02-19.wav"  # укажи свой путь
    print("Emotion:", predict(audio_file))
