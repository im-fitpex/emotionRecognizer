import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import RavdessDataset
from model import EmotionCNN
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем датасет
dataset = RavdessDataset(root="data")  # под Windows путь к папке data
train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = EmotionCNN(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # tqdm прогрессбар по батчам
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss/((loop.n)+1))

    print(f"Epoch {epoch+1}, Average Loss: {running_loss/len(train_loader):.4f}")

# Сохраняем модель
torch.save(model.state_dict(), "models/emotion_cnn.pth")
print("Модель сохранена в models/emotion_cnn.pth")
