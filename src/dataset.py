import torch
import librosa
from torch.utils.data import Dataset
import os
import numpy as np

class RavdessDataset(Dataset):
    def __init__(self, root, max_len=150):
        self.files = []
        self.max_len = max_len
        for actor_folder in os.listdir(root):
            actor_path = os.path.join(root, actor_folder)
            if os.path.isdir(actor_path):
                for f in os.listdir(actor_path):
                    if f.endswith(".wav"):
                        self.files.append(os.path.join(actor_path, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        y, sr = librosa.load(file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # padding / обрезка по max_len
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = torch.tensor(np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant'))
        else:
            mfcc = torch.tensor(mfcc[:, :self.max_len])
        mfcc = mfcc.unsqueeze(0).float()  # [1, 40, max_len]

        label = int(os.path.basename(file).split("-")[2]) - 1
        return mfcc, label
