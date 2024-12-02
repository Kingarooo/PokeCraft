import numpy as np
from PIL import Image
import torch.nn as nn
from PIL import Image
from torchvision.datasets import ImageFolder
import torch
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função de ativação Tanh
def tanh(x):
    return np.tanh(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)  # Inicialização normal
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-7  # Evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping antes do log
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)


class CustomImageFolder(ImageFolder):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGBA")  # Garantir 4 canais (RGBA)
        if self.transform:
            image = self.transform(image)
        return image