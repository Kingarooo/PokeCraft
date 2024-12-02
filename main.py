import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from torch.utils.data import Dataset
import utils # Importando funções auxiliares
from utils import CustomImageFolder
from generator import Generator
from discriminator import Discriminator

# Parâmetros principais
latent_dim = 100  # Dimensão do vetor de entrada do gerador
img_size = 128  # Tamanho das imagens (128x128)
channels = 4  # 4 canais RGBA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicialização dos modelos
noise_dim = 100
channels_img = 4  # RGBA
device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator(noise_dim, channels_img).to(device)
disc = Discriminator(channels_img).to(device)

gen_path = "generator.pth"
disc_path = "discriminator.pth"

# Função para carregar o modelo, se existir
if os.path.exists(gen_path) and os.path.exists(disc_path):
    print("Carregando modelos salvos...")
    gen.load_state_dict(torch.load(gen_path))
    disc.load_state_dict(torch.load(disc_path))
else:
    print("Nenhum modelo salvo encontrado. Treinamento será iniciado do zero.")



criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Transformações
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensiona para 64x64
    transforms.ToTensor(),       # Converte para tensor PyTorch
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))  # Normaliza
])

dataset = CustomImageFolder(image_folder="data/images/all", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


if os.path.exists('generator.pth'):
    gen.load_state_dict(torch.load('gen.pth'))
    gen.eval()
    print('Generator loaded')
if os.path.exists('discriminator.pth'):
    disc.load_state_dict(torch.load('disc.pth'))
    disc.eval()
    print('Discriminator loaded')


# Função de treinamento
epochs = 100
for epoch in range(epochs):
    for batch in dataloader:
        real = batch.to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

        # Treinar Discriminador
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Treinar Gerador
        gen_updates = 2
        for i in range(gen_updates):
                fake = gen(noise)
                output = disc(fake).view(-1)
                loss_gen = criterion(output, torch.ones_like(output))

                opt_gen.zero_grad()
                loss_gen.backward(retain_graph=(i < gen_updates - 1))  # Retain graph for all but the last backward call
                opt_gen.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    # Salvar imagens geradas a cada epoch
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = gen(noise).cpu()
            save_image(fake, f"generated_epoch_{epoch+1}.png", normalize=True)
            torch.save(gen.state_dict(), gen_path)
            torch.save(disc.state_dict(), disc_path)
            print(f"Modelos salvos após a época {epoch+1}.")