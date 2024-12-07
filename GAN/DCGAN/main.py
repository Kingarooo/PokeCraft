import torch
import torch.nn as nn
import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from utils import initialize_weights
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 200

FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose([
  transforms.Resize(IMAGE_SIZE),
  transforms.ToTensor(),
  transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])

])

dataset = datasets.ImageFolder(root="../data/processed", transform=transforms)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

if os.path.exists("gen.pth"):
  gen.load_state_dict(torch.load("gen.pth"))
  print("Generator loaded")
if os.path.exists("disc.pth"):
  disc.load_state_dict(torch.load("disc.pth"))
  print("Discriminator loaded")

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)


writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
gen.train()
disc.train()


for epoch in range(NUM_EPOCHS):
  for batch_idx, (real, _) in enumerate(loader):
    real = real.to(device)
    noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
    fake = gen(noise)

    disc_real = disc(real).reshape(-1)
    loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
    disc_fake = disc(fake).reshape(-1)
    loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_disc_real + loss_disc_fake) / 2
    disc.zero_grad()
    loss_disc.backward(retain_graph=True)
    opt_disc.step()

    output = disc(fake).reshape(-1)
    loss_gen = criterion(output, torch.ones_like(output))
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    if batch_idx % 50 == 0:
      print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

      with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)
        #save the model
        torch.save(gen.state_dict(), "gen.pth")
        torch.save(disc.state_dict(), "disc.pth")

      step += 1

