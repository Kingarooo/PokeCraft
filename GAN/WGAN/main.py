from sys import argv
import torch
import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image
from utils import initialize_weights, gradient_penalty
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 500
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose([
  transforms.Resize(IMAGE_SIZE),
  transforms.ToTensor(),
  transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])

])

dataset = datasets.ImageFolder(root="../data/processed", transform=transforms)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

#handle arguments
if argv[1] == 'clean':

  if os.path.exists("gen.pth"):
    os.remove("gen.pth")
  if os.path.exists("disc.pth"):
    os.remove("disc.pth")
  if os.path.exists("logs"):
    os.system("rm -r logs")
  print("Models cleaned")
elif argv[1] == 'generate':
  #argv 2 must be a path to model
  if not os.path.exists(argv[2]):
    print("Model not found")
    exit()
  gen.load_state_dict(torch.load(argv[2]))
  if not os.path.exists("fake"):
    os.mkdir("fake")
  gen.eval()
  for i in range(10):
    with torch.no_grad():
      noise = torch.randn(1,Z_DIM, 1, 1).to(device)
      generated_image = gen(noise)
    save_image(generated_image,f"fake/generated_image_{i}_.png")
    print(f"Saved image in fake/generated_image_{i}_.png")
  exit()
   
if os.path.exists("gen.pth"):
  gen.load_state_dict(torch.load("gen.pth"))
  print("Generator loaded")
if os.path.exists("disc.pth"):
  critic.load_state_dict(torch.load("disc.pth"))
  print("Discriminator loaded")

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)


writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
gen.train()
critic.train()


for epoch in range(NUM_EPOCHS):
  for batch_idx, (real, _) in enumerate(loader):
    real = real.to(device)
    cur_batch_size = real.shape[0]
    for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device) #type: ignore
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()


    output = critic(fake).reshape(-1)
    loss_gen = -torch.mean(output)
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()


    #save model per 10 epochs
    if epoch % 10 == 0:
      torch.save(gen.state_dict(), "gen.pth")
      torch.save(critic.state_dict(), "disc.pth")

    if batch_idx % 100 == 0:
      print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")

      with torch.no_grad():

        fake = gen(fixed_noise)
        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)


      step += 1

