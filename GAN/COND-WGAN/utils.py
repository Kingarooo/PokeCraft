import torch.nn as nn
import torch
import os
from torchvision.utils import save_image

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic,labels, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images,labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def generate_image(type_label, gen, device, Z_DIM):
  if not os.path.exists("fake"):
    os.mkdir("fake")
  gen.eval()
  for i in range(10):
    with torch.no_grad():
      noise = torch.randn(1,Z_DIM, 1, 1).to(device)
      generated_image = gen(noise, type_label)
    save_image(generated_image,f"fake/generated_image_{i}_.png")
    print(f"Saved image in fake/generated_image_{i}_.png")