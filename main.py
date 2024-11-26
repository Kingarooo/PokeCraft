from torchvision import datasets, transforms
from discriminator import Discriminator
from torch.utils.data import DataLoader
from generator import Generator
from PIL import Image
import numpy as np
import kagglehub
import pickle
import utils
import sys
import os


# Parameters
latent_dim = 100  # Size of random noise input
image_dim = 128 * 128  # Assuming 128x128 RGB images
batch_size = 64
epochs = 10000
lr = 0.0002
target_size = (128,128)
real_images = []
images = []

path = kagglehub.dataset_download("kvpratama/pokemon-images-dataset")
path = path + "/pokemon/pokemon"

path2 = kagglehub.dataset_download("aaronyin/oneshotpokemon")
path2+= "/kaggle-one-shot-pokemon/pokemon-a"
processed_folder  =  "./data/images/all"


def take_data(paths):
    for file in os.listdir(processed_folder):
        return 
    images = []
    for id,path in enumerate(paths):
        for filename in os.listdir(path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # Open the image file
                img = Image.open(os.path.join(path, filename))
                
                # Resize the image to a consistent size
                img = img.resize(target_size)
                
                # Convert the image to grayscale (if needed) and then to a NumPy array
                img_array = np.array(img)

                # If the image is RGB, you can convert it to grayscale by averaging the channels
                if len(img_array.shape) == 3:  # If it's an RGB image
                    img_array = np.mean(img_array, axis=-1)  # Convert to grayscale by averaging channels
                
                # Flatten the image to a 1D array (e.g., 28x28 -> 784)
                img_array = img_array.flatten()
                #save image in processed folder
                img.save(processed_folder + "/" +str(id)+ "p-" + filename)
                
                # Append the image to the list
                images.append(img_array)

def handle_arguments():
    n = len(sys.argv)
    if n > 1:
        if sys.argv[1] == "clean":
            print("Removing model files")
            if os.path.exists("generator.pkl"):
                os.remove("generator.pkl")
            if os.path.exists("discriminator.pkl"):
                os.remove("discriminator.pkl")
            print("Model files removed")

def process_data():
    os.makedirs(processed_folder, exist_ok=True)
    take_data([path,path2])
    # Convert the list of images to a NumPy array
    real_images = np.array(images)

handle_arguments()

process_data()
# Check the shape of real_images to ensure they are consistent

gen = Generator(latent_dim, image_dim)
disc = Discriminator(image_dim)

# Path to your processed images
image_folder = "data/images"

# Transformations to apply to each image (e.g., normalization, resizing)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to PyTorch Tensor
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values to [-1, 1]
])

# Load dataset from folder
dataset = datasets.ImageFolder(root=image_folder, transform=transform)

# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#check if the model is already trained
if os.path.exists("generator.pkl"):
    with open("generator.pkl", "rb") as f:
        gen = pickle.load(f)
if os.path.exists("discriminator.pkl"):
    with open("discriminator.pkl", "rb") as f:
        disc = pickle.load(f)

for epoch in range(epochs):
    for _ in range(5):  # Train the discriminator 5 times for every 1 generator update
        # Get a batch of real images
        real_images, _ = next(iter(dataloader))
        real_images = real_images.view(batch_size, -1)
        real_labels = np.ones((batch_size, 1))

        # Generate a batch of fake images
        z = np.random.randn(batch_size, latent_dim)
        fake_images = gen.forward(z)
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator
        real_preds = disc.forward(real_images)
        fake_preds = disc.forward(fake_images)

        # Calculate the discriminator loss
        loss_disc = utils.binary_cross_entropy(real_labels, real_preds) + utils.binary_cross_entropy(fake_labels, fake_preds)

        # Backpropagate the discriminator
        disc.backward((real_preds - real_labels) / batch_size, lr)
        disc.backward((fake_preds - fake_labels) / batch_size, lr)

    # Train the generator
    fake_preds = disc.forward(fake_images)
    loss_gen = utils.binary_cross_entropy(real_labels, fake_preds)
    gen.backward((fake_preds - real_labels) / batch_size, lr)
    # Monitor the training progress
    print(f"Epoch {epoch}: Loss Disc: {loss_disc}, Loss Gen: {loss_gen}")
    if epoch % 10 == 0:  # Every 10 epochs
        utils.show_generated_images(fake_images, epoch)  # Show multiple images after each 10 epochs
        
        # Save models
        with open("generator.pkl", "wb") as f:
            pickle.dump(gen, f)

        with open("discriminator.pkl", "wb") as f:
            pickle.dump(disc, f)

