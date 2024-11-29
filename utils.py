import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função de ativação Tanh
def tanh(x):
    return np.tanh(x)

def initialize_weights(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])  # He initialization for ReLU


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def show_generated_images(images, epoch,num_images=16):
    """Shows multiple generated images in a grid format."""
    # Reshape images if needed (for example, 128x128x3 images)
    images = (images + 1) / 2 * 255 
    images = images[:num_images]  # Limit to num_images if there are more
    images = images.reshape(-1, 128, 128)  # Reshape to 128x128 RGB (assuming images are 128x128x3)
    images = images.astype(np.uint8)
    # Plotting the images in a grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # Adjust the grid size as needed

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    axes = axes.ravel()

    for i in range(num_images):
        img = images[i]  # Get a single image (assuming shape is [batch_size, height, width, channels])

        # Convert the image to a PIL Image object
        pil_img = Image.fromarray(img)

        # Resize the image if necessary (to target_size)

        # Save the image as a .png file
        pil_img.save(os.path.join(output_dir, f"epoch_{epoch}_image_{i}.png"))
    
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
    
    plt.show()