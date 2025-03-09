import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Initializing W&B
wandb.init(project="DA6401 - Assignment 1", name="fashion-mnist-visualization")

# Loading dataset
(train_images, train_labels), (_, _) = fashion_mnist.load_data()

# Class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Selecting one image per class
samples = {}
for img, label in zip(train_images, train_labels):
    if label not in samples:
        samples[label] = img
    if len(samples) == 10:
        break

# Plotting images in a grid
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, (label, img) in zip(axes.flatten(), samples.items()):
    ax.imshow(img, cmap='gray')
    ax.set_title(class_names[label])
    ax.axis("off")

plt.tight_layout()

# Logging the image to W&B
wandb.log({"Fashion-MNIST Samples": wandb.Image(fig)})

wandb.finish()
