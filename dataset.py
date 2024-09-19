import numpy as np
import os
from PIL import Image

def load_images(path):
    # This function should load images and labels from the given path
    images = []
    labels = []
    
    for filename in os.listdir(path):
        if filename.endswith(".png"):  # Assuming images are PNGs
            img = Image.open(os.path.join(path, filename)).convert('L')  # Convert to grayscale
            img = img.resize((20, 20))  # Resize to 20x20
            images.append(np.array(img).flatten())  # Flatten the image
            label = int(filename[0])  # Assuming the filename starts with the label (e.g., 0_1.png)
            labels.append(label)
    
    return np.array(images), np.array(labels)
