#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:43:18 2025

@author: pavanpaj
"""

# -*- coding: utf-8 -*-

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from dcgan import G



# Initialize the Generator
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")  # If using Mac GPU

# Load the saved checkpoint
checkpoint_path = './models/checkpoint_epoch_24.pth'  # Replace with the path to your saved checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)


netG = G().to(device)

# Load the generator weights from the checkpoint
netG.load_state_dict(checkpoint['netG_state_dict'])

# Set the generator to evaluation mode
netG.eval()

# Generate new samples
num_samples = 1  # Number of samples to generate
noise = torch.randn(num_samples, 100, 1, 1, device=device)  # Random noise as input
with torch.no_grad():  # Disable gradient calculation for inference
    fake_images = netG(noise)  # Generate images


# Convert the tensor to a numpy array and reshape for visualization
fake_images = fake_images.detach().cpu().numpy()  # Convert to numpy array
fake_images = np.transpose(fake_images, (0, 2, 3, 1))  # Change from (N, C, H, W) to (N, H, W, C) for matplotlib

# Denormalize the images (if they were normalized to [-1, 1])
fake_images = (fake_images + 1) / 2  # Scale to [0, 1]

# Display the single image using matplotlib
plt.imshow(fake_images[0])  # Display the first (and only) image
plt.axis('off')  # Hide axes
plt.show()

'''
# Display the images in a grid if num_samples = 64using matplotlib
def show_images(images, nrow=8):
    fig, axes = plt.subplots(nrows=images.shape[0] // nrow, ncols=nrow, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()

# Show the generated images
show_images(fake_images)
'''

## Save the generated images
#vutils.save_image(fake_images.detach().cpu(), './results/generated_samples.png', normalize=True)

#print("Generated samples saved to './results/generated_samples.png'")
