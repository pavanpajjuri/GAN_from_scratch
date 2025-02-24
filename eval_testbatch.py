#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:43:18 2025

@author: pavanpaj
"""

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from dcgan import G
from torchvision import transforms

# Quantitative metrics
from torchmetrics.image.inception import InceptionScore
# from torchmetrics.image.fid import FrechetInceptionDistance  # Uncomment if you have a set of real images

# Device setup (using MPS if available, otherwise CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using Device:", device)

# Load the saved checkpoint for the generator
checkpoint_path = './models/checkpoint_epoch_24_colab.pth'  # Update with your checkpoint path
checkpoint = torch.load(checkpoint_path, map_location=device)

netG = G().to(device)
netG.load_state_dict(checkpoint['netG_state_dict'])
netG.eval()

# Generate 64 samples
num_samples = 64
noise = torch.randn(num_samples, 100, 1, 1, device=device)
with torch.no_grad():
    fake_images = netG(noise)  # Shape: (64, 3, H, W)

# Denormalize the images: from [-1, 1] to [0,1]
fake_images_denorm = (fake_images + 1) / 2  
# Convert to uint8 (values [0,255]) as expected by the Inception metric
fake_images_uint8 = (fake_images_denorm * 255).to(torch.uint8)

# Display the generated images in a grid using torchvision's make_grid
grid = vutils.make_grid(fake_images_denorm, nrow=8, padding=2, normalize=False)
# Convert the grid to numpy for plotting; note that grid is (3, H, W)
np_grid = grid.cpu().numpy().transpose((1, 2, 0))
plt.figure(figsize=(8, 8))
plt.imshow(np_grid)
plt.title("Generated Images Grid")
plt.axis("off")
plt.show()

# Compute the Inception Score (IS)
# The metric expects images of shape (N, 3, H, W) with dtype=torch.uint8 and pixel values in [0,255]
inception_metric = InceptionScore().to("cpu")
is_score = inception_metric(fake_images_uint8.cpu())
if isinstance(is_score, tuple):
    is_value = is_score[0]
else:
    is_value = is_score
print(f"Inception Score: {is_value:.4f}")

# Optionally, compute the Fréchet Inception Distance (FID) if you have real images available.
# For example:
# real_images = ...  # load a batch of real images as a tensor (N, 3, H, W) with dtype=torch.uint8
# fid_metric = FrechetInceptionDistance().to("cpu")
# fid_score = fid_metric(fake_images_uint8.cpu(), real_images)
# print(f"FID Score: {fid_score:.4f}")
