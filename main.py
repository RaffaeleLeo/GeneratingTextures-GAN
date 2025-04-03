# main.py - Texture Synthesis with GAN (inference-only demo)

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.utils as vutils
from torchvision import transforms
import os


class Config:
    img_size: int = 256
    depth: int = 4
    outputFolder: str = "/content/gdrive/MyDrive/Masters Semester 2/Texture Synthesis/Experimental Results/Github Stuff"
    nz: int = 96
    zx: int = 6

opt = Config()

def setNoise(noise):
    noise=noise.detach()*1.0
    noise.uniform_(-1, 1)
    return noise

# ----------------------
# Generator Model
# ----------------------
# @title Generator
class Generator_SGAN4(nn.Module):

    def __init__(self, noise_channels=20):
      super(Generator_SGAN4, self).__init__()
      '''
      # :param ngf is channels of first layer, doubled up after every stride operation, or halved after upsampling
      # :param nDep is depth, both of decoder and of encoder
      # :param nz is dimensionality of stochastic noise
      '''
      in_features = noise_channels
      layers = []

      # Use upsampling to get a stride of 1/2
      # Initial convolution block
      out_features = 256
      model = [
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.Conv2d(in_features, out_features,  5, 1, 2),
          nn.BatchNorm2d(out_features),
          nn.ReLU(inplace=True),
      ]
      in_features = out_features

      # layer 2, half the output channels for each layer
      # Keeping a stride of 1/2 for each layer so I continue to upsample
      out_features = out_features // 2
      model += [
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.Conv2d(in_features, out_features,  5, 1, 2),
          nn.BatchNorm2d(out_features),
          nn.ReLU(),
      ]
      in_features = out_features

      # layer 3
      out_features = out_features // 2
      model += [
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.Conv2d(in_features, out_features,  5, 1, 2),
          nn.BatchNorm2d(out_features),
          nn.ReLU(),
      ]
      in_features = out_features

      # output layer
      # output features are always 3 for the image
      out_features = 3
      model += [
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.Conv2d(in_features, out_features,  5, 1, 2),
          nn.Tanh(),
      ]

      self.G = nn.Sequential(*model)

    def forward(self, input):
        return self.G(input)
    


# ----------------------
# Generate and Save Image
# ----------------------
def generate_and_save(generator_path: str, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NZ = opt.img_size//2**opt.depth
    print('length and width of noise: ', NZ)

    # Initialize and load generator
    generator = Generator_SGAN4().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    # Here we can decide how large to make the output image, at NZ*4 the image will be 512x512
    large_single_noise = torch.FloatTensor(1, opt.noise_channels, NZ*4,NZ*4)
    large_single_noise=setNoise(large_single_noise)
    large_single_noise=large_single_noise.to(device)
    with torch.no_grad():
        single_image=generator(large_single_noise)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save image
    vutils.save_image(single_image, output_path, normalize=True)
    print(f"Generated image saved to {output_path}")

# ----------------------
# Main Entry Point
# ----------------------
if __name__ == "__main__":
    generate_and_save("generator.pth", "outputs/generated.png")
