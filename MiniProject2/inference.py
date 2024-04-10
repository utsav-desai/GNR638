import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def denoise_image(image_path):
    # Preprocess the input image if needed
    # Assuming image is a torch tensor with shape [batch_size, channels, height, width]
    image = Image.open(image_path)

    # Resize the image while maintaining aspect ratio
    desired_size = (448, 256)
    image = image.resize(desired_size, Image.ANTIALIAS)

    # Convert to tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Apply preprocessing
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        denoised_image = DAE(image)
    return denoised_image

if __name__ == "__main__":
    # Define argparse to accept image folder and output folder paths
    parser = argparse.ArgumentParser(description='Denoising Autoencoder')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing input images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder to save output images')
    args = parser.parse_args()

    # Load the pre-trained model
    DAE = DenoisingAE()

    # Iterate over all images in the input folder
    for filename in os.listdir(args.image_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(args.image_folder, filename)
            print(f"Processing {image_path}...")

            # Denoise the image
            denoised_output = denoise_image(image_path)

            # Convert the output tensor to a numpy array and remove the batch dimension
            output_image = denoised_output.squeeze(0).cpu().numpy()

            # Rescale values from [-1, 1] to [0, 1]
            output_image = (output_image + 1) / 2.0

            # Transpose the dimensions to match the expected format for saving
            output_image = np.transpose(output_image, (1, 2, 0))

            # Save the output image
            output_path = os.path.join(args.output_folder, filename)
            print(output_image.shape)
            # plt.imsave(output_path, output_image)
            # save image using cv2
            #scale image from 0 to 1 to 0 to 255
            output_image = (output_image * 255).astype(np.uint8)
            cv2.imwrite(output_path, output_image)
            print(f"Saved denoised image as {output_path}")

    print("All images processed.")