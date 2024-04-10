import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

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
    # Define argparse to accept image path
    parser = argparse.ArgumentParser(description='Denoising Autoencoder')
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    # Load the pre-trained model
    DAE = DenoisingAE()

    # Load the preprocessed image and pass it through the inference function
    denoised_output = denoise_image(args.image_path)

    # Convert the output tensor to a numpy array and remove the batch dimension
    output_image = denoised_output.squeeze(0).cpu().numpy()

    # Rescale values from [-1, 1] to [0, 1]
    output_image = (output_image + 1) / 2.0

    # Transpose the dimensions to match the expected format for displaying using matplotlib
    output_image = np.transpose(output_image, (1, 2, 0))

    # Display the output image
    plt.imshow(output_image)
    plt.axis('off')  # Turn off axis
    plt.show()
