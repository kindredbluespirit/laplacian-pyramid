import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys

def gaussian_pyramid(image, levels):
    gp = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gp.append(image)
    return gp

def laplacian_pyramid(image, levels):
    gp = gaussian_pyramid(image, levels)
    lp = []
    for i in range(levels, 0, -1):
        upsampled = cv2.pyrUp(gp[i], dstsize=(gp[i-1].shape[1], gp[i-1].shape[0]))
        laplacian = cv2.subtract(gp[i-1], upsampled)
        lp.append(laplacian)
    lp.append(gp[-1])  # The smallest Gaussian level
    return lp

def display_pyramid(pyramid, title="Pyramid Levels"):
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(pyramid):
        plt.subplot(1, len(pyramid), i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Level {i}')
    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        sys.exit(1)
    
    levels = 4
    
    # Generate Laplacian Pyramid
    laplacian_pyr = laplacian_pyramid(image, levels)
    
    # Display Laplacian Pyramid
    display_pyramid(laplacian_pyr, title="Laplacian Pyramid")

