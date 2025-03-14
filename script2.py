import cv2
import numpy as np
import argparse
import os

def gaussian_pyramid(img, levels):
    # Create the Gaussian pyramid
    gaussian_pyramid = [img]
    for i in range(levels - 1):
        img = cv2.pyrDown(img)  # Downsample the image
        gaussian_pyramid.append(img)
    return gaussian_pyramid

def laplacian_pyramid(gaussian_pyramid):
    # Create the Laplacian pyramid
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        # Upsample the next level of the Gaussian pyramid and subtract from the current
        next_gaussian = cv2.pyrUp(gaussian_pyramid[i + 1])  # Upsample
        laplacian = cv2.subtract(gaussian_pyramid[i], next_gaussian)  # Laplacian is difference
        laplacian_pyramid.append(laplacian)
    
    # Append the last level of the Gaussian pyramid (no subtraction possible)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid

def save_pyramid(pyramid, base_filename, prefix=""):
    # Ensure the "results/" directory exists
    os.makedirs("results", exist_ok=True)

    # Save each level of the pyramid to the "results/" folder
    for i, img in enumerate(pyramid):
        filename = f"results/{base_filename}_{prefix}_level_{i}.png"
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate Laplacian and Gaussian pyramids from an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--levels", type=int, default=5, help="Number of pyramid levels (default: 5)")
    
    # Parse command line arguments
    args = parser.parse_args()

    # Load the image from the provided path
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Unable to load image from {args.image_path}")
        return

    # Create Gaussian and Laplacian pyramids
    gaussian_pyr = gaussian_pyramid(img, args.levels)
    laplacian_pyr = laplacian_pyramid(gaussian_pyr)

    # Save the Gaussian and Laplacian pyramids
    save_pyramid(gaussian_pyr, "gaussian_pyramid", "gaussian")
    save_pyramid(laplacian_pyr, "laplacian_pyramid", "laplacian")

if __name__ == "__main__":
    main()
