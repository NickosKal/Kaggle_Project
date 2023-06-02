import os
import random
import shutil
import cv2
import numpy as np

# Set the paths for the source folder and the destination folder
source_folder = 'C:/Users/emama/OneDrive/Desktop/Uppsala/Semester 2 Period 2/Deep Learning for Image Analysis/project/train copy/positive'  # Replace with your source folder path
destination_folder = 'C:/Users/emama/OneDrive/Desktop/Uppsala/Semester 2 Period 2/Deep Learning for Image Analysis/project/train_mini4/positive'  # Replace with your destination folder path

# Set the number of images you want to sample
sample_size = 12500

# Set the range of rotation angles (in degrees)
rotation_angles = [0, 90, 180, 270]

# Set the range of Gaussian noise strengths (mean and standard deviation)
# noise_mean = 0
# noise_std = 10

# Get a list of all image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg'))]

# Randomly select a sample of images
sample_images = random.sample(image_files, sample_size)

# Augment and copy the sample images to the destination folder
for image_file in sample_images:
    source_path = os.path.join(source_folder, image_file)
    destination_path = os.path.join(destination_folder, image_file)

    # Load the image using OpenCV
    image = cv2.imread(source_path)

    # Normalize the image
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Randomly rotate the normalized image
    rotation_angle = random.choice(rotation_angles)
    rotated_image = np.rot90(image, rotation_angle // 90)

    def add_gaussian_blur(img):
        possible_kernels = [(11, 11), (5, 5), (3, 3), (7, 7), (13, 13)]
        blurred = cv2.GaussianBlur(img, random.choice(possible_kernels), 0)
        return blurred

    # Add Gaussian blur to the rotated image
    blurred_image = add_gaussian_blur(rotated_image)

    # Save the augmented image to the destination folder
    # cv2.imwrite(destination_path, blurred_image)

print(f'Sample dataset created with {sample_size} augmented images.')
