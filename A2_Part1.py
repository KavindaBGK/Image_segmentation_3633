import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std_dev=10):
    row, col = image.shape
    gauss = np.random.normal(mean, std_dev, (row, col)).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return noisy_image

# Load the image
image = cv2.imread('objects.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Add Gaussian noise
noisy_image = add_gaussian_noise(gray_image)

# Apply Otsu's thresholding
_, binary_image = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display original, noisy, and binary images
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Noisy Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Binary Image (Otsu)', cv2.WINDOW_NORMAL)

cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Binary Image (Otsu)', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()