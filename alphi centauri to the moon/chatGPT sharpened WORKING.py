import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2, ifft2
from scipy.ndimage import zoom

def load_images_from_folder(folder_path):
    """Loads all images in a folder."""
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def find_brightest_point(image):
    """Finds the brightest point in an image."""
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    return max_loc  # (x, y) coordinates

def extract_roi(image, center, size=64):
    """Extracts an ROI around the given center point."""
    x, y = center
    half_size = size // 2
    h, w = image.shape
    x1, x2 = max(0, x - half_size), min(w, x + half_size)
    y1, y2 = max(0, y - half_size), min(h, y + half_size)
    return image[y1:y2, x1:x2]

def compute_mtf(image):
    """Computes the MTF from an image's FFT magnitude."""
    fft_image = fft2(image)
    mtf = np.abs(fftshift(fft_image))
    mtf /= np.max(mtf)  # Normalize
    return mtf

def resize_mtf(mtf, target_shape):
    """Resizes the MTF to match a target shape using interpolation."""
    scale_y = target_shape[0] / mtf.shape[0]
    scale_x = target_shape[1] / mtf.shape[1]
    return zoom(mtf, (scale_y, scale_x))

def apply_mtf_convolution(image, mtf):
    """Applies the MTF convolution to an image."""
    fft_image = fft2(image)
    convolved = fft_image * mtf
    sharpened = np.abs(ifft2(convolved))
    return sharpened

def main():
    # Load images and compute average MTF
    folder_path = r'C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images'  # Change to the actual path
    images = load_images_from_folder(folder_path)

    mtfs = []
    for img in images:
        center = find_brightest_point(img)
        roi = extract_roi(img, center)
        mtf = compute_mtf(roi)
        mtfs.append(mtf)

    avg_mtf = np.mean(mtfs, axis=0)  # Compute average MTF

    # Load second image and apply convolution
    second_image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png"  # Change as needed
    second_image = cv2.imread(second_image_path, cv2.IMREAD_GRAYSCALE)

    resized_mtf = resize_mtf(avg_mtf, second_image.shape)
    convolved_image = apply_mtf_convolution(second_image, resized_mtf)

    # Display results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(second_image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("MTF")
    plt.imshow(resized_mtf, cmap="hot")

    plt.subplot(1, 3, 3)
    plt.title("Sharpened Image")
    plt.imshow(convolved_image, cmap="gray")

    plt.show()

if __name__ == "__main__":
    main()