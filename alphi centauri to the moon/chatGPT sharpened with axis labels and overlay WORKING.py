import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2, ifft2
from scipy.ndimage import zoom

def load_images_from_folder(folder_path):
    """Loads all images from a folder."""
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def get_roi_around_brightest(image, roi_size=50):
    """Finds the brightest point in the image and extracts a centered ROI."""
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    x, y = max_loc

    half_size = roi_size // 2
    x1, x2 = max(0, x - half_size), min(image.shape[1], x + half_size)
    y1, y2 = max(0, y - half_size), min(image.shape[0], y + half_size)

    return image[y1:y2, x1:x2]

def compute_mtf(image):
    """Computes the MTF using FFT magnitude."""
    fft_image = fft2(image)
    mtf = np.abs(fftshift(fft_image))
    mtf /= np.max(mtf)  # Normalize to 1
    return mtf

def compute_average_mtf(images, roi_size=50):
    """Computes the average MTF across multiple images using ROI centered at the brightest point."""
    mtfs = [compute_mtf(get_roi_around_brightest(img, roi_size)) for img in images]
    avg_mtf = np.mean(mtfs, axis=0)  # Compute average MTF
    return avg_mtf

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

def overlay_difference(original, corrected):
    """Generates an overlay of the original and corrected images to highlight differences."""
    difference = np.clip(corrected - original, 0, 255)
    overlay = cv2.addWeighted(original.astype(np.uint8), 0.5, difference.astype(np.uint8), 0.5, 0)
    return overlay

def main():
    # Load images from folder and compute average MTF
    folder_path = r'C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images'  # Change this to actual folder path
    images = load_images_from_folder(folder_path)

    if not images:
        print("No images found in the folder.")
        return

    avg_mtf = compute_average_mtf(images, roi_size=50)

    # Load second image and apply convolution
    second_image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png"  # Change as needed
    second_image = cv2.imread(second_image_path, cv2.IMREAD_GRAYSCALE)

    if second_image is None:
        print("Failed to load second image.")
        return

    resized_mtf = resize_mtf(avg_mtf, second_image.shape)
    convolved_image = apply_mtf_convolution(second_image, resized_mtf)

    # Generate overlay image to highlight changes
    overlay_image = overlay_difference(second_image, convolved_image)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.imshow(second_image, cmap="gray")
    plt.colorbar()
    plt.axis("on")

    plt.subplot(1, 3, 2)
    plt.title("Resized MTF")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.imshow(resized_mtf, cmap="hot")
    plt.colorbar()
    plt.axis("on")

    plt.subplot(1, 3, 3)
    plt.title("Corrected Image")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.imshow(convolved_image, cmap="gray")
    plt.colorbar()
    plt.axis("on")

    plt.show()

    # Display overlay difference
    plt.figure(figsize=(6, 5))
    plt.title("Overlay of Original and Corrected Image")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.imshow(overlay_image, cmap="coolwarm")
    plt.colorbar()
    plt.axis("on")
    
    plt.show()

if __name__ == "__main__":
    main()