import os
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    """Load all images from a folder."""
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def find_brightest_point(image):
    """Find the brightest point in the image."""
    y, x = np.unravel_index(np.argmax(image), image.shape)
    return x, y

def extract_psf(image, point, psf_size=15):
    """Extract the PSF around the brightest point with a fixed region of interest."""
    x, y = point
    half_size = psf_size // 2

    # Ensure the PSF region is within the image boundaries
    x_start = max(x - half_size, 0)
    x_end = min(x + half_size + 1, image.shape[1])
    y_start = max(y - half_size, 0)
    y_end = min(y + half_size + 1, image.shape[0])

    # Extract the PSF region
    psf = image[y_start:y_end, x_start:x_end]

    # If the PSF region is smaller than the desired size, pad it with zeros
    if psf.shape != (psf_size, psf_size):
        pad_x = (psf_size - psf.shape[1]) // 2
        pad_y = (psf_size - psf.shape[0]) // 2
        psf = np.pad(psf, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')

    return psf

def compute_average_psf(images, psf_size=15):
    """Compute the average PSF from the brightest points in all images."""
    psfs = []
    for img in images:
        point = find_brightest_point(img)
        psf = extract_psf(img, point, psf_size)
        psfs.append(psf)
    return np.mean(psfs, axis=0)

def compute_mtf(psf):
    """Compute the MTF from the PSF."""
    psf_normalized = psf / np.sum(psf)  # Normalize the PSF
    mtf = np.abs(fft2(psf_normalized))
    mtf = fftshift(mtf)
    return mtf

def apply_deconvolution(image, mtf, noise_var=0.01, epsilon=1e-8):
    """Apply Wiener deconvolution using the MTF."""
    mtf_shifted = fftshift(mtf)
    H = mtf_shifted
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2

    # Wiener filter formula with epsilon to avoid division by zero
    wiener_filter = H_conj / (H_mag_sq + noise_var / (np.abs(image) ** 2 + epsilon))

    # Apply the Wiener filter in the frequency domain
    image_fft = fft2(image)
    deconvolved_fft = image_fft * wiener_filter
    deconvolved_image = np.abs(ifft2(deconvolved_fft))

    return deconvolved_image

def main():
    folder = r'C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images'  # Replace with your folder path
    new_image_path = r'C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png'  # Replace with your new image path
    psf_size = 15  # Size of the PSF window (must be odd)

    # Load images from the folder
    images = load_images_from_folder(folder)

    # Compute the average PSF
    average_psf = compute_average_psf(images, psf_size)

    # Compute the MTF from the average PSF
    mtf = compute_mtf(average_psf)

    # Load the new image
    new_image = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply deconvolution to the new image using the computed MTF
    deconvolved_image = apply_deconvolution(new_image, mtf)

    # Display the original and deconvolved images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(new_image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Average PSF')
    plt.imshow(average_psf, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Deconvolved Image')
    plt.imshow(deconvolved_image, cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()