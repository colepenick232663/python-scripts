import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load the uploaded image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to load the image. Check the file path.")
    return image

def crop_to_roi(image, roi):
    """Crop the image to the region of interest (ROI)."""
    x, y, w, h = roi
    return image[y:y+h, x:x+w]

def compute_esf(edge_image):
    """Compute the Edge Spread Function (ESF)."""
    # Detect the edge using Sobel operator or Canny
    sobel = cv2.Sobel(edge_image, cv2.CV_64F, 1, 0, ksize=3)
    edge = np.mean(np.abs(sobel), axis=0)  # Average along the rows
    return edge

def compute_mtf(esf):
    """Compute the MTF from the ESF."""
    # Perform Fourier Transform of the ESF
    fft = np.fft.fft(esf)
    mtf = np.abs(fft) / np.max(np.abs(fft))  # Normalize
    return mtf

def plot_mtf(mtf):
    """Plot the MTF."""
    freqs = np.fft.fftfreq(len(mtf))
    plt.figure(figsize=(8, 5))
    plt.plot(freqs[:len(mtf)//2], mtf[:len(mtf)//2])  # Only positive frequencies
    plt.title("Modulation Transfer Function (MTF)")
    plt.xlabel("Spatial Frequency (cycles/pixel)")
    plt.ylabel("MTF")
    plt.grid()
    plt.show()

def main(image_path, roi=(50, 50, 300, 300)):
    """Main function to process the image and compute MTF."""
    image = load_image(image_path)
    roi_image = crop_to_roi(image, roi)
    
    plt.imshow(roi_image, cmap='gray')
    plt.title("Region of Interest (ROI)")
    plt.show()

    esf = compute_esf(roi_image)
    mtf = compute_mtf(esf)

    plot_mtf(mtf)

if __name__ == "__main__":
    # Replace 'uploaded_image.jpg' with the actual uploaded image file path
    image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"
    main(image_path)