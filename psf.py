import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to load the image. Check the file path.")
    return image

def crop_to_point(image, center, size):
    """Crop a region around the point source."""
    x, y = center
    half_size = size // 2
    cropped = image[y-half_size:y+half_size, x-half_size:x+half_size]
    return cropped

def normalize_psf(psf):
    """Normalize the PSF to have a maximum value of 1."""
    return psf / np.max(psf)

def compute_psf(image, center, size):
    """Compute the Point Spread Function (PSF) around a point source."""
    cropped = crop_to_point(image, center, size)
    normalized = normalize_psf(cropped)
    return normalized

def visualize_psf(psf):
    """Visualize the PSF in 2D and 3D."""
    plt.figure(figsize=(12, 6))

    # 2D visualization
    plt.subplot(1, 2, 1)
    plt.imshow(psf, cmap='hot', extent=[-psf.shape[1]//2, psf.shape[1]//2, -psf.shape[0]//2, psf.shape[0]//2])
    plt.title("2D PSF")
    plt.colorbar(label="Intensity")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    # 3D visualization
    ax = plt.subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(range(psf.shape[1]), range(psf.shape[0]))
    ax.plot_surface(X, Y, psf, cmap='hot')
    ax.set_title("3D PSF")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Intensity")

    plt.tight_layout()
    plt.show()

def main(image_path, center, size=50):
    """Main function to compute and visualize the PSF."""
    # Load the image
    image = load_image(image_path)

    # Compute the PSF
    psf = compute_psf(image, center, size)

    # Visualize the PSF
    visualize_psf(psf)

if __name__ == "__main__":
    # Replace 'uploaded_image.jpg' with the actual uploaded image file path
    image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"

    # Define the approximate location of the point source in (x, y) format
    #below, use point source locator .py file to find coordinates; plug them in
    point_center = (1001, 1651)  # Replace with the actual coordinates of the point source

    # Define the size of the region to crop around the point source
    #for now, try different numbers until it fits; default was 50
    crop_size = 100  # Adjust based on the size of the point source in the image

    # Run the main function
    main(image_path, point_center, crop_size)