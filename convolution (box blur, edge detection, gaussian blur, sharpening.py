import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to load the image. Check the file path.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def apply_convolution(image, kernel):
    """Apply a convolution operation to the image using the specified kernel."""
    return cv2.filter2D(image, -1, kernel)

def display_images(original, processed, title="Processed Image"):
    """Display the original and processed images side by side."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title(title)
    plt.axis("off")
    plt.show()

def main(image_path, kernel, kernel_name="Custom Kernel"):
    """Main function to load the image, apply convolution, and display results."""
    # Load the image
    original_image = load_image(image_path)

    # Apply convolution
    processed_image = apply_convolution(original_image, kernel)

    # Display results
    display_images(original_image, processed_image, title=f"Image with {kernel_name}")

if __name__ == "__main__":
    # Replace 'uploaded_image.jpg' with the actual uploaded image file path
    image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\thin vbars\thin vbars.jpg"
    
    # Example kernels
    kernels = {
        "Edge Detection": np.array([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]]),
        
        "Sharpening": np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]]),
        
        "Box Blur": np.ones((3, 3), dtype=np.float32) / 9.0,
        
        "Gaussian Blur": np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=np.float32) / 16.0
    }
    
    for kernel_name, kernel in kernels.items():
        print(f"Applying {kernel_name} kernel...")
        main(image_path, kernel, kernel_name)
