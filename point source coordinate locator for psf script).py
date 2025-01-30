import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image in grayscale format.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to load the image. Check the file path.")
    return image

def find_brightest_pixel(image):
    """
    Find the coordinates of the brightest pixel in the image.
    Returns:
        coordinates: (x, y) tuple of the brightest pixel.
        max_intensity: Intensity value of the brightest pixel.
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    return max_loc, max_val

def main(image_path):
    """
    Main function to load the image and find the brightest pixel.
    """
    # Load the image
    image = load_image(image_path)

    # Find the brightest pixel
    coordinates, max_intensity = find_brightest_pixel(image)
    print(f"Brightest Pixel Coordinates: {coordinates}")
    print(f"Maximum Intensity Value: {max_intensity}")

    # Optional: Display the image with the point source highlighted
    highlighted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(highlighted_image, coordinates, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)

    # Show the image with the detected point source
    cv2.imshow("Point Source Detection", highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'uploaded_image.jpg' with the actual image path
    image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"
    main(image_path)