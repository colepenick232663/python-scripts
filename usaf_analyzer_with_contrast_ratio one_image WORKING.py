import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def calculate_contrast(profile, peaks, troughs, orientation=None):
    """Calculate contrast between peaks and troughs in the intensity profile."""
    contrasts = []
    
    # Ensure we have exactly 3 peaks if possible
    if len(peaks) > 3:
        # Find the 3 most prominent peaks
        peak_heights = [profile[p] for p in peaks]
        sorted_indices = np.argsort(peak_heights)[-3:]  # Get indices of 3 highest peaks
        peaks = [peaks[i] for i in sorted_indices]
        peaks.sort()  # Sort by position
    
    # Ensure we have exactly 2 troughs if possible
    if len(troughs) > 2:
        # Find the 2 most prominent troughs
        trough_depths = [profile[t] for t in troughs]
        sorted_indices = np.argsort(trough_depths)[:2]  # Get indices of 2 lowest troughs
        troughs = [troughs[i] for i in sorted_indices]
        troughs.sort()  # Sort by position
    
    # Calculate contrast for each peak-trough pair
    for peak in peaks:
        if len(troughs) > 0:
            closest_trough_idx = np.argmin(np.abs(troughs - peak))
            closest_trough = troughs[closest_trough_idx]
            
            peak_val = profile[peak]
            trough_val = profile[closest_trough]
            
            # Calculate Michelson contrast
            contrast = (peak_val - trough_val) / (peak_val + trough_val)
            
            contrasts.append((contrast, peak, closest_trough))
    
    return contrasts

def calculate_mtf_width(profile, peak, trough):
    """Calculate MTF width using the trough value as the tangent line."""
    peak_height = profile[peak]
    trough_height = profile[trough]
    
    # Use the trough value as the tangent line height
    tangent_height = trough_height
    
    x = np.arange(len(profile))
    interp_func = interp1d(x, profile, kind='cubic', bounds_error=False, fill_value=0)
    
    x_high_res = np.linspace(0, len(profile)-1, num=1000)
    y_high_res = interp_func(x_high_res)
    
    # Find where profile crosses the tangent line
    above_tangent = y_high_res > tangent_height
    crossings = np.where(np.diff(above_tangent))[0]
    
    if len(crossings) >= 2:
        left_idx = x_high_res[crossings[0]]
        right_idx = x_high_res[crossings[-1]]
        width = right_idx - left_idx
        
        # Calculate MTF directly
        mtf = 389.5901512 / width
        
        return width, mtf, tangent_height
    
    return None, None, None

def calculate_contrast_ratio(roi, roi_display):
    """Calculate contrast ratio between bright box and totally dark region."""
    # Ask user to select both bright and dark regions in one UI
    print("\nSelect bright and dark regions...")
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
    plt.title('First: Click two points to define the bright box\nThen: Click two more points to define the dark region')
    
    # Get 4 points (2 for bright, 2 for dark)
    all_points = plt.ginput(4, timeout=-1)
    plt.close()
    
    if len(all_points) != 4:
        return None
    
    # First two points define bright box
    bright_points = all_points[:2]
    # Last two points define dark region
    dark_points = all_points[2:]
    
    # Define bright box
    bx1, by1 = int(min(bright_points[0][0], bright_points[1][0])), int(min(bright_points[0][1], bright_points[1][1]))
    bx2, by2 = int(max(bright_points[0][0], bright_points[1][0])), int(max(bright_points[0][1], bright_points[1][1]))
    
    # Extract the bright region
    bright_region = roi[by1:by2, bx1:bx2]
    
    if len(dark_points) != 2:
        return None
    
    # Define dark region
    dx1, dy1 = int(min(dark_points[0][0], dark_points[1][0])), int(min(dark_points[0][1], dark_points[1][1]))
    dx2, dy2 = int(max(dark_points[0][0], dark_points[1][0])), int(max(dark_points[0][1], dark_points[1][1]))
    
    # Extract dark region
    dark_region = roi[dy1:dy2, dx1:dx2]
    
    # Calculate mean intensities
    bright_mean = np.mean(bright_region)
    dark_mean = np.mean(dark_region)
    
    # Calculate contrast ratio (corrected - no percentage)
    contrast_ratio = bright_mean / dark_mean
    
    # Draw boxes on visualization
    cv2.rectangle(roi_display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # Green for bright box
    cv2.putText(roi_display, "Bright", (bx1, by1-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.rectangle(roi_display, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)  # Blue for dark region
    cv2.putText(roi_display, "Dark", (dx1, dy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # No need to show a separate plot here as this information will be displayed later
    
    return contrast_ratio

def lp_to_degrees(group, element):
    """Convert USAF group/element to line pairs per degree."""
    lp_mm = 2**(group + (element - 1)/6)
    viewing_distance_mm = 250  # Standard viewing distance
    return lp_mm * (viewing_distance_mm * np.pi / 180)

def process_element(roi, roi_display, group, element_num, px, py):
    """Process a single USAF target element."""
    # Estimate element size based on group/element
    # USAF elements get smaller as group/element increases
    base_size = roi.shape[0] // 12  # Adjusted base size
    
    # Scaling factors for different groups - more precise scaling for elements 4-6
    if group <= 2:
        size_factor = 2.2 ** ((3 - group) / 2)  # Larger for groups 0-2
    elif group == 3:
        size_factor = 1.15  # Standard size for group 3
    elif group == 4:
        # Adjust size based on element number for group 4
        if element_num <= 3:
            size_factor = 0.8  # Larger for elements 1-3
        else:
            size_factor = 0.7  # Smaller for elements 4-6
    else:
        # Adjust size based on element number for group 5+
        if element_num <= 3:
            size_factor = 0.5 * (0.8 ** (group - 5))  # Larger for elements 1-3
        else:
            size_factor = 0.45 * (0.8 ** (group - 5))  # Smaller for elements 4-6
    
    element_size = max(int(base_size * size_factor), 10)  # Minimum size of 10 pixels
    
    # Extract region around the point
    half_size = element_size // 2
    x_start = max(0, px - half_size)
    y_start = max(0, py - half_size)
    x_end = min(roi.shape[1], px + half_size)
    y_end = min(roi.shape[0], py + half_size)
    
    # Extract the element region
    element_region = roi[y_start:y_end, x_start:x_end]
    
    # Skip if region is too small
    if element_region.shape[0] < 5 or element_region.shape[1] < 5:
        print(f"Group {group}, Element {element_num}: Region too small")
        return None
    
    # Detect if bars are horizontal or vertical
    # Calculate profiles in both directions
    h_profile = np.mean(element_region, axis=0)  # Average across columns for horizontal profile
    v_profile = np.mean(element_region, axis=1)  # Average across rows for vertical profile
    
    # Find peaks and troughs in both directions
    h_peaks, _ = find_peaks(h_profile, distance=2)
    h_troughs, _ = find_peaks(-h_profile, distance=2)
    v_peaks, _ = find_peaks(v_profile, distance=2)
    v_troughs, _ = find_peaks(-v_profile, distance=2)
    
    # Count features in both directions
    h_features = len(h_peaks) + len(h_troughs)
    v_features = len(v_peaks) + len(v_troughs)
    
    # Determine orientation based on which direction has more features
    # More features in horizontal profile means horizontal bars
    # More features in vertical profile means vertical bars
    if h_features > v_features:
        orientation = "horizontal"
        # For horizontal bars, use vertical profile (top to bottom)
        profile = v_profile
        peaks = v_peaks
        troughs = v_troughs
    else:
        orientation = "vertical"
        # For vertical bars, use horizontal profile (left to right)
        profile = h_profile
        peaks = h_peaks
        troughs = h_troughs
    
    # For higher groups (5+), check if we can clearly see 3 bars
    # We should have at least 3 peaks or 3 troughs
    min_features = 3
    if len(peaks) < min_features and len(troughs) < min_features:
        # Draw rectangle with red color to indicate failure
        color = (0, 0, 255)  # Red for failed elements
        cv2.rectangle(roi_display, (x_start, y_start), (x_end, y_end), color, 2)
        
        # Add a label with new format
        orient_label = "H" if orientation == "horizontal" else "V"
        label = f"G{group}E{element_num}{orient_label}"
        cv2.putText(roi_display, label, (x_start, y_start-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        print(f"Group {group}, Element {element_num} ({orientation}): Failed - cannot discern 3 bars clearly")
        return None
    
    # Calculate contrast with orientation information
    contrasts = calculate_contrast(profile, peaks, troughs, orientation)
    
    if not contrasts:
        return None
        
    # Get best contrast
    best_contrast, peak, trough = max(contrasts, key=lambda x: x[0])
    
    # Calculate MTF width using trough as tangent line
    width, mtf_value, tangent_height = calculate_mtf_width(profile, peak, trough)
    
    # Check if contrast is close to 30%
    target_contrast = 0.3
    contrast_diff = best_contrast - target_contrast
    
    # Use different colors based on contrast value
    if contrast_diff < -0.1:  # Significantly below 30%
        color = (0, 0, 255)  # Red for too low
    elif contrast_diff > 0.1:  # Significantly above 30%
        color = (255, 0, 255)  # Purple for too high
    elif abs(contrast_diff) <= 0.05:  # Within 5% of target
        color = (0, 255, 0)  # Green for good match
    else:  # Within 5-10% of target
        color = (0, 255, 255)  # Yellow for okay match
    
    # Draw rectangle on visualization
    cv2.rectangle(roi_display, (x_start, y_start), (x_end, y_end), color, 2)
    
    # Add a clear label with new format
    orient_label = "H" if orientation == "horizontal" else "V"
    label = f"G{group}E{element_num}{orient_label}"
    font_size = 0.5 if group >= 5 else 0.7
    cv2.putText(roi_display, label, (x_start, y_start-5), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
    
    # Return element information
    return {
        'group': group,
        'element': element_num,
        'orientation': orientation,
        'x': x_start,
        'y': y_start,
        'width': x_end - x_start,
        'height': y_end - y_start,
        'contrast': best_contrast,
        'profile': profile,
        'peak': peak,
        'trough': trough,
        'mtf_width': width,
        'mtf_value': mtf_value,
        'tangent_height': tangent_height,
        'region': element_region
    }

def analyze_usaf_target(img):
    """Analyze USAF target using group-by-group approach."""
    # Get ROI from user
    print("\nSelect region of interest containing the USAF target...")
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.title('Click two points to define the region of interest')
    points = plt.ginput(2, timeout=-1)
    plt.close()
    
    if len(points) != 2:
        return None
    
    # Define ROI
    x1, y1 = int(min(points[0][0], points[1][0])), int(min(points[0][1], points[1][1]))
    x2, y2 = int(max(points[0][0], points[1][0])), int(max(points[0][1], points[1][1]))
    roi = img[y1:y2, x1:x2]
    
    # Calculate contrast ratio first
    print("\nCalculating contrast ratio...")
    roi_display = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
    result = calculate_contrast_ratio(roi, roi_display)
    if result is not None:
        contrast_ratio = result
        print(f"Contrast Ratio: {contrast_ratio:.3f}")
    else:
        contrast_ratio = None
    
    # Process elements group by group
    all_elements = []
    best_element = None
    
    while True:
        # Ask which group to analyze
        try:
            group_input = input("\nEnter group number to analyze (or 'done' to finish): ")
            if group_input.lower() == 'done':
                break
            group = int(group_input)
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")
            continue
        
        print(f"\nProcessing Group {group}...")
        
        # Create a figure for this group
        fig = plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
        plt.title(f'Click on elements of Group {group} (press q to finish this group)')
        
        # Store elements for this group
        group_elements = []
        
        # Variables to track click pairs and current element number
        click_count = 0
        current_element = 1
        
        # Function to handle clicks
        def onclick(event):
            nonlocal best_element, click_count, current_element
            
            if event.xdata is None or event.ydata is None:
                return
            
            px, py = int(event.xdata), int(event.ydata)
            
            # Automatically assign element number based on click count
            # Every two clicks get the same element number
            element_num = current_element
            
            # Process the element
            element = process_element(roi, roi_display, group, element_num, px, py)
            
            if element:
                group_elements.append(element)
                all_elements.append(element)
                
                # Update the display
                plt.clf()
                plt.imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
                
                # Show contrast information
                contrast = element['contrast']
                target_contrast = 0.3
                contrast_diff = abs(contrast - target_contrast)
                
                # Update title with element info and contrast ratio
                orient_label = "H" if element['orientation'] == "horizontal" else "V"
                title = f'G{group}E{element_num}{orient_label}: Contrast = {contrast:.3f}'
                if contrast_ratio:
                    title += f' | Contrast Ratio: {contrast_ratio:.3f}'
                
                # Increment click count and update element number if needed
                click_count += 1
                if click_count % 2 == 0:
                    # After every 2 clicks, increment the element number
                    current_element += 1
                    title += f'\nMoving to element {current_element}. Click next element or press q to finish Group {group}'
                else:
                    title += f'\nClick on element {element_num} again or press q to finish Group {group}'
                
                plt.title(title)
                plt.draw()
                
                # Check if this is the best element so far
                if best_element is None or contrast_diff < abs(best_element['contrast'] - target_contrast):
                    best_element = element
                    print(f"New best element: G{group}E{element_num}{orient_label} (Contrast: {contrast:.3f})")
        
        # Function to handle key presses
        def onkey(event):
            if event.key == 'q':
                plt.close(fig)
        
        # Connect the events
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        
        print(f"Click on elements of Group {group}:")
        print("1. Click on element 1 (first click)")
        print("2. Click on element 1 again (second click)")
        print("3. Element number will automatically increment after every two clicks")
        print("The orientation (H/V) will be detected automatically")
        print("Press 'q' when done with this group")
        
        plt.show()
        
        print(f"Processed {len(group_elements)} elements in Group {group}")
    
    # Show the final result with all identified elements
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
    if contrast_ratio:
        plt.title(f'All Identified USAF Elements - Contrast Ratio: {contrast_ratio:.3f}')
    else:
        plt.title('All Identified USAF Elements')
    plt.show()
    
    if not all_elements:
        print("No elements were successfully processed.")
        return None
    
    # Find element with contrast closest to 30%
    target_contrast = 0.3
    closest_element = None
    min_contrast_diff = float('inf')
    
    for element in all_elements:
        contrast_diff = abs(element['contrast'] - target_contrast)
        
        if contrast_diff < min_contrast_diff:
            min_contrast_diff = contrast_diff
            closest_element = element
    
    # Print results for all elements
    print("\n=== ALL IDENTIFIED ELEMENTS ===")
    for element in sorted(all_elements, key=lambda e: (e['group'], e['orientation'], e['element'])):
        print(f"Group {element['group']}, Element {element['element']} ({element['orientation']}): "
              f"Contrast = {element['contrast']:.3f}")
    
    if result is not None:
        return closest_element, contrast_ratio, None
    else:
        return closest_element, None, None

def main():
    root = tk.Tk()
    root.withdraw()

    print("Select USAF target image...")
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return

    result = analyze_usaf_target(img)
    if result is None:
        print("Analysis failed or no suitable elements found.")
        return
    
    closest_element, contrast_ratio, bar_orientation = result
    
    group = closest_element['group']
    element = closest_element['element']
    orientation = closest_element['orientation']
    contrast = closest_element['contrast']
    profile = closest_element['profile']
    peak = closest_element['peak']
    trough = closest_element['trough']
    mtf_width = closest_element['mtf_width']
    mtf_value = closest_element['mtf_value']
    tangent_height = closest_element['tangent_height']
    region = closest_element['region']
    
    lp_mm = 2**(group + (element - 1)/6)
    lp_deg = lp_to_degrees(group, element)
    
    print("\n=== ELEMENT WITH CONTRAST CLOSEST TO 30% ===")
    print(f"Group {group}, Element {element} ({orientation})")
    print(f"Measured contrast: {contrast:.3f}")
    print(f"Spatial frequency: {lp_mm:.2f} lp/mm")
    print(f"Spatial frequency: {lp_deg:.2f} lp/degree")
    
    if mtf_width is not None:
        print(f"MTF Width: {mtf_width:.2f} pixels")
        print(f"MTF: {mtf_value:.2f} lp/deg")
    
    print(f"Contrast Ratio: {contrast_ratio:.3f}")
    
    # Plot pattern and intensity profile with analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.imshow(region, cmap='gray')
    plt.title(f'Pattern G{group}E{element} ({orientation})')
    
    # Plot intensity profile with MTF analysis in one combined graph
    plt.subplot(2, 1, 2)
    plt.plot(profile, 'b-', label='Intensity Profile')
    
    # Add peak and trough markers for contrast
    if peak is not None:
        plt.plot(peak, profile[peak], 'r^', label=f'Peak (value={profile[peak]:.1f})')
    if trough is not None:
        plt.plot(trough, profile[trough], 'gv', label=f'Trough (value={profile[trough]:.1f})')
    
    # Mark the tangent line for MTF calculation
    if mtf_width is not None and peak is not None and tangent_height is not None:
        plt.axhline(y=tangent_height, color='r', linestyle='--', label=f'Tangent Line ({tangent_height:.1f})')
        plt.axhline(y=profile[peak], color='g', linestyle=':', label=f'Peak ({profile[peak]:.1f})')
        
        # Estimate MTF width region
        x = np.arange(len(profile))
        interp_func = interp1d(x, profile, kind='cubic', bounds_error=False, fill_value=0)
        x_high_res = np.linspace(0, len(profile)-1, num=1000)
        y_high_res = interp_func(x_high_res)
        above_tangent = y_high_res > tangent_height
        crossings = np.where(np.diff(above_tangent))[0]
        
        if len(crossings) >= 2:
            left_idx = x_high_res[crossings[0]]
            right_idx = x_high_res[crossings[-1]]
            plt.axvspan(left_idx, right_idx, alpha=0.3, color='yellow', label=f'Width = {mtf_width:.2f} pixels')
    
    title = f'Intensity Profile\nContrast = {contrast:.3f}'
    if mtf_value is not None:
        title += f'\nMTF = {mtf_value:.2f} lp/deg'
    plt.title(title)
    
    plt.xlabel('Position (pixels)')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()