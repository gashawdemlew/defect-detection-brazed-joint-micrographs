import os
import cv2
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Configuration ---

# 1. MODEL AND PATHS CONFIGURATION
# Path to your trained YOLO model's weights file.
MODEL_PATH = 'model/defect_detector_model.pt'

# Input folder containing micrographs.
IMAGE_FOLDER_PATH = 'unprocessed_images/'

# Output folder for processed images and reports.
OUTPUT_FOLDER_PATH = 'processed_images/'

# CSV file to store the aggregate defect summary.
SUMMARY_CSV_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'defect_summary.csv')

# --- SCALE CONVERSION ---
# IMPORTANT: You must calculate this value from the 1000 µm scale bar in your images.
# Example: If the 1000 µm bar is 500 pixels long, PIXELS_PER_MICRON = 500 / 1000 = 0.5
PIXELS_PER_MICRON = 0.5 # <--- REPLACE WITH YOUR VALUE

# --- IMAGE PROCESSING PARAMETERS (Configurable) ---
# Parameters for Hough Circle Transform to find the pipes.
# NOTE: These parameters might need adjustment if your original images vary significantly in size
# or if resizing affects circle detection.
HOUGH_DP = 1.5          # Inverse ratio of accumulator resolution.
HOUGH_MIN_DIST = 400    # Minimum distance between the centers of detected circles.
HOUGH_PARAM1 = 100      # Upper threshold for the internal Canny edge detector.
HOUGH_PARAM2 = 80       # Threshold for center detection.
HOUGH_MIN_RADIUS = 200  # Minimum circle radius to be detected.
HOUGH_MAX_RADIUS = 600  # Maximum circle radius to be detected.

# --- PREPROCESSING CONFIGURATION ---
# Target dimensions for image preprocessing before feeding to the model.
TARGET_IMAGE_DIM = (640, 640) # Width, Height

# --- MODEL PREDICTION CONFIDENCE THRESHOLD ---
# Only predictions with a confidence score above this threshold will be considered.
CONFIDENCE_THRESHOLD = 0.25 # 25% confidence level

def analyze_defects(model, image_path, pixels_per_micron):
    """
    Processes a single image to find defects, calculate metrics, and return structured data.
    This version includes image preprocessing (resizing to TARGET_IMAGE_DIM) before
    model prediction and adjusts metric calculations to account for scaling.
    Applies a confidence threshold to filter model predictions.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return [], None

    # Store original dimensions for later use (to scale metrics back)
    original_height, original_width = img_bgr.shape[:2]
    image_id = os.path.basename(image_path)

    # --- Preprocess image for model prediction ---
    # Resize the image to the target dimensions (640x640)
    # Using INTER_AREA for downsampling, INTER_LINEAR or INTER_CUBIC for upsampling
    resized_img_bgr = cv2.resize(img_bgr, TARGET_IMAGE_DIM, interpolation=cv2.INTER_AREA)

    # Convert the resized image to grayscale for Hough Circle Transform
    resized_img_gray = cv2.cvtColor(resized_img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Step 4: Nominal joint detection (using resized_img_gray) ---
    # Apply a median blur to reduce noise for circle detection.
    blurred_gray = cv2.medianBlur(resized_img_gray, 5)
    circles = cv2.HoughCircles(
        blurred_gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS
    )

    # Store circle data. Circles are detected in the resized image's coordinate system.
    inner_circle, outer_circle = None, None
    if circles is not None and len(circles[0]) >= 2:
        # Sort circles by radius to identify inner and outer
        sorted_circles = sorted(circles[0, :], key=lambda x: x[2])
        inner_circle = sorted_circles[0]
        outer_circle = sorted_circles[1]
    
    # Run YOLO inference on the PREPROCESSED (resized) image (NumPy array)
    results = model(resized_img_bgr, verbose=False)
    single_result = results[0]
    
    # Get the overlay image with color-coded contours from the model.
    # This overlay image will be generated based on the 640x640 dimensions.
    overlay_image = single_result.plot()

    defect_data = []

    # Check if there are any detected masks or boxes
    if single_result.masks is None or single_result.boxes is None:
        return [], overlay_image

    # --- Defect Labelling and Metric Calculation ---
    # Iterate through detections, filtering by confidence threshold
    for i in range(len(single_result.masks)):
        mask = single_result.masks[i]
        box = single_result.boxes[i]
        
        # Apply confidence threshold
        if box.conf is not None and box.conf.item() < CONFIDENCE_THRESHOLD:
            continue # Skip this detection if confidence is too low

        # Get YOLO's classification
        yolo_class_name = single_result.names[int(box.cls)]
        
        # --- Calculate Metrics ---
        # The mask data is already in the dimensions of the preprocessed image (640x640)
        mask_data = mask.data[0].cpu().numpy().astype(np.uint8)

        # Calculate Area (on the 640x640 mask)
        area_pixels_resized = np.sum(mask_data)
        
        # Scale factor for area: (original_width/target_width) * (original_height/target_height)
        width_scale_factor = float(original_width) / TARGET_IMAGE_DIM[0]
        height_scale_factor = float(original_height) / TARGET_IMAGE_DIM[1]
        
        # Area in original image pixel scale
        area_pixels_original_scale = area_pixels_resized * width_scale_factor * height_scale_factor
        area_microns_sq = area_pixels_original_scale / (pixels_per_micron ** 2)

        # Calculate Aspect Ratio (AR) from fitted ellipse. AR is a ratio, so it's scale-invariant.
        contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aspect_ratio = 1.0 # Default AR for small/non-fittable contours
        if contours and len(contours[0]) >= 5: # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(contours[0])
            (w, h) = ellipse[1]
            aspect_ratio = max(w, h) / min(w,h) if min(w,h) > 0 else 1.0

        # Centroid and Distance from circles
        # Centroid is in the resized image's coordinate system
        dist_from_inner_microns = -1.0
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cX_resized = int(M["m10"] / M["m00"])
                cY_resized = int(M["m01"] / M["m00"])
                centroid_resized = (cX_resized, cY_resized)

                # Calculate distance to inner circle if found (both are in resized scale)
                if inner_circle is not None:
                    circle_center_resized = (int(inner_circle[0]), int(inner_circle[1]))
                    # Distance is from centroid to the circle's perimeter in resized pixels
                    dist_pixels_resized = abs(np.linalg.norm(np.array(centroid_resized) - np.array(circle_center_resized)) - inner_circle[2])
                    
                    # Scale this distance back to original image pixel scale
                    # Use an average scale factor for distance to account for potential non-uniform scaling
                    avg_scale_factor = (width_scale_factor + height_scale_factor) / 2
                    dist_pixels_original_scale = dist_pixels_resized * avg_scale_factor

                    dist_from_inner_microns = dist_pixels_original_scale / pixels_per_micron

        # --- CSV row for every defect ---
        defect_data.append({
            "Image ID": image_id,
            "Defect Area (µm²)": round(area_microns_sq, 2),
            "Aspect Ratio (AR)": round(aspect_ratio, 2),
            "Distance from Inner Circle (µm)": round(dist_from_inner_microns, 2),
            "Defect Classification": yolo_class_name,
            "Confidence": round(box.conf.item(), 4) # Add confidence to the output
        })
            
    return defect_data, overlay_image


def update_summary_csv(new_defects_df):
    """
    Appends new defect data to the main summary CSV and recalculates the header.
    """
    # Load existing data if the file exists, otherwise start with an empty DataFrame
    if os.path.exists(SUMMARY_CSV_PATH) and os.path.getsize(SUMMARY_CSV_PATH) > 0:
        try:
            # Read the existing CSV, skipping lines that start with '#' (our header)
            existing_df = pd.read_csv(SUMMARY_CSV_PATH, comment='#', encoding='utf-8')
            all_data_df = pd.concat([existing_df, new_defects_df], ignore_index=True)
            # Remove duplicates based on 'Image ID' and 'Defect Area (µm²)' if re-processing the same image multiple times
            all_data_df.drop_duplicates(inplace=True) 
        except pd.errors.EmptyDataError:
            # Handle case where file exists but is empty after comments
            all_data_df = new_defects_df
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError when reading {SUMMARY_CSV_PATH}: {e}")
            print("Attempting to read with 'latin1' encoding as a fallback.")
            try:
                existing_df = pd.read_csv(SUMMARY_CSV_PATH, comment='#', encoding='latin1')
                all_data_df = pd.concat([existing_df, new_defects_df], ignore_index=True)
                all_data_df.drop_duplicates(inplace=True)
            except Exception as fallback_e:
                print(f"Fallback to 'latin1' also failed: {fallback_e}")
                print("Proceeding with only newly processed data for the summary report.")
                all_data_df = new_defects_df
    else:
        all_data_df = new_defects_df

    if all_data_df.empty:
        print("No defect data to write to summary CSV.")
        return

    # Compute totals per defect classification from the combined data
    summary_stats = all_data_df.groupby('Defect Classification').agg(
        Count=('Image ID', 'size'),
        Total_Area_um2=('Defect Area (µm²)', 'sum')
    ).reset_index()
    
    summary_stats.rename(columns={'Count': 'Total Count', 'Total_Area_um2': 'Total Area (µm²)'}, inplace=True)

    # Prepare file header with summary data 
    report_header = "# Defect Summary Report\n"
    report_header += f"# Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_header += f"# Minimum Prediction Confidence Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%\n" # Add threshold
    report_header += "# Total Defects Found Across All Processed Images: {}\n".format(len(all_data_df))
    report_header += "# --- Totals per Classification (All Images) ---\n"
    for _, row in summary_stats.iterrows():
        report_header += "# {}: Count = {}, Total Area = {:.2f} µm²\n".format(
            row['Defect Classification'], row['Total Count'], row['Total Area (µm²)']
        )
    report_header += "#\n"

    # Write the header, then append the full data
    with open(SUMMARY_CSV_PATH, 'w', encoding='utf-8') as f: # Explicitly set encoding
        f.write(report_header)
    
    all_data_df.to_csv(SUMMARY_CSV_PATH, mode='a', index=False, encoding='utf-8') # Explicitly set encoding
    
    print(f"Aggregate report updated and saved to: {SUMMARY_CSV_PATH}")
    print("\nSummary Statistics (Current):")
    print(summary_stats.to_string(index=False))


def main():
    """
    Main function to run the defect analysis pipeline.
    """
    # Create output directory if it doesn't exist 
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    # Load the trained YOLO model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)

    # Find all images in the input folder 
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER_PATH, '*.png')) + \
                  glob.glob(os.path.join(IMAGE_FOLDER_PATH, '*.jpg'))

    if not image_paths:
        print(f"ERROR: No images found in {IMAGE_FOLDER_PATH}")
        return

    newly_processed_defects_data = []

    print("\n--- Starting Image Preprocessing ---")
    for image_path in image_paths:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_image_path = os.path.join(OUTPUT_FOLDER_PATH, f"processed_{os.path.basename(image_path)}")
        individual_csv_path = os.path.join(OUTPUT_FOLDER_PATH, f"{base_filename}_defects.csv")

        # Skip already processed images
        if os.path.exists(output_image_path) and os.path.exists(individual_csv_path):
            print(f"Skipping {os.path.basename(image_path)}: Already processed.")
            continue

        print(f"Processing {os.path.basename(image_path)}...")
        
        # Analyze the image to get defect data and the overlay image
        defects, overlay = analyze_defects(model, image_path, PIXELS_PER_MICRON)
        
        if defects:
            # Add this image's data to the list for the aggregate report
            newly_processed_defects_data.extend(defects)
            
            # Generate Output per image in an independent .csv
            individual_df = pd.DataFrame(defects)
            
            # Prepare header for individual CSV (optional but good for clarity)
            individual_header = f"# Defect Report for Image: {os.path.basename(image_path)}\n"
            individual_header += f"# Total Defects Found in this Image: {len(individual_df)}\n"
            individual_header += f"# Minimum Prediction Confidence Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%\n"
            individual_header += "#\n"

            with open(individual_csv_path, 'w', encoding='utf-8') as f: # Explicitly set encoding
                f.write(individual_header)
            individual_df.to_csv(individual_csv_path, mode='a', index=False, encoding='utf-8') # Explicitly set encoding
            print(f"  -> Individual defect report saved to: {individual_csv_path}")

        # Save the overlay image to the processed_images/ folder 
        # The overlay image is already generated at the 640x640 scale, so save it directly.
        cv2.imwrite(output_image_path, overlay)
        print(f"  -> Processed image saved to: {output_image_path}")
    
    if not newly_processed_defects_data:
        print("No new defects were detected or all images were already processed.")
    else:
        # Update the main summary CSV with newly processed data
        new_df = pd.DataFrame(newly_processed_defects_data)
        update_summary_csv(new_df)
    
    print("\n--- Processing Complete ---")


if __name__ == '__main__':
    main()