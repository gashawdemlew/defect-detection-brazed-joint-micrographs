import os
import cv2
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Configuration ---

# 1. MODEL AND PATHS CONFIGURATION
# Path to your trained YOLO model's weights file.
MODEL_PATH = 'defect_detector_model.pt'

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


def analyze_defects(model, image_path, pixels_per_micron):
    """
    Processes a single image to find defects, calculate metrics, and return structured data.
    This version uses ONLY the YOLO model for classification.
    """
    image_id = os.path.basename(image_path)

    try:
        # Run YOLO inference
        results = model(image_path, verbose=False)
        single_result = results[0]
        
        # Get the overlay image with color-coded contours from the model.
        overlay_image = single_result.plot()

        defect_data = []

        # Check if there are any detected masks
        if single_result.masks is None:
            return [], overlay_image

        # --- Defect Labelling and Metric Calculation ---
        for i in range(len(single_result.masks)):
            mask = single_result.masks[i]
            box = single_result.boxes[i]
            
            # Get YOLO's classification
            yolo_class_name = single_result.names[int(box.cls)]
            
            # --- Calculate Metrics ---
            mask_data = mask.data[0].cpu().numpy().astype(np.uint8)

            # Calculate Area 
            area_pixels = np.sum(mask_data)
            area_microns_sq = area_pixels / (pixels_per_micron ** 2)

            # Calculate Aspect Ratio (AR) from fitted ellipse 
            contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            aspect_ratio = 1.0 # Default AR for small/non-fittable contours
            if contours and len(contours[0]) >= 5: # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(contours[0])
                (w, h) = ellipse[1]
                aspect_ratio = max(w, h) / min(w, h) if min(w,h) > 0 else 1.0

            # --- CSV row for every defect  ---
            # This dictionary now contains the simplified data.
            defect_data.append({
                "Image ID": image_id,
                "Defect Area (µm²)": round(area_microns_sq, 2),
                "Aspect Ratio (AR)": round(aspect_ratio, 2),
                "Defect Classification": yolo_class_name,
            })
            
        return defect_data, overlay_image
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        # Return empty data and a dummy image if an error occurs
        # A black image or original image can be returned if desired, for now, a blank numpy array
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8) 
        return [], dummy_image

def update_summary_csv(new_defects_df):
    """
    Appends new defect data to the main summary CSV and recalculates the header.
    """
    # Load existing data if the file exists, otherwise start with an empty DataFrame
    if os.path.exists(SUMMARY_CSV_PATH) and os.path.getsize(SUMMARY_CSV_PATH) > 0:
        # Read the existing CSV, skipping lines that start with '#' (our header)
        existing_df = pd.read_csv(SUMMARY_CSV_PATH, comment='#')
        all_data_df = pd.concat([existing_df, new_defects_df], ignore_index=True)
        # Remove duplicates based on 'Image ID' and 'Defect Area (µm²)' if re-processing the same image multiple times
        all_data_df.drop_duplicates(inplace=True) 
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
    report_header += "# Total Defects Found Across All Processed Images: {}\n".format(len(all_data_df))
    report_header += "# --- Totals per Classification (All Images) ---\n"
    for _, row in summary_stats.iterrows():
        report_header += "# {}: Count = {}, Total Area = {:.2f} µm²\n".format(
            row['Defect Classification'], row['Total Count'], row['Total Area (µm²)']
        )
    report_header += "#\n"

    # Write the header, then append the full data
    with open(SUMMARY_CSV_PATH, 'w') as f:
        f.write(report_header)
    
    all_data_df.to_csv(SUMMARY_CSV_PATH, mode='a', index=False)
    
    print(f"Aggregate report updated and saved to: {SUMMARY_CSV_PATH}")
    print("\nSummary Statistics (Current):")
    print(summary_stats.to_string(index=False))


def main():
    """
    Main function to run the simplified defect analysis pipeline with skipping and incremental updates.
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
            individual_df.to_csv(individual_csv_path, index=False)
            print(f"  -> Individual defect report saved to: {individual_csv_path}")

        # Save the overlay image to the processed_images/ folder 
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

