Defect Detection and Analysis
This project provides a Python script to detect and analyze defects in micrographs using a trained YOLOv8 model. It processes images, calculates defect metrics, generates individual and aggregate reports, and provides tools for containerization.

Features
YOLOv11 Integration: Uses a pre-trained YOLOv11 model for defect classification and segmentation.

Defect Metrics: Calculates defect area (in µm²) and aspect ratio.

Automated Processing: Batch processes images from a specified input folder.

Skipping Processed Images: Automatically skips images that have already been processed, preventing redundant work.

Incremental Reporting: Updates defect_summary.csv incrementally, appending new data and recalculating overall statistics.

Individual Reports: Generates a CSV report for each processed image detailing detected defects.

Visual Output: Saves overlay images with defect contours and classifications.

Containerization Ready: Includes Dockerfile and requirements.txt for easy deployment using Docker.

Setup
Prerequisites
Python 3.12.11+

pip (Python package installer)

Docker (optional, for containerized deployment)

Installation
Clone the repository (or download the files):

git clone <repository_url>
cd defect-detection-analysis

Install dependencies:

pip install -r requirements.txt

Place your trained YOLOv8 model weights:

Ensure your trained YOLOv8 model file (.pt extension) is named defect_detector_model.pt and placed in the same directory as the main.py script. If your model has a different name or path, update the MODEL_PATH variable in main.py.

.
├── defect_analyzer.py
├── defect_detector_model.pt  <-- Your trained model
├── unprocessed_images/       <-- Place your input images here
└── requirements.txt
└── Dockerfile

Prepare your input images:

Create a folder named unprocessed_images/ in the same directory as main.py and place your micrograph images (.png, .jpg) inside it.

Configure PIXELS_PER_MICRON:

Open main.py and carefully set the PIXELS_PER_MICRON variable. This value is crucial for accurate area calculations. It should be derived from a known scale bar in your images (e.g., if a 1000 µm scale bar is 500 pixels long in your image, PIXELS_PER_MICRON = 500 / 1000 = 0.5).

# main.py
PIXELS_PER_MICRON = 0.5 # <--- REPLACE WITH YOUR ACTUAL VALUE

Model Training
The YOLOv8 model used in this project was trained with the following steps:

Image Annotation: Images were meticulously annotated using the Roboflow tool. Two primary defect classes were defined: porosity and incomplete_penetration.

Model Training: A YOLOv8 model (specifically, using a version 11 training configuration, though the script expects a standard YOLOv8 model) was trained on the annotated dataset.

Model Evaluation: Post-training, the model was evaluated, achieving a Mean Average Precision (mAP) of 78.2%. This metric indicates the model's overall detection and classification performance.

Usage
To run the defect detection and analysis:

python main.py

The script will:

Read images from unprocessed_images/.

Process new images, skipping those already processed.

Generate individual defect reports (<image_id>_defects.csv) in processed_images/.

Save processed overlay images (processed_<image_id>.<ext>) in processed_images/.

Update the main aggregate report (defect_summary.csv) in processed_images/.

Output
The processed_images/ directory will contain:

processed_<image_id>.<ext>: Original images with detected defects highlighted and classified.

<image_id>_defects.csv: A CSV file for each image detailing the defects found, their area, and aspect ratio.

defect_summary.csv: An aggregate CSV file containing all detected defects from all processed images, including a header with overall summary statistics.

Docker Deployment (Optional)
You can build and run this application using Docker for a consistent environment.

Build the Docker image:

Navigate to the directory containing Dockerfile, main.py, requirements.txt, and defect_detector_model.pt.

docker build -t defect-detector .

Run the Docker container:

You need to mount your unprocessed_images and processed_images directories as volumes to allow the container to access your data and save outputs.

docker run -it --rm \
    -v "$(pwd)/unprocessed_images:/app/unprocessed_images" \
    -v "$(pwd)/processed_images:/app/processed_images" \
    defect-detector

--rm: Automatically remove the container when it exits.

-v: Mounts a host directory into the container. Replace $(pwd) with the absolute path to your project directory if you are not in it.

Customization
New Metrics: You can extend the analyze_defects function to calculate additional defect metrics relevant to your analysis.

Reporting: Modify the update_summary_csv function to change the format or content of the aggregate report.

Troubleshooting
MODEL_PATH not found: Ensure defect_detector_model.pt is in the correct location or update the MODEL_PATH variable in main.py.

No images found: Check if unprocessed_images/ exists and contains supported image files (.png, .jpg).

OpenCV errors (Docker): The Dockerfile includes system dependencies for OpenCV. If you encounter issues, ensure your base image supports them or install additional libraries.

CUDA/GPU issues: If you are using a GPU for YOLO inference, ensure your environment (and Docker setup) has the necessary CUDA drivers and libraries installed.