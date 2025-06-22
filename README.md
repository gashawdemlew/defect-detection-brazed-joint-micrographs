✨ Defect Detection and Analysis for Brazed Joint Micrographs

This project provides a Python-based tool that uses a trained YOLOv11 model to detect and analyze defects in brazed joint micrographs. It supports batch processing, computes defect metrics, generates visual and CSV reports, and offers containerized deployment via Docker.

🚀 Features

🔍 YOLOv11 Integration
Leverages a pre-trained YOLOv11 model for accurate defect classification and segmentation.

📐 Defect Metrics
Calculates defect area (in µm²) and aspect ratio from micrographs.

⚙️ Batch Processing
Automatically processes all images from the unprocessed_images/ directory.

⏩ Smart Skipping
Skips previously processed images to avoid redundancy.

📊 Incremental Summary Reporting
Appends new results to defect_summary.csv and updates overall statistics.

🧾 Individual Image Reports
Creates a per-image CSV with detailed defect information.

🖼️ Visual Overlays
Saves output images with overlaid defect contours and class labels.

🐳 Docker Support
Includes Dockerfile and dependencies for streamlined deployment.

🛠️ Setup Guide

✅ Prerequisites

Python 3.12.11+

pip (Python package installer)

Docker (optional, for containerized use)

📦 Installation

Clone the repository:

bash
Copy
Edit
git clone https://github.com/gashawdemlew/defect-detection-brazed-joint-micrographs.git
cd defect-detection-analysis
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Prepare your YOLOv11 model weights:

Rename your trained .pt model file to: defect_detector_model.pt

Place it in the same directory as defect_analyzer.py

Project Structure Overview:

sql
Copy
Edit
.
├── defect_analyzer.py
├── model_training/
│   ├── Defect-detection using segmentation_approch.ipynb
│   └── defect_detector_model.pt
├── unprocessed_images/      <-- Input images go here
├── processed_images/        <-- Processed outputs saved here
├── requirements.txt
└── Dockerfile
🔧 Configuration
📷 Input Images
Place your .png or .jpg micrograph images in the unprocessed_images/ directory.

📏 Set Scale for Area Calculations
Open defect_analyzer.py and set PIXELS_PER_MICRON based on your image’s scale bar.

python
Copy
Edit
# Example:
PIXELS_PER_MICRON = 0.5  # Replace with your actual value
Tip: If a 1000 µm scale bar equals 500 pixels in your image, then:
PIXELS_PER_MICRON = 500 / 1000 = 0.5

🧠 Model Training Overview
The YOLOv11 model was developed as follows:

Annotation
Images were annotated using Roboflow, defining two classes:

porosity

incomplete_penetration

Training
Trained using YOLOv11 with segmentation-capable configurations.

Evaluation
Achieved a Mean Average Precision (mAP) of 78.2%, indicating strong performance.

💻 How to Run
To start the analysis:

bash
Copy
Edit
python defect_analyzer.py
What happens:

Reads all new images from unprocessed_images/

Detects and classifies defects

Saves:

Visual overlays (processed_<image_id>.<ext>)

Per-image reports (<image_id>_defects.csv)

Aggregate summary (defect_summary.csv)

Moves all outputs to processed_images/

📁 Output Overview
All results are saved in the processed_images/ directory:

File	Description
processed_<image_id>.jpg/png	Visual overlay with defect contours and labels
<image_id>_defects.csv	CSV listing defects, areas, and aspect ratios
defect_summary.csv	Aggregate CSV of all processed images

🐳 Docker Deployment (Optional)
You can deploy using Docker for a consistent environment.

🏗️ Build the Docker Image
From the project root:

bash
Copy
Edit
docker build -t defect-detector .
▶️ Run the Container
bash
Copy
Edit
docker run -it --rm \
  -v "$(pwd)/unprocessed_images:/app/unprocessed_images" \
  -v "$(pwd)/processed_images:/app/processed_images" \
  defect-detector
Flags explanation:

--rm: Deletes the container after execution

-v: Mounts your host folders inside the container

✅ Make sure the model .pt file and all required scripts are inside the container path.

⚙️ Customization Options
Extend Metrics:
Modify analyze_defects() in defect_analyzer.py to compute new metrics.

Change Report Format:
Customize update_summary_csv() for alternative summary structures.

🧩 Troubleshooting
Issue	Solution
MODEL_PATH error	Check if defect_detector_model.pt is in the correct directory.
No images found	Ensure unprocessed_images/ exists and contains .png/.jpg files.
OpenCV errors (Docker)	Confirm all dependencies are installed in your Docker base image.
CUDA/GPU issues	Verify GPU drivers and CUDA support are correctly configured in your environment.

📬 Feedback or Contributions
Feel free to open issues or contribute pull requests to improve this tool. Contributions are welcome!

