Bangla License Plate Character Detector (YOLOv8)
This repository provides a deep learning solution for detecting individual Bangla characters on license plates using the YOLOv8 object detection framework. The goal is to accurately localize and classify each character, forming a crucial step in a full License Plate Recognition (LPR) system for Bangladeshi vehicles.

Table of Contents
Features
Getting Started
Prerequisites
Installation
Dataset
Training
Prediction
Project Structure
Results and Artifacts
License
Contact
Features
YOLOv8-based Detection: Utilizes the state-of-the-art YOLOv8 architecture for robust and efficient character detection.
Bangla Character Focus: Specifically trained to recognize the unique script of Bangla characters.
Custom Training Script: main.py provides a customizable training pipeline.
Inference Script: predict.py for easily running predictions on new images.
Comprehensive Training Outputs: Generates various plots and metrics (F1 curves, confusion matrix, batch predictions) to monitor training progress and model performance.
Pre-trained Weights: Includes best.pt for immediate inference.
Getting Started
Follow these instructions to set up and run the project.

Prerequisites
Python 3.8+
NVIDIA GPU (recommended for training) with CUDA support
Installation
Clone the repository:

Bash

git clone https://github.com/samin2703/Bangla_license_plate_project
cd your-repo-name


Install ultralytics:
The core dependency is ultralytics, which includes YOLOv8.

Bash

pip install ultralytics
(For GPU support, ensure you have the correct CUDA toolkit installed for your PyTorch version.)

Dataset
This model is trained on a custom dataset of Bangla license plates. The dataset configuration is defined in config.yaml. It is expected to be in YOLO format, with bounding box annotations for each Bangla character.

config.yaml example structure:

YAML

# Train/val/test sets as 1-3 paths, e.g. /usr/src/datasets/coco128...
path: /path/to/your/bangla_lp_dataset # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

# Classes
nc: 68  # number of classes (0-9, A-Z, ঁ ং ঃ অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল শ ষ স হ ড় ঢ় য়)
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়']
Note: The actual nc and names list should reflect all characters you are trying to detect.

Training
The main.py script is configured to train a YOLOv8-based model for Bangla character detection.

To start training, simply run:

Bash

python main.py
Training Parameters (from main.py):

model: yolo11m.pt (initial weights - likely a custom or slightly modified YOLOv8m checkpoint)
data: /media/rmedu-4090/New Volume1/samin-workforceRTX/license_plate_detector/config.yaml (path to your dataset configuration)
epochs: 500
plots: True (generates various training plots)
imgsz: 640 (input image size)
batch: 16 (batch size)
fliplr: 0.0 (no horizontal flipping during augmentation)
mosaic: 0.4 (mosaic augmentation probability)
name: bangla-char-detector (name for the training run, results will be saved in runs/detect/bangla-char-detector/)
workers: 0 (number of data loading workers - set to 0 to avoid multiprocessing issues, especially on Windows)
Prediction
Use the predict.py script or the yolo CLI command to run inference on new images or videos.

Using predict.py (assuming it uses the YOLO object):

Bash

python predict.py --source /path/to/your/image.jpg --weights best.pt --conf 0.25 --save-txt --save-conf
(You might need to adjust predict.py to match your specific inference needs, e.g., how it handles the model loading and prediction.)

Using the yolo CLI (recommended for simplicity):

To detect characters on a single image:

Bash

yolo predict model=best.pt source=test_image.jpg conf=0.25
To detect characters on all images in a folder:

Bash

yolo predict model=best.pt source=/path/to/image/folder conf=0.25
Results (images with bounding boxes and labels) will be saved in runs/detect/predict/ (or runs/detect/predictX/ if multiple prediction runs).

Project Structure
Here's an overview of the key files and directories:

text

.
├── main.py                    # Main script for training the YOLOv8 model
├── predict.py                 # Script for running inference (prediction)
├── config.yaml                # Dataset configuration (paths, class names)
├── args.yaml                  # Auto-generated YAML containing training arguments
├── best.pt                    # Pre-trained model weights (best performing checkpoint)
├── test_image.jpg             # Example image for testing predictions
├── labels.jpg                 # Example image showing ground truth labels (if present)
├── README.md                  # This README file
└── runs/                      # Directory where training results are saved
    └── detect/
        └── bangla-char-detector/ # Training run specific folder
            ├── weights/           # Saved model weights (best.pt, last.pt)
            ├── args.yaml          # Training arguments for this run
            ├── results.csv        # CSV file with epoch-wise metrics
            ├── results.png        # Plot of training metrics
            ├── BoxF1_curve.png    # F1-score curve
            ├── BoxPR_curve.png    # Precision-Recall curve
            ├── BoxP_curve.png     # Precision curve
            ├── BoxR_curve.png     # Recall curve
            ├── confusion_matrix.png         # Confusion matrix
            ├── confusion_matrix_normalized.png # Normalized confusion matrix
            ├── train_batch0.jpg
            ├── train_batch1.jpg
            ├── train_batch2.jpg   # Example training batches with labels
            ├── val_batch0_labels.jpg
            ├── val_batch0_pred.jpg
            ├── val_batch1_labels.jpg
            ├── val_batch1_pred.jpg
            ├── val_batch2_labels.jpg
            └── val_batch2_pred.jpg # Example validation batches (ground truth and predictions)
Results and Artifacts
After training, the runs/detect/bangla-char-detector/ directory will contain:

best.pt: The weights of the model with the highest validation performance.
results.csv: A CSV file logging various metrics (loss, precision, recall, mAP) for each epoch.
results.png: A plot visualizing the training and validation metrics over epochs.
confusion_matrix.png: A visualization of the model's classification performance.
BoxF1_curve.png, BoxP_curve.png, BoxR_curve.png, BoxPR_curve.png: Plots showing object detection specific metrics.
train_batch*.jpg and val_batch*.jpg: Images showing example batches from the training and validation sets, including ground truth labels and model predictions, allowing for visual inspection of the model's learning process.
License
This project is open-sourced under the MIT License. (If you have a different license, update this section and include a LICENSE file)

Contact
For any questions or suggestions, feel free to open an issue or contact [Your Name/Email].
