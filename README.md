```markdown
# 🇧🇩 Bangla License Plate Character Detector (YOLOv8)

This repository provides a deep learning solution for detecting individual Bangla characters on license plates using the YOLOv8 object detection framework. The goal is to accurately localize and classify each character, forming a crucial step in a full License Plate Recognition (LPR) system for Bangladeshi vehicles.

---

## 📷 Demo / Example Image

![Demo Image](assets/demo.png)  
*Replace this with your own image showing detection results.*

---

## 📑 Table of Contents

- [Features](#-features)  
- [Getting Started](#-getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#-installation)  
- [Dataset](#-dataset)  
- [Training](#-training)  
- [Prediction](#-prediction)  
- [Project Structure](#-project-structure)  
- [Results and Artifacts](#-results-and-artifacts)  
- [License](#-license)  
- [Contact](#-contact)  

---

## ✨ Features

- **YOLOv8-based Detection:** Utilizes the state-of-the-art YOLOv8 architecture for robust and efficient character detection.  
- **Bangla Character Focus:** Specifically trained to recognize the unique script of Bangla characters.  
- **Custom Training Script:** `main.py` provides a customizable training pipeline.  
- **Inference Script:** `predict.py` for easily running predictions on new images.  
- **Comprehensive Training Outputs:** Generates plots and metrics (F1 curves, confusion matrix, batch predictions) to monitor training progress.  
- **Pre-trained Weights:** Includes `best.pt` for immediate inference.  

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- NVIDIA GPU (recommended for training) with CUDA support  

---

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/samin2703/Bangla_license_plate_project.git
   cd Bangla_license_plate_project
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics
   ```
   - For GPU support, ensure you have the correct CUDA-enabled PyTorch installed.

---

## 🗂 Dataset

The model is trained on a custom dataset of Bangla license plates in YOLO format. The dataset configuration is defined in `config.yaml`.

Example `config.yaml`:
```yaml
path: /path/to/your/bangla_lp_dataset
train: images/train
val: images/val

# Update nc and names according to the characters you want to detect.
# The example below lists 85 classes.
nc: 85
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'ঁ', 'ং', 'ঃ',
        'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
        'ক', 'খ', 'গ', 'ঘ', 'ঙ',
        'চ', 'ছ', 'জ', 'ঝ', 'ঞ',
        'ট', 'ঠ', 'ড', 'ঢ', 'ণ',
        'ত', 'থ', 'দ', 'ধ', 'ন',
        'প', 'ফ', 'ব', 'ভ', 'ম',
        'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ',
        'ড়', 'ঢ়', 'য়']
```

---

## 🏋️ Training

The `main.py` script is configured to train a YOLO-based model.

Run:
```bash
python main.py
```

Training Parameters (from `main.py`):
- model: `yolo11m.pt` (initial weights)
- data: Path to your `config.yaml`
- epochs: 500
- plots: True (generate training plots)
- imgsz: 640
- batch: 16
- fliplr: 0.0
- mosaic: 0.4
- name: `bangla-char-detector` (results saved in `runs/detect/bangla-char-detector/`)
- workers: 0 (set to 0 to avoid multiprocessing issues)

---

## 🔮 Prediction

Run inference using `predict.py` or the YOLO CLI.

Using `predict.py`:
```bash
python predict.py --source /path/to/image.jpg --weights best.pt --conf 0.25 --save-txt --save-conf
```

Using YOLO CLI:

- Single image:
  ```bash
  yolo predict model=best.pt source=test_image.jpg conf=0.25
  ```

- Folder of images:
  ```bash
  yolo predict model=best.pt source=/path/to/image/folder conf=0.25
  ```

Results are saved in `runs/detect/predict/`.

---

## 🗃 Project Structure

```text
.
├── main.py                    # Training script
├── predict.py                 # Inference script
├── config.yaml                # Dataset configuration
├── args.yaml                  # Training arguments
├── best.pt                    # Pre-trained model weights
├── test_image.jpg             # Example image for testing
├── labels.jpg                 # Example labeled image
├── README.md                  # This file
└── runs/
    └── detect/
        └── bangla-char-detector/
            ├── weights/               # Model weights
            ├── args.yaml              # Training arguments
            ├── results.csv            # Epoch-wise metrics
            ├── results.png            # Metric plots
            ├── BoxF1_curve.png
            ├── BoxPR_curve.png
            ├── BoxP_curve.png
            ├── BoxR_curve.png
            ├── confusion_matrix.png
            ├── confusion_matrix_normalized.png
            ├── train_batch*.jpg       # Example training batches
            └── val_batch*_*.jpg       # Example validation batches
```

---

## 📊 Results and Artifacts

After training, `runs/detect/bangla-char-detector/` contains:

- `best.pt`: Best model weights  
- `results.csv`: Training metrics per epoch  
- `results.png`: Training metrics plot  
- `confusion_matrix.png`: Classification performance  
- `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxPR_curve.png`  
- `train_batch*.jpg` and `val_batch*.jpg`: Example images for visual inspection  

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or support, please open an issue on the repository.
```
