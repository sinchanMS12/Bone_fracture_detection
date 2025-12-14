

***

# 🦴 Bone Fracture Detection with Hybrid Deep Learning

Welcome to the Bone Fracture Detection repository. This project aims to automate the preliminary detection of bone fractures in X-ray images using a hybrid deep learning pipeline, providing a robust machine learning solution for healthcare applications.

***

## Status

- **Current Status:** In progress  
- **Completed:** ML model training, hybrid fusion design, and core evaluation pipeline

***

## Features

- **Automated Binary Classification:** Classifies X-ray images as **fractured** or **unfractured**.
- **Data Augmentation:** Uses rotation, shifting, and zooming to improve generalization and reduce overfitting.
- **Transfer Learning Baselines:** Trains and evaluates **DenseNet-121** and **ResNet-50** with **ImageNet** weights.
- **Hybrid Ensemble Fusion:** Concatenates feature maps from DenseNet-121 and ResNet-50 to form a high-performance hybrid model.
- **Staged Tuning:** First trains a new classification head, then fine-tunes backbone layers with differential learning rates.
- **Comprehensive Evaluation:** Computes and plots accuracy/loss curves, precision, recall, F1-score, confusion matrix, and ROC/AUC.
- **Explainability with Grad-CAM:** Generates heatmaps highlighting key regions (fracture areas) that drive the model’s predictions.

***

## Tech Stack

### Core Technology

- **Type:** Deep learning–based medical image classification  
- **Language:** Python (3.8+)  

### Frameworks and Libraries

- **Deep Learning:** TensorFlow / Keras  
- **ML Utilities:** Scikit-learn  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Preprocessing:** Keras-Preprocessing

### Architecture Overview

- **Feature Extraction:**  
  - Pre-trained **DenseNet-121** and **ResNet-50** (ImageNet weights).
- **Feature Fusion:**  
  - 4D feature maps concatenated into a single fused tensor (`concatenate_features`).
- **Classification Head:**  
  - `GlobalAveragePooling2D` → `Dense` → `Dropout` → final `Dense(1, activation="sigmoid")`.

### Integrations

- **Pre-trained Weights:** ImageNet initialization for DenseNet-121 and ResNet-50.  
- **Data Source:** Local bone X-ray dataset (train/test folders with `fractured` and `unfractured` subdirectories).

***

## Installation

### Prerequisites

Ensure the following are installed:

- Python 3.8 or later  
- `pip` (Python package manager)  

### Clone the Repository

```bash
git clone https://github.com/your-username/bone_fracture_detection.git
cd bone_fracture_detection
```

### Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn keras-preprocessing
```

***

## Configuration

### 1. Dataset Structure

Organize your dataset as:

```text
<DATA_ROOT_DIR>/BoneFractureDataset/
├── training/
│   ├── fractured/
│   └── unfractured/
└── testing/
    ├── fractured/
    └── unfractured/
```

### 2. Update Data Path

In `bone_fracture_pipeline.py`, update `DATA_ROOT_DIR` in the global configuration section:

```python
# bone_fracture_pipeline.py (GLOBAL CONFIGURATION)
# Use a raw string (r'...') for safe Windows paths
DATA_ROOT_DIR = r'C:\Users\user\OneDrive\Desktop\bone_fracture_detection\BoneFractureDataset'
```

Change the path to match your actual dataset location.

***

## Usage

### Running the Pipeline (With Pretrained Baselines)

The pipeline is configured to load existing baseline weights and proceed with fusion and tuning.

```bash
# Ensure virtual environment is active
python bone_fracture_pipeline.py
```

### Execution Flow

The main script performs:

1. **Data Loading & Augmentation** using `ImageDataGenerator`.  
2. **Load Baselines:** Loads `DenseNet121_best_weights.h5` and `ResNet50_best_weights.h5`.  
3. **Hybrid Model Build & Tuning:** Constructs the fusion architecture and runs two fine-tuning stages (e.g., 5 epochs each).  
4. **Inference & Reporting:**  
   - Generates `final_test_predictions.csv` with test predictions.  
   - Produces evaluation plots and Grad-CAM visualizations.

### Example Outputs

After a successful run, the `model_outputs/` folder may contain:

- `HybridModel_optimized_weights.h5`  
- `DenseNet121_best_weights.h5`  
- `HybridModel_Final_confusion_matrix.png`  
- ROC curve plot(s)  
- `final_test_predictions.csv`  
- `gradcam_output_<image_name>.png` (Grad-CAM heatmaps)

***

## Testing and Evaluation

Evaluation is integrated into the pipeline and runs automatically after training or loading models:

- Accuracy and loss curves  
- Precision, recall, F1-score  
- Confusion matrix and ROC/AUC plots  

There is no separate CLI test command; evaluation is triggered inside `main_pipeline` within `bone_fracture_pipeline.py`.

***

## Project Structure

```text
bone_fracture_detection/
├── BoneFractureDataset/      # Training and testing X-ray images
├── model_outputs/            # Saved weights, plots, and reports
│   ├── DenseNet121_best_weights.h5
│   ├── HybridModel_optimized_weights.h5
│   ├── final_test_predictions.csv
│   ├── *roc_curve.png
│   └── *confusion_matrix.png
└── bone_fracture_pipeline.py # Main end-to-end pipeline script
```

***

## Roadmap

Planned improvements:

- Final validation and configuration of Grad-CAM layer selection.  
- Exploration of more advanced ensemble methods (e.g., stacked generalization).  
- Addition of a lightweight inference API (e.g., **FastAPI**) for real-time predictions.

***

## Contributing

Contributions are welcome.

### How to Contribute

1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with clear messages.  
4. Push the branch and open a Pull Request against the `main` branch.

### Reporting Issues

- Open an issue on the GitHub **Issues** page.  
- Provide a clear description, steps to reproduce (if applicable), and any relevant logs or screenshots.

***

## License

This project is released under the **MIT License** (assumed).  
You can modify this section to match your actual license.

***

## Acknowledgements

- Pre-trained ImageNet weights used via Keras applications.  
- Open-source communities behind **TensorFlow**, **Keras**, **Scikit-learn**, and related libraries that power this project.
