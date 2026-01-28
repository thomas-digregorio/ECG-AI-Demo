# Cardio AI: ECG Structural Heart Disease Screener

An AI-powered screening tool that detects structural heart abnormalities from standard 12-lead ECGs using Deep Learning (ResNet1d-50).

## 🫀 Project Overview
This project classifies ECG signals into 5 Diagnostic Superclasses based on the **PTB-XL** dataset standard:
1.  **NORM**: Normal ECG
2.  **MI**: Myocardial Infarction
3.  **STTC**: ST/T Change
4.  **CD**: Conduction Disturbance
5.  **HYP**: Hypertrophy

It includes:
- A **Streamlit Dashboard** for interactive visualization and demo.
- **Saliency Maps (Grad-CAM)** to explain model predictions.
- **Stratified Group K-Fold** training evaluation to prevent patient leakage.

## 🛠️ Installation

1.  **Create Environment**
    ```bash
    conda create -n cardio_ai python=3.12 -y
    conda activate cardio_ai
    ```

2.  **Install Dependencies**
    *(Includes PyTorch with CUDA 12.8 support)*
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    ```

3.  **Data Setup**
    - Download the **PTB-XL** dataset (version 1.0.3) from [PhysioNet](https://physionet.org/content/ptb-xl/).
    - Extract it so `ptbxl_database.csv` is at `data/ptb-xl/ptbxl_database.csv`.

## 🚀 Usage

### 1. Training
Train the ResNet1d-50 model on your local GPU.
```bash
python src/train.py --fold=1
```
*   Logs metrics to **Weights & Biases**.
*   Saves best weights to `src/models/resnet1d_best.pth`.

### 2. Verification
Check for data leakage (ensure no patient overlaps between train/val splits).
```bash
python src/utils/check_data_leakage.py
```

### 3. Run Demo App
Launch the interactive dashboard.
```bash
streamlit run app.py
```

## 🏗️ Project Structure
```
.
├── app.py                  # Streamlit Dashboard Entrypoint
├── notebooks/              # Jupyter Notebooks for EDA & Analysis
├── requirements.txt        # Python Dependencies
├── src/
│   ├── data/               # Dataset loading & Preprocessing
│   ├── models/             # ResNet1d Architecture
│   ├── utils/              # Helper scripts
│   └── train.py            # Training Loop
└── data/                   # (Ignored by Git) Local dataset storage
```
