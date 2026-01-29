# Cardio AI: ECG Structural Heart Disease Screener

An AI-powered screening tool that detects structural heart abnormalities from standard 12-lead ECGs using Deep Learning (ResNet1d-50).

## Project Overview
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

## Installation

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

## Usage

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

## Phase 1: Blind Interpretation (Completed)
The initial model was a **Signal-Only** classifier, similar to how a cardiologist reads an ECG without knowing the patient.
*   **Model**: ResNet1d-50 trained on raw waveforms.
*   **Limitation**: It struggled to differentiate between **Ischemia (MI)** and **Digitalis Effect** (drug-induced ST depression), leading to False Positives.
*   *Verdict*: Good signal processing, but lacks clinical context.

### Results (Phase 1: Blind)
| Diagnostic Class | AUROC | Observation |
| :--- | :--- | :--- |
| **NORM** (Normal) | **0.94** | Excellent detection of healthy patients. |
| **MI** (Infarction) | **0.88** | Moderate. **Confused by Digitalis effect.** |
| **STTC** (ST/T Changes) | **0.86** | Struggled to distinguish from MI and drugs. |
| **CD** (Conduction Dist.) | **0.91** | Good detection of bundle branch blocks. |
| **HYP** (Hypertrophy) | **0.89** | Reasonable sensitivity. |
| **AVERAGE** | **0.90** | Strong baseline, but lacks specificity in complex cases. |

## Phase 2: Multimodal Fusion (Current)
We have enhanced the model to mimic a clinician's workflow by incorporating **Clinical Metadata**.
*   **Dual-Branch Architecture**:
    *   **Branch A (Vision)**: ResNet1d-50 processing the raw ECG waveform.
    *   **Branch B (Context)**: MLP processing tabular patient data.
*   **Context Features (11 Dimensions)**:
    *   **Demographics**: Age (with Missingness Mask), Sex, Weight, Height.
    *   **Medical History**: Presence of Pacemaker (mined from labels).
    *   **Medication**: Detection of Digoxin/Quinidine usage (to reduce False Positive Ischemia).

## System Design & Technical Architecture
### Multimodal Late Fusion
To achieve robust performance, we use a **Late Fusion** architecture:
1.  **Visual Backbone (ResNet1d-50)**:
    *   Input: `(12, 5000)` ECG Signal (500Hz, 10s).
    *   Output: 128-dimensional latent vector (The "Visual Embedding").
    *   *Why*: ResNets are state-of-the-art for time-series extraction.
2.  **Context Network (MLP)**:
    *   Input: 11-dimensional Clinical Vector (Age, Sex, Weight, Drugs).
    *   Architecture: `Linear(11->64) -> ReLU -> Dropout -> Linear(64->16)`.
    *   Output: 16-dimensional latent vector (The "Risk Embedding").
    *   *Why*: We use **Medical Entity Embedding** to let the model learn non-linear risk profiles (e.g., "Male+Overweight = High Risk", "Female+Underweight = Different Risk").
3.  **Fusion Head**:
    *   Operation: Concatenation `[Visual_128, Context_16]`.
    *   Final Classifier: `Linear(144 -> 5 Classes)`.

## Data Nuances & Cleaning
Real-world medical data is messy. We implemented several specific cleaning steps:
*   **The "Age 300" Problem**: Legacy sensors recorded unknown ages as `300`. We implemented a cleaning logic: `if age > 120 or NaN -> Age=0, Mask=0` (Zero-Masking Strategy).
*   **The "Digitalis Scoop"**: We explicitly mine the `scp_codes` for Digoxin (`DIG`) usage. This creates a feature `On_Digoxin=1`.
    *   *Impact*: Digoxin causes ST-depression (a "scoop" shape) that mimics myocardial infarction. Without this context, the model falsely predicts Heart Attacks. With this context, it learns to ignore the drug-induced artifact.

## References
1.  **PTB-XL Dataset**: Wagner, P. et al. "PTB-XL, a large publicly available electrocardiography dataset." Scientific Data (2020).
2.  **ResNet1d**: He, K. et al. "Deep Residual Learning for Image Recognition." CVPR (2016) (Adapted for 1D Signal).
3.  **Grad-CAM**: Selvaraju, R. R. et al. "Grad-CAM: Visual Explanations from Deep Networks." ICCV (2017).

## Project Structure
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
