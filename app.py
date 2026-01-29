import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import os
import ast
import wfdb

# Project Imports
from src.models.resnet1d import resnet1d50

# Configuration
DATA_DIR = os.path.join("data", "ptb-xl")
MODEL_PATH = os.path.join("src", "models", "resnet1d_best.pth")
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# 1. Load Resources (Cached)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet1d50(num_classes=5)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        st.warning("Model weights not found. Running with random weights for demo UI purposes.")
    
    model.to(device)
    model.eval()
    return model, device

@st.cache_data
def load_metadata():
    csv_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    # Load SCP Statements for Mapping
    scp_path = os.path.join(DATA_DIR, "scp_statements.csv")
    if os.path.exists(scp_path):
        agg_df = pd.read_csv(scp_path, index_col=0)
        agg_df = agg_df[agg_df.diagnostic_class.notnull()]
        code_map = agg_df.diagnostic_class.to_dict()
        
        # Helper to get superclass list
        def get_superclasses(scp_str):
            try:
                d = ast.literal_eval(scp_str)
                res = set()
                for k in d:
                    if k in code_map:
                        res.add(code_map[k])
                return list(res)
            except:
                return []
        
        df['superclasses'] = df['scp_codes'].apply(get_superclasses)
    else:
        df['superclasses'] = []
        
    return df

# 2. Helper Functions
def load_signal(filename):
    # Handle both 100/500Hz paths if needed, usually 500 for this app
    path = os.path.join(DATA_DIR, filename)
    try:
        data, _ = wfdb.rdsamp(path)
        # Normalize
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-6)
    except Exception as e:
        return np.zeros((5000, 12))

def plot_ecg(signal, title="12-Lead ECG"):
    # Signal is (5000, 12)
    fig = go.Figure()
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Plot a few leads for clarity (stacked)
    # We create subplots or just offset them
    for i, lead in enumerate(leads):
        # Offset graph by i * 2 to stack them
        fig.add_trace(go.Scatter(y=signal[:, i] + (i * 3), mode='lines', name=lead))
    
    fig.update_layout(
        title=title,
        height=800,
        yaxis=dict(showticklabels=False, title="Leads (Stacked)"),
        xaxis=dict(title="Time (samples)"),
        template="plotly_dark"
    )
    return fig

# 3. GradCAM Implementation
def compute_gradcam(model, x, target_class_idx):
    # x: (1, 12, 5000)
    # Target last conv layer: model.layer4[-1]
    
    # Hook for gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    # Register hooks on the last bottleneck block of layer4
    target_layer = model.layer4[-1]
    h1 = target_layer.register_backward_hook(backward_hook)
    h2 = target_layer.register_forward_hook(forward_hook)
    
    # Forward
    output = model(x)
    model.zero_grad()
    
    # Backward target
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Get Grads & Acts
    grads = gradients[0].cpu().data.numpy()[0] # (Channels, Length)
    acts = activations[0].cpu().data.numpy()[0] # (Channels, Length)
    
    # Global Average Pooling of gradients (Weights)
    weights = np.mean(grads, axis=1) # (Channels,)
    
    # Weighted combination
    cam = np.zeros(acts.shape[1], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Upsample to 5000
    # Simple linear interpolation
    cam = np.interp(np.linspace(0, len(cam), 5000), np.linspace(0, len(cam), len(cam)), cam)
    
    # Normalize
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    h1.remove()
    h2.remove()
    return cam

# 4. Main App UI
def main():
    st.set_page_config(page_title="Cardio AI Screener", page_icon="❤️", layout="wide")
    st.title("❤️ AI Structural Heart Disease Screener")
    st.markdown("Analyzing **10-second 12-lead ECGs** for hidden pathologies.")

    with st.expander("ℹ️ Guide: Diagnostic Categories"):
        st.markdown("""
        The model classifies ECGs into 5 "Superclasses":
        
        | Class | Name | Description |
        |-------|------|-------------|
        | **NORM** | **Normal** | Healthy heart rhythm and structure. No abnormalities. |
        | **MI** | **Myocardial Infarction** | Heart Attack (past or present). Signs of tissue damage or ischemia. |
        | **STTC** | **ST/T Changes** | Repolarization abnormalities. Can indicate ischemia, electrolyte imbalance, or drug effects (like Digitalis). |
        | **CD** | **Conduction Disturbance** | "Wiring" issues. Blockages in the electrical pathways (e.g., Bundle Branch Blocks). |
        | **HYP** | **Hypertrophy** | Enlarged heart muscle (thickening of walls), often due to high blood pressure. |
        """)

    model, device = load_model()
    df = load_metadata()

    if df is None:
        st.error(f"Data not found! Please place 'ptbxl_database.csv' in `{DATA_DIR}`.")
        return

    # Sidebar: Selection
    st.sidebar.header("Patient Selection")
    
    # Filter for Curated Examples
    # We try to find 2 of each class
    curated_ids = []
    
    example_mode = st.sidebar.radio("Input Source", ["Curated Examples", "Random Patient", "By ID"])
    
    selected_id = None
    
    if example_mode == "Curated Examples":
        # Quick lookup (cached ideally)
        examples = {}
        for cls in CLASSES:
            # Find rows where superclasses list contains this class
            # df['superclasses'] is a list of strings
            
            # Simple apply to filter
            matches = df[df['superclasses'].apply(lambda x: cls in x if isinstance(x, list) else False)]
            
            if len(matches) > 0:
                examples[cls] = matches.index[:5].tolist() # Take top 5
        
        if not examples:
            st.warning("No examples found. Check scp_statements.csv mapping.")
            return

        category = st.sidebar.selectbox("Disease Type", list(examples.keys()))
        selected_id = st.sidebar.selectbox("Patient ID", examples[category])
        
    elif example_mode == "Random Patient":
        if st.sidebar.button("Roll Dice"):
            selected_id = np.random.choice(df.index)
        else:
            selected_id = df.index[0]
            
    else:
        selected_id = st.sidebar.number_input("Enter ECG ID", min_value=1, max_value=22000, value=1)

    if selected_id not in df.index:
        st.error("ID not found.")
        return

    # Load & Infer
    row = df.loc[selected_id]
    signal_raw = load_signal(row.get('filename_hr', row.get('filename_lr')))
    
    # Prepare Tensor
    x_np = signal_raw.T # (12, 5000)
    x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # --- Display ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"ECG Trace (Patient {selected_id})")
        st.plotly_chart(plot_ecg(signal_raw), use_container_width=True)
        
    with col2:
        st.subheader("AI Diagnosis")
        
        # Ground Truth (Parse metadata)
        st.caption("Ground Truth Labels (from Report):")
        try:
            gt = ast.literal_eval(row['scp_codes'])
            st.write(gt)
        except:
            st.write("Unknown")
            
        st.divider()
        
        # Predictions
        for i, cls in enumerate(CLASSES):
            p = probs[i]
            delta = p - 0.5 # Simple heuristic
            st.metric(label=cls, value=f"{p:.1%}", delta=None)
            st.progress(float(p))
            
    # Explainability Section
    st.divider()
    st.subheader("Explainability (Saliency Map)")
    
    target_cls = st.selectbox("Explain why model predicts:", CLASSES, index=int(np.argmax(probs)))
    target_idx = CLASSES.index(target_cls)
    
    if st.button(f"Generate Heatmap for {target_cls}"):
        cam = compute_gradcam(model, x_tensor, target_idx)
        
        # Plot CAM overlay on Lead II (most common)
        lead_ii = signal_raw[:, 1]
        
        fig_cam = go.Figure()
        fig_cam.add_trace(go.Scatter(y=lead_ii, mode='lines', name='Lead II', line=dict(color='white', width=1)))
        
        # Color the line by importance? Or overlay heatmap?
        # ScatterGL with marker color is good
        fig_cam.add_trace(go.Scatter(
            x=np.arange(5000), 
            y=lead_ii, 
            mode='markers',
            marker=dict(
                size=5,
                color=cam, # offset/scale
                colorscale='Hot',
                showscale=True,
                opacity=0.5
            ),
            name="Attention"
        ))
        
        fig_cam.update_layout(title=f"Where the Model looked for {target_cls}", template="plotly_dark")
        st.plotly_chart(fig_cam, use_container_width=True)

if __name__ == "__main__":
    main()
