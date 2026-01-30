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
from src.models.resnet1d_fusion import resnet1d_fusion

# Configuration
DATA_DIR = os.path.join("data", "ptb-xl")
MODEL_PATH = os.path.join("src", "models", "resnet1d_fusion_best.pth")
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# 1. Load Resources (Cached)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet1d_fusion(num_classes=5)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            st.error(f"Failed to load weights: {e}")
    else:
        st.warning(f"Fusion Model weights not found at {MODEL_PATH}. Training needed?")
    
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
    path = os.path.join(DATA_DIR, filename)
    try:
        data, _ = wfdb.rdsamp(path)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-6)
    except:
        return np.zeros((5000, 12))

def plot_ecg(signal, title="12-Lead ECG"):
    fig = go.Figure()
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for i, lead in enumerate(leads):
        fig.add_trace(go.Scatter(y=signal[:, i] + (i * 3), mode='lines', name=lead))
    
    fig.update_layout(
        title=title,
        height=600,
        yaxis=dict(showticklabels=False, title="Leads (Stacked)"),
        xaxis=dict(title="Time"),
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# 3. Clinical Data Logic
def extract_default_context(row):
    # Extract defaults from DataFrame row
    age = row['age'] if not pd.isna(row['age']) else 50
    sex = 'Female' if row['sex'] == 1 else 'Male'
    height = row['height'] if not pd.isna(row['height']) else 170
    weight = row['weight'] if not pd.isna(row['weight']) else 70
    
    # Check drugs
    scp = row['scp_codes']
    has_dig = 'DIG' in scp
    has_qu = 'QU' in scp
    has_mim = 'MIM' in scp
    
    is_pace = False
    if isinstance(row['pacemaker'], str) and 'pacemaker' in row.get('pacemaker','').lower():
        is_pace = True
        
    return age, sex, height, weight, is_pace, has_dig, has_qu, has_mim

# 4. Main App
def main():
    st.set_page_config(page_title="Cardio Fusion", page_icon="🧬", layout="wide")
    st.title("🧬 Cardio AI: Multimodal Fusion Screener")
    st.markdown("Combines **ECG Vision** + **Clinical Context** for higher accuracy.")

    model, device = load_model()
    df = load_metadata()

    if df is None:
        st.error("Data missing.")
        return

    # --- Sidebar: Patient Selection ---
    st.sidebar.header("1. Select Patient")
    example_mode = st.sidebar.radio("Source", ["Curated Examples", "Random", "By ID"], horizontal=True)
    
    selected_id = None
    if example_mode == "Curated Examples":
        # Quick find logic
        examples = {}
        for cls in CLASSES:
            subset = df[df['superclasses'].apply(lambda x: cls in x)]
            if len(subset) > 0: examples[cls] = subset.index[:5]
        
        cat = st.sidebar.selectbox("Category", list(examples.keys()))
        selected_id = st.sidebar.selectbox("ID", examples[cat])
    elif example_mode == "Random":
        if st.sidebar.button("Shuffle"): selected_id = np.random.choice(df.index)
        else: selected_id = df.index[0]
    else:
        selected_id = st.sidebar.number_input("ID", 1, 22000, 1)

    if selected_id not in df.index:
        st.error("Invalid ID")
        return

    # Load Signal
    row = df.loc[selected_id]
    signal = load_signal(row.get('filename_hr', row.get('filename_lr')))

    # --- Sidebar: Clinical Context (Interactive) ---
    st.sidebar.divider()
    st.sidebar.header("2. Clinical Context (Edit to Test)")
    
    # Get Defaults
    d_age, d_sex, d_h, d_w, d_pace, d_dig, d_qu, d_mim = extract_default_context(row)
    
    # Inputs
    age = st.sidebar.slider("Age", 0, 120, int(d_age))
    sex = st.sidebar.radio("Sex", ["Male", "Female"], index=0 if d_sex=='Male' else 1)
    
    col_a, col_b = st.sidebar.columns(2)
    height = col_a.number_input("Height (cm)", 50, 250, int(d_h))
    weight = col_b.number_input("Weight (kg)", 20, 200, int(d_w))
    
    st.sidebar.subheader("History & Meds")
    pace = st.sidebar.checkbox("Has Pacemaker", value=d_pace)
    dig = st.sidebar.checkbox("Taking Digoxin", value=d_dig)
    qu = st.sidebar.checkbox("Taking Quinidine", value=d_qu)
    
    # --- Inference ---
    # Construct Context Vector (11 dim)
    # [AgeVal, AgeMask, Sex(1=F), H_Val, H_Mask, W_Val, W_Mask, Pace, Dig, Qu, MIM]
    
    # Normalization Logic (Must match dataset_fusion.py)
    # Age
    age_val = age / 100.0
    age_mask = 1.0 # UI always provides a value
    
    # Sex
    sex_val = 1.0 if sex == "Female" else 0.0
    
    # Height
    h_val = 0.0 if np.isnan(height) else min(max(int(height), 0), 250) / 250.0
    h_mask = 1.0
    
    # Weight
    w_val = 0.0 if np.isnan(weight) else min(max(int(weight), 0), 300) / 300.0
    w_mask = 1.0
    
    # Flags
    p_val = 1.0 if pace else 0.0
    d_val = 1.0 if dig else 0.0
    q_val = 1.0 if qu else 0.0
    m_val = 1.0 if d_mim else 0.0 # Keep original MIM status, not exposed to UI to avoid confusion
    
    ctx = torch.tensor([
        age_val, age_mask, sex_val, h_val, h_mask, w_val, w_mask, p_val, d_val, q_val, m_val
    ], dtype=torch.float32).unsqueeze(0).to(device)
    
    img = torch.tensor(signal.T, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img, ctx)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # --- Main Layout ---
    col_l, col_r = st.columns([7, 3])
    
    with col_l:
        st.subheader(f"Patient {selected_id}")
        st.plotly_chart(plot_ecg(signal), use_container_width=True)
        
    with col_r:
        st.subheader("Model Prediction")
        
        # Check for Digitalis Warning
        if dig and probs[1] < 0.3: # If Taking Digoxin but MI probability is low
            st.info("ℹ️ Fusion Model detects Digoxin - suppressing false positive Ischemia.")
            
        for i, cls in enumerate(CLASSES):
            st.progress(float(probs[i]))
            st.caption(f"{cls}: {probs[i]:.1%}")

    st.divider()
    st.caption("Try toggling **'Taking Digoxin'** in the sidebar to see how the 'MI' prediction changes!")

if __name__ == "__main__":
    main()
