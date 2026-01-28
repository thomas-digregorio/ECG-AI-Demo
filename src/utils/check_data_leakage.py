import sys
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import ast

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.train import load_metadata, DATA_DIR

def check_leakage():
    print("Checking for Data Leakage (Patient Overlap)...")
    try:
        df = load_metadata(DATA_DIR)
    except Exception as e:
        print(f"Skipping check: Data not found ({e})")
        return

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(sgkf.split(df.index, df['strat_target'], df['patient_id']))
    
    # Check Fold 1
    train_idx, val_idx = splits[0]
    train_patients = set(df.iloc[train_idx]['patient_id'])
    val_patients = set(df.iloc[val_idx]['patient_id'])
    
    overlap = train_patients.intersection(val_patients)
    
    if len(overlap) == 0:
        print("✅ SUCCESS: 0 patients overlap between Train and Validation.")
        print(f"   Train Patients: {len(train_patients)}")
        print(f"   Val Patients:   {len(val_patients)}")
    else:
        print(f"❌ CRITICAL FAIL: {len(overlap)} patients found in both sets!")
        print(list(overlap)[:5])

if __name__ == "__main__":
    check_leakage()
