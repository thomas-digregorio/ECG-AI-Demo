import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wfdb
import ast

# Project Imports
from src.data.dataset import PTBXLDataset
from src.models.resnet1d import resnet1d50

# Configuration
DATA_DIR = os.path.join("data", "ptb-xl") # Assumes running from root
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
NUM_CLASSES = 5

def load_metadata(data_dir):
    """Loads and preprocesses metadata for splitting."""
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata not found at {csv_path}. Did you download the data?")
    
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    # Process scp_codes to get a single 'stratify_label' for splitting
    # We pick the most frequent class for stratification purposes if multiple exist
    # Or just use the first valid superclass found
    lbl_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    
    # Helper to find primary class
    def get_primary_class(scp_str):
        try:
            d = ast.literal_eval(scp_str)
            for k in d.keys():
                if k in lbl_map:
                    return lbl_map[k]
        except:
            pass
        return 0 # Default to NORM

    df['strat_target'] = df['scp_codes'].apply(get_primary_class)
    return df

def calculate_pos_weights(dataset):
    """Calculates pos_weight for BCEWithLogitsLoss based on training data imbalance."""
    # This is a heuristic estimation based on the dataset labels
    # We iterate once to count (or use pandas metadata if trusted)
    # For speed, we'll use pandas metadata 'scp_codes'
    
    counts = np.zeros(NUM_CLASSES)
    total = len(dataset)
    
    print("Calculating class balance for Loss Weights...")
    # Using the dataset's get_labels might be slow if we iterate all, 
    # but let's do it safely via pandas source to be fast
    df = dataset.metadata
    lbl_map = dataset.lbl_map
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calc Weights"):
        try:
            d = ast.literal_eval(row['scp_codes'])
            for k in d:
                if k in lbl_map:
                    counts[lbl_map[k]] += 1
        except:
            pass
            
    # Weight = (Negative / Positive)
    # Avoid div by zero
    counts = np.maximum(counts, 1)
    negatives = total - counts
    weights = negatives / counts
    
    return torch.tensor(weights, dtype=torch.float32)

def train(fold=1):
    # 1. Setup Device
    if not torch.cuda.is_available():
        raise SystemError("CRITICAL: CUDA is not available. Training on CPU is strictly forbidden as per user request.")
    
    device = torch.device("cuda")
    print(f"Training on: {device} ({torch.cuda.get_device_name(0)})")
    print("Optimization: Enable cuDNN benchmarking for fixed input sizes...")
    torch.backends.cudnn.benchmark = True

    # 2. Data Preparation
    try:
        df = load_metadata(DATA_DIR)
    except FileNotFoundError:
        print(f"Error: Data not found in {DATA_DIR}. Please run 'src/data/download_ptbxl.py' first.")
        return

    # 3. Stratified Group Split
    # Split by Patient ID to prevent leakage
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
    # We just want the 'fold'-th split
    # X=index, y=strat_target, groups=patient_id
    splits = list(sgkf.split(df.index, df['strat_target'], df['patient_id']))
    train_idx, val_idx = splits[fold-1] # 0-indexed list, 1-based fold arg
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Fold {fold}: Train size {len(train_df)}, Val size {len(val_df)}")

    # Create Datasets (RAM Cache enabled for speed)
    train_dataset = PTBXLDataset(train_df, DATA_DIR, sampling_rate=500, use_ram_cache=True)
    val_dataset = PTBXLDataset(val_df, DATA_DIR, sampling_rate=500, use_ram_cache=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 0 workers if caching to avoid fork overhead
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model Setup
    model = resnet1d50(num_classes=NUM_CLASSES).to(device)
    
    # 5. Weighted Loss
    pos_weight = calculate_pos_weights(train_dataset).to(device)
    print(f"Using Positional Weights: {pos_weight.cpu().numpy()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    # 6. WandB (Optional Check)
    import wandb
    wandb.init(project="cardio-ecg-screen", padding=fold, config={"model": "resnet1d50", "fold": fold})
    wandb.watch(model)

    # 7. Training Loop
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                # Sigmoid for probabilities
                probs = torch.sigmoid(outputs)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())
                
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        # Calculate Macro AUROC
        try:
            val_auc = roc_auc_score(val_targets, val_preds, average='macro')
        except ValueError:
            val_auc = 0.5 # Handle edge cases if a class is missing in val batch
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val AUROC: {val_auc:.4f}")
        
        wandb.log({
            "train_loss": train_loss/len(train_loader),
            "val_loss": val_loss/len(val_loader),
            "val_auroc": val_auc,
            "lr": scheduler.get_last_lr()[0]
        })
        
        # Save Best
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join("src", "models", "resnet1d_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, help='Fold number (1-10) for validation')
    args = parser.parse_args()
    
    train(fold=args.fold)
