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
import sys

# Ensure 'src' module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Fusion Imports
from src.data.dataset_fusion import PTBXLDatasetFusion
from src.models.resnet1d_fusion import resnet1d_fusion
# We iterate on train.py's load_metadata logic but import it locally if needed,
# or better yet, duplicate minimal logic to be self-contained in this phase 2 script.

# Configuration
DATA_DIR = os.path.join("data", "ptb-xl")
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
NUM_CLASSES = 5

def load_metadata(data_dir):
    """Loads and preprocesses metadata for splitting."""
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata not found at {csv_path}.")
    
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    # --- Load SCP Statements for Mapping ---
    scp_path = os.path.join(data_dir, "scp_statements.csv")
    if os.path.exists(scp_path):
        agg_df = pd.read_csv(scp_path, index_col=0)
        agg_df = agg_df[agg_df.diagnostic_class.notnull()]
        code_map = agg_df.diagnostic_class.to_dict()
    else:
        code_map = {}

    def aggregate_diagnostic(y_dic):
        tmp = []
        try:
            if isinstance(y_dic, str): y_dic = ast.literal_eval(y_dic)
            for key in y_dic:
                if key in code_map: tmp.append(code_map[key])
        except: pass
        return list(set(tmp))

    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    lbl_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    
    def get_primary_class(superclasses):
        for c in superclasses:
            if c in lbl_map: return lbl_map[c]
        return 0

    df['strat_target'] = df['diagnostic_superclass'].apply(get_primary_class)
    return df

def calculate_pos_weights(dataset):
    counts = np.zeros(NUM_CLASSES)
    total = len(dataset)
    df = dataset.metadata
    lbl_map = dataset.lbl_map
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calc Weights"):
        try:
            if 'diagnostic_superclass' in row:
                for k in row['diagnostic_superclass']:
                    if k in lbl_map: counts[lbl_map[k]] += 1
        except: pass
            
    counts = np.maximum(counts, 1)
    negatives = total - counts
    weights = negatives / counts
    return torch.tensor(weights, dtype=torch.float32)

def train_fusion(fold=1):
    if not torch.cuda.is_available():
        raise SystemError("CRITICAL: CUDA is not available.")
    
    device = torch.device("cuda")
    print(f"Training Fusion Model on: {device}")
    torch.backends.cudnn.benchmark = True

    try:
        df = load_metadata(DATA_DIR)
    except FileNotFoundError:
        return

    # Stratified Group Split
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(sgkf.split(df.index, df['strat_target'], df['patient_id']))
    train_idx, val_idx = splits[fold-1]
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Fold {fold}: Train size {len(train_df)}, Val size {len(val_df)}")

    # Create Fusion Datasets
    train_dataset = PTBXLDatasetFusion(train_df, DATA_DIR, sampling_rate=500, use_ram_cache=True)
    val_dataset = PTBXLDatasetFusion(val_df, DATA_DIR, sampling_rate=500, use_ram_cache=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Fusion Model
    model = resnet1d_fusion(num_classes=NUM_CLASSES).to(device)
    
    pos_weight = calculate_pos_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    import wandb
    wandb.init(project="cardio-ecg-screen", config={"model": "resnet1d_fusion", "fold": fold})
    wandb.watch(model)

    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x_img, x_ctx, y in loop:
            x_img, x_ctx, y = x_img.to(device), x_ctx.to(device), y.to(device)
            
            optimizer.zero_grad()
            # FUSION FORWARD PASS
            outputs = model(x_img, x_ctx)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        model.eval()
        val_preds, val_targets, val_loss = [], [], 0
        
        with torch.no_grad():
            for x_img, x_ctx, y in val_loader:
                x_img, x_ctx, y = x_img.to(device), x_ctx.to(device), y.to(device)
                outputs = model(x_img, x_ctx)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())
                
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        try:
            val_auc = roc_auc_score(val_targets, val_preds, average='macro')
        except:
            val_auc = 0.5
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val AUROC: {val_auc:.4f}")
        
        wandb.log({
            "train_loss": train_loss/len(train_loader),
            "val_loss": val_loss/len(val_loader),
            "val_auroc": val_auc,
            "lr": scheduler.get_last_lr()[0]
        })
        
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join("src", "models", "resnet1d_fusion_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    train_fusion(fold=args.fold)
