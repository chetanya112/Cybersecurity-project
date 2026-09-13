import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error

# Config
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRIMARY_DATA = "data/processed/model_ready_v2_primary/dataset.npz"
META_PATH = "data/processed/model_ready_v2_primary/metadata.json"
OUTPUT_DIR = "models/fallback"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FALLBACK_FEATURES = [
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts"
]

# Models
class AttackClassifierLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_classes=15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class DirectFutureStateModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, future_steps=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, future_steps * input_dim)
        self.future_steps = future_steps
        self.input_dim = input_dim
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, self.future_steps, self.input_dim)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def train():
    print(f"Loading data from {PRIMARY_DATA}")
    data = np.load(PRIMARY_DATA, allow_pickle=True)
    meta = json.load(open(META_PATH, "r"))
    
    # Identify indices
    original_features = meta["feature_columns_order"]
    idx_map = {feat: i for i, feat in enumerate(original_features)}
    fallback_indices = [idx_map[feat] for feat in FALLBACK_FEATURES]
    
    # The arrays are already split and scaled! Wait, they are already scaled by the primary scaler!
    # If they are already scaled, I can't easily re-scale them with the fallback scaler.
    # Actually, I CAN just train on the raw data? No, the NPZ is scaled!
    # Let me check if the NPZ is scaled. Yes, `model_ready_v2_primary` is usually scaled.
    # Let me just unscale it using the primary scaler, then rescale.
    
    # Let's load the primary scaler
    primary_scaler = joblib.load(os.path.join(os.path.dirname(PRIMARY_DATA), "scaler.joblib"))
    
    def extract_and_rescale(X_scaled, Y_scaled):
        # X is (N, 10, 71)
        # Y is (N, 10, 71)
        B, T, F = X_scaled.shape
        # Unscale
        X_flat = X_scaled.reshape(-1, F)
        X_raw = primary_scaler.inverse_transform(X_flat).reshape(B, T, F)
        
        B_y, T_y, F_y = Y_scaled.shape
        Y_flat = Y_scaled.reshape(-1, F_y)
        Y_raw = primary_scaler.inverse_transform(Y_flat).reshape(B_y, T_y, F_y)
        
        # Extract the 6 features
        X_raw_6 = X_raw[:, :, fallback_indices]
        Y_raw_6 = Y_raw[:, :, fallback_indices]
        return X_raw_6, Y_raw_6
        
    X_train_raw, Y_train_raw = extract_and_rescale(data['X_train'], data['Y_train'])
    X_val_raw, Y_val_raw = extract_and_rescale(data['X_val'], data['Y_val'])
    X_test_raw, Y_test_raw = extract_and_rescale(data['X_test'], data['Y_test'])
    
    y_train_att = data['ylab_train']
    y_val_att = data['ylab_val']
    y_test_att = data['ylab_test']
    
    print(f"Train shapes: {X_train_raw.shape}, {Y_train_raw.shape}, {y_train_att.shape}")
    
    # Fit fallback scaler on train only
    scaler = StandardScaler()
    flat_train = X_train_raw.reshape(-1, 6)
    scaler.fit(flat_train)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    
    def scale_data(data3d):
        B, T, F = data3d.shape
        flat = data3d.reshape(-1, F)
        scaled = scaler.transform(flat)
        return scaled.reshape(B, T, F)
        
    X_train_hist = scale_data(X_train_raw)
    X_val_hist = scale_data(X_val_raw)
    X_test_hist = scale_data(X_test_raw)
    y_train_fut = scale_data(Y_train_raw)
    y_val_fut = scale_data(Y_val_raw)
    y_test_fut = scale_data(Y_test_raw)
    
    label_classes = meta['label_classes']
    label_to_idx = {l: i for i, l in enumerate(label_classes)}
    
    def map_labels(ylab):
        return np.array([label_to_idx[l[0]] for l in ylab], dtype=np.int64)
        
    y_train_att = map_labels(data['ylab_train'])
    y_val_att = map_labels(data['ylab_val'])
    y_test_att = map_labels(data['ylab_test'])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_hist), torch.FloatTensor(y_train_fut), torch.LongTensor(y_train_att))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    clf = AttackClassifierLSTM().to(DEVICE)
    fs = DirectFutureStateModel().to(DEVICE)
    
    opt_clf = optim.Adam(clf.parameters(), lr=LR)
    opt_fs = optim.Adam(fs.parameters(), lr=LR)
    
    criterion_clf = nn.CrossEntropyLoss()
    criterion_fs = nn.MSELoss()
    
    print("\n--- Training Fallback Models ---")
    for epoch in range(EPOCHS):
        clf.train()
        fs.train()
        total_loss_clf = 0
        total_loss_fs = 0
        
        for batch_hist, batch_fut, batch_att in train_loader:
            batch_hist, batch_fut, batch_att = batch_hist.to(DEVICE), batch_fut.to(DEVICE), batch_att.to(DEVICE)
            
            # Train Clf
            opt_clf.zero_grad()
            preds_att = clf(batch_hist)
            loss_clf = criterion_clf(preds_att, batch_att)
            loss_clf.backward()
            opt_clf.step()
            total_loss_clf += loss_clf.item()
            
            # Train FS
            opt_fs.zero_grad()
            preds_fut = fs(batch_hist)
            loss_fs = criterion_fs(preds_fut, batch_fut)
            loss_fs.backward()
            opt_fs.step()
            total_loss_fs += loss_fs.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Clf Loss: {total_loss_clf/len(train_loader):.4f} | FS Loss: {total_loss_fs/len(train_loader):.4f}")
        
    torch.save(clf.state_dict(), os.path.join(OUTPUT_DIR, "attack_classifier.pt"))
    torch.save(fs.state_dict(), os.path.join(OUTPUT_DIR, "state_transition_direct.pt"))
    
    # Save Metadata
    fallback_meta = {
        "engine": "fallback",
        "feature_count": 6,
        "feature_names": FALLBACK_FEATURES,
        "sequence_length": 10,
        "window_seconds": 30,
        "label_mapping": label_to_idx,
        "label_classes": meta["label_classes"]
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(fallback_meta, f, indent=4)
        
    print("\n--- Evaluation on Test Set ---")
    clf.eval()
    fs.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test_hist).to(DEVICE)
        out_att = clf(X_t)
        out_fut = fs(X_t)
        
        probs = torch.softmax(out_att, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        fs_preds = out_fut.cpu().numpy()
        
    y_true = y_test_att
    macro_f1 = f1_score(y_true, preds, average='macro')
    acc = np.mean(preds == y_true)
    precision = precision_score(y_true, preds, average='macro', zero_division=0)
    recall = recall_score(y_true, preds, average='macro', zero_division=0)
    
    brier = np.mean(np.sum((probs - np.eye(15)[y_true])**2, axis=1))
    
    # ECE for binary "is_attack" (class > 0)
    is_attack_true = (y_true > 0).astype(int)
    attack_probs = 1.0 - probs[:, 0]
    ece = expected_calibration_error(is_attack_true, attack_probs)
    
    fs_true = y_test_fut
    mae = mean_absolute_error(fs_true.reshape(-1, 6), fs_preds.reshape(-1, 6))
    rmse = np.sqrt(mean_squared_error(fs_true.reshape(-1, 6), fs_preds.reshape(-1, 6)))
    
    print(f"Test Samples: {len(X_test_hist)}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE (Attack): {ece:.4f}")
    print(f"Future MAE: {mae:.4f}")
    print(f"Future RMSE: {rmse:.4f}")
    
if __name__ == "__main__":
    train()
