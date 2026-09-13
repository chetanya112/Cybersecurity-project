import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Configurations
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRIMARY_DATA = "data/processed/model_ready_v2_primary/dataset.npz"
META_PATH = "data/processed/model_ready_v2_primary/metadata.json"
OUTPUT_DIR = "models/phase4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Models
class AttackClassifierLSTM(nn.Module):
    def __init__(self, input_dim=71, hidden_dim=64, num_classes=15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class DirectFutureStateModel(nn.Module):
    def __init__(self, input_dim=71, hidden_dim=128, future_steps=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, future_steps * input_dim)
        self.future_steps = future_steps
        self.input_dim = input_dim
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, self.future_steps, self.input_dim)

class RecursiveFutureStateModel(nn.Module):
    def __init__(self, input_dim=71, hidden_dim=128):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, history, future_targets=None, teacher_forcing_ratio=0.0, future_k=10):
        batch_size = history.size(0)
        if future_targets is not None: future_k = future_targets.size(1)
        
        _, (hn, cn) = self.encoder(history)
        dec_input = history[:, -1:, :]
        
        outputs = []
        for t in range(future_k):
            out, (hn, cn) = self.decoder(dec_input, (hn, cn))
            pred = self.fc(out)
            outputs.append(pred)
            
            if future_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = future_targets[:, t:t+1, :]
            else:
                dec_input = pred
                
        return torch.cat(outputs, dim=1)

# 2. Data Loading
print("Loading Protocol A...")
data = np.load(PRIMARY_DATA, allow_pickle=True)
X_tr, Y_tr, yl_tr = data['X_train'], data['Y_train'], data['ylab_train']
X_va, Y_va, yl_va = data['X_val'], data['Y_val'], data['ylab_val']
X_te, Y_te, yl_te = data['X_test'], data['Y_test'], data['ylab_test']

with open(META_PATH, 'r') as f:
    meta = json.load(f)
label_classes = meta['label_classes']
label_to_idx = {l: i for i, l in enumerate(label_classes)}

# For classifier, we need (X, final_label). The true final label of sequence history X is Y_labels[:, 0] ?
# Wait, no. The 15-class classifier predicts the NEXT window's label given a 10-window history.
# For X = W0..W9, the next label is W10, which is yl[:, 0].
def make_clf_dataset(X, yl):
    labels = np.array([label_to_idx[l[0]] for l in yl])
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))

def make_fs_dataset(X, Y):
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))

clf_tr_loader = DataLoader(make_clf_dataset(X_tr, yl_tr), batch_size=BATCH_SIZE, shuffle=True)
clf_va_loader = DataLoader(make_clf_dataset(X_va, yl_va), batch_size=BATCH_SIZE, shuffle=False)

fs_tr_loader = DataLoader(make_fs_dataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)
fs_va_loader = DataLoader(make_fs_dataset(X_va, Y_va), batch_size=BATCH_SIZE, shuffle=False)
fs_te_loader = DataLoader(make_fs_dataset(X_te, Y_te), batch_size=BATCH_SIZE, shuffle=False)

# 3. Train Classifier
print("\n--- Training Attack Classifier (Protocol A safe split) ---")
clf = AttackClassifierLSTM().to(DEVICE)
opt_clf = optim.Adam(clf.parameters(), lr=LR)
crit_clf = nn.CrossEntropyLoss()

for ep in range(EPOCHS):
    clf.train()
    tr_loss = 0
    for bx, by in clf_tr_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt_clf.zero_grad()
        out = clf(bx)
        loss = crit_clf(out, by)
        loss.backward()
        opt_clf.step()
        tr_loss += loss.item()
    
    clf.eval()
    va_loss, va_acc, total = 0, 0, 0
    with torch.no_grad():
        for bx, by in clf_va_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = clf(bx)
            loss = crit_clf(out, by)
            va_loss += loss.item()
            va_acc += (out.argmax(dim=1) == by).sum().item()
            total += by.size(0)
    print(f"CLF Ep {ep+1}: Tr Loss={tr_loss/len(clf_tr_loader):.4f}, Va Loss={va_loss/len(clf_va_loader):.4f}, Va Acc={va_acc/total:.4f}")

torch.save(clf.state_dict(), os.path.join(OUTPUT_DIR, "attack_classifier_protocolA_v1.pt"))

# 4. Train Direct Multi-Horizon
print("\n--- Training Direct Multi-Horizon Model ---")
direct_mod = DirectFutureStateModel().to(DEVICE)
opt_dir = optim.Adam(direct_mod.parameters(), lr=LR)
crit_fs = nn.MSELoss()

for ep in range(EPOCHS):
    direct_mod.train()
    tr_loss = 0
    for bx, by in fs_tr_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt_dir.zero_grad()
        out = direct_mod(bx)
        loss = crit_fs(out, by)
        loss.backward()
        opt_dir.step()
        tr_loss += loss.item()
    
    direct_mod.eval()
    va_loss = 0
    with torch.no_grad():
        for bx, by in fs_va_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = direct_mod(bx)
            va_loss += crit_fs(out, by).item()
    print(f"DIR Ep {ep+1}: Tr Loss={tr_loss/len(fs_tr_loader):.4f}, Va Loss={va_loss/len(fs_va_loader):.4f}")

torch.save(direct_mod.state_dict(), os.path.join(OUTPUT_DIR, "state_transition_direct_v1.pt"))

# 5. Train Recursive LSTM
print("\n--- Training Recursive LSTM Model (Teacher Forcing) ---")
rec_mod = RecursiveFutureStateModel().to(DEVICE)
opt_rec = optim.Adam(rec_mod.parameters(), lr=LR)

for ep in range(EPOCHS):
    rec_mod.train()
    tr_loss = 0
    tf_ratio = max(0.0, 0.5 - (ep * 0.05)) # Decay from 0.5 to 0.0
    for bx, by in fs_tr_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt_rec.zero_grad()
        out = rec_mod(bx, future_targets=by, teacher_forcing_ratio=tf_ratio)
        loss = crit_fs(out, by)
        loss.backward()
        opt_rec.step()
        tr_loss += loss.item()
    
    rec_mod.eval()
    va_loss = 0
    with torch.no_grad():
        for bx, by in fs_va_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = rec_mod(bx, future_targets=None, teacher_forcing_ratio=0.0)
            va_loss += crit_fs(out, by).item()
    print(f"REC Ep {ep+1}: TF={tf_ratio:.2f}, Tr Loss={tr_loss/len(fs_tr_loader):.4f}, Va Loss={va_loss/len(fs_va_loader):.4f}")

torch.save(rec_mod.state_dict(), os.path.join(OUTPUT_DIR, "state_transition_lstm_v1.pt"))

# 6. Evaluation (Persistence, Direct, Recursive)
print("\n--- Phase 4 EVALUATION ---")
# Predict test states
test_X_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    direct_preds = direct_mod(test_X_t).cpu().numpy()
    rec_preds = rec_mod(test_X_t, future_targets=None, teacher_forcing_ratio=0.0).cpu().numpy()

# Persistence
# W_t+k = W_t (last window of X)
persist_preds = np.repeat(X_te[:, -1:, :], 10, axis=1)

np.savez_compressed(os.path.join(OUTPUT_DIR, "predicted_states_protocolA_direct_v1.npz"), preds=direct_preds, true=Y_te)
np.savez_compressed(os.path.join(OUTPUT_DIR, "predicted_states_protocolA_recursive_v1.npz"), preds=rec_preds)
np.savez_compressed(os.path.join(OUTPUT_DIR, "persistence_states_protocolA_v1.npz"), preds=persist_preds)

def calc_errors(preds, true):
    # preds, true: (N, 10, 71)
    results = {}
    for k in [1, 3, 5, 10]:
        idx = k - 1
        p = preds[:, idx, :]
        t = true[:, idx, :]
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t)**2))
        results[k] = {'MAE': mae, 'RMSE': rmse}
    return results

err_p = calc_errors(persist_preds, Y_te)
err_d = calc_errors(direct_preds, Y_te)
err_r = calc_errors(rec_preds, Y_te)

print("--- FUTURE STATE ERRORS (Normalized scale) ---")
for k in [1, 3, 5, 10]:
    print(f"Horizon K={k}:")
    print(f"  Persistence: MAE={err_p[k]['MAE']:.4f}, RMSE={err_p[k]['RMSE']:.4f}")
    print(f"  Direct:      MAE={err_d[k]['MAE']:.4f}, RMSE={err_d[k]['RMSE']:.4f}")
    print(f"  Recursive:   MAE={err_r[k]['MAE']:.4f}, RMSE={err_r[k]['RMSE']:.4f}")

# 7. Attack Forecasting Evaluation
# Correct rolling context as requested:
# W11_input = [W1..W10]
# W12_input = [W2..W10, pred_W11]
def evaluate_attack_forecasts(preds, name):
    print(f"\n--- ATTACK FORECASTS: {name} ---")
    clf.eval()
    
    # Store metrics per K
    for k in [1, 3, 5, 10]:
        idx = k - 1
        true_labels = np.array([label_to_idx[l[idx]] for l in yl_te])
        
        # Build context
        # If k=1, context = X
        # If k=2, context = X[1:], preds[0:1]
        k_contexts = []
        for i in range(len(X_te)):
            x_seq = X_te[i]
            if k == 1:
                ctx = x_seq
            else:
                ctx = np.concatenate([x_seq[k-1:], preds[i, :k-1, :]], axis=0)
            k_contexts.append(ctx)
            
        ctx_tensor = torch.tensor(np.array(k_contexts), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = clf(ctx_tensor)
            pred_class = out.argmax(dim=1).cpu().numpy()
            
        macro_f1 = f1_score(true_labels, pred_class, average='macro', zero_division=0)
        
        # Binary stats
        benign_id = label_to_idx['Benign']
        bin_true = (true_labels != benign_id).astype(int)
        bin_pred = (pred_class != benign_id).astype(int)
        
        attack_f1 = f1_score(bin_true, bin_pred, zero_division=0)
        prec = precision_score(bin_true, bin_pred, zero_division=0)
        rec = recall_score(bin_true, bin_pred, zero_division=0)
        
        # FPR = FP / (FP + TN)
        tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred, labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        
        print(f"K={k}: Macro F1={macro_f1:.4f}, Attack F1={attack_f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, FPR={fpr:.4f}")

evaluate_attack_forecasts(Y_te, "GROUND-TRUTH FUTURE STATES (Theoretical Max)")
evaluate_attack_forecasts(persist_preds, "PERSISTENCE STATES")
evaluate_attack_forecasts(direct_preds, "DIRECT MULTI-HORIZON STATES")
evaluate_attack_forecasts(rec_preds, "RECURSIVE LSTM STATES")

print("\nDONE.")
