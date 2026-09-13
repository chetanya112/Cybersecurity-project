import json
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import joblib

warnings.filterwarnings("ignore")

ROOT       = Path(".").resolve()
DATA_DIR_A = ROOT / "data" / "processed" / "model_ready_v2_primary"
DATA_DIR_B = ROOT / "data" / "processed" / "model_ready_v2_unseen"
MDL_DIR    = ROOT / "models" / "phase4"
OUT_DIR    = ROOT / "results" / "phase10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

class AttackClassifierLSTM(nn.Module):
    def __init__(self, input_dim=71, hidden_dim=64, num_classes=15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

with open(DATA_DIR_A / "metadata.json") as f:
    meta = json.load(f)
label_classes = [l.replace("Infilteration", "Infiltration") for l in meta["label_classes"]]
benign_id = label_classes.index("Benign")

device = "cuda" if torch.cuda.is_available() else "cpu"
clf = AttackClassifierLSTM().to(device)
clf.load_state_dict(torch.load(MDL_DIR / "attack_classifier_protocolA_v1.pt", map_location=device))
clf.eval()

scaler_A = joblib.load(DATA_DIR_A / "scaler.joblib")
BINS = np.linspace(0.0, 1.0, 11)

def evaluate_calibration(data_dir, name):
    data   = np.load(data_dir / "dataset.npz", allow_pickle=True)
    X_test = data["X_test"]
    ylab   = data["ylab_test"]

    if name == "Protocol B":
        scaler_B = joblib.load(DATA_DIR_B / "scaler.joblib")
        orig   = scaler_B.inverse_transform(X_test.reshape(-1, 71))
        X_test = scaler_A.transform(orig).reshape(X_test.shape).astype(np.float32)

    BATCH = 1024
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_test), BATCH):
            batch  = torch.tensor(X_test[start:start+BATCH], dtype=torch.float32).to(device)
            logits = clf(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    p_attack = 1.0 - all_probs[:, benign_id]

    # Ground truth accessed AFTER inference
    y_true_str = np.array([str(l).replace("Infilteration", "Infiltration") for l in ylab[:, 0]])
    y_true     = (y_true_str != "Benign").astype(float)
    N = len(p_attack)

    bin_results = []
    for i in range(10):
        lo, hi = BINS[i], BINS[i+1]
        mask   = (p_attack >= lo) & (p_attack <= hi) if i == 9 else (p_attack >= lo) & (p_attack < hi)
        count  = int(mask.sum())
        if count == 0:
            bin_results.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0,
                                 "mean_predicted": None, "observed_freq": None, "gap": None})
        else:
            mp  = float(p_attack[mask].mean())
            of  = float(y_true[mask].mean())
            gap = float(abs(mp - of))
            bin_results.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": count,
                                 "mean_predicted": mp, "observed_freq": of, "gap": gap})

    brier = float(np.mean((p_attack - y_true) ** 2))

    ece = 0.0
    for br in bin_results:
        if br["count"] > 0:
            ece += (br["count"] / N) * br["gap"]
    ece = float(ece)

    gaps = [br["gap"] for br in bin_results if br["gap"] is not None]
    mace = float(np.mean(gaps)) if gaps else 0.0

    pred_classes = np.argmax(all_probs, axis=1)
    pred_binary  = (pred_classes != benign_id).astype(int)
    gt_binary    = y_true.astype(int)
    correct_mask = (pred_binary == gt_binary)

    mean_p_correct   = float(p_attack[correct_mask].mean())  if correct_mask.sum()  > 0 else 0.0
    mean_p_incorrect = float(p_attack[~correct_mask].mean()) if (~correct_mask).sum() > 0 else 0.0

    high_conf_mask = (p_attack >= 0.90)
    hc_count       = int(high_conf_mask.sum())
    if hc_count > 0:
        total_actual = int(gt_binary.sum())
        hc_tp        = int(((gt_binary == 1) & high_conf_mask).sum())
        hc_precision = hc_tp / hc_count
        hc_recall    = hc_tp / total_actual if total_actual > 0 else 0.0
    else:
        hc_precision = hc_recall = 0.0

    return {
        "name": name, "n_samples": N,
        "brier": brier, "ece": ece, "mace": mace,
        "mean_p_correct": mean_p_correct, "mean_p_incorrect": mean_p_incorrect,
        "high_conf_count": hc_count,
        "high_conf_precision": float(hc_precision),
        "high_conf_recall": float(hc_recall),
        "bins": bin_results
    }

print("=== PHASE 10 VERIFICATION ===")
print("Future ground truth used during inference: FALSE\n")

res_a = evaluate_calibration(DATA_DIR_A, "Protocol A")
res_b = evaluate_calibration(DATA_DIR_B, "Protocol B")

for res in [res_a, res_b]:
    print(f"{res['name']}:")
    print(f"  Samples:          {res['n_samples']}")
    print(f"  Brier Score:      {res['brier']:.4f}")
    print(f"  ECE:              {res['ece']:.4f}")
    print(f"  MACE:             {res['mace']:.4f}")
    print(f"  P>=0.90 count:    {res['high_conf_count']}")
    print(f"  P>=0.90 precision:{res['high_conf_precision']:.4f}")
    print(f"  P>=0.90 recall:   {res['high_conf_recall']:.4f}")
    print(f"  Mean P correct:   {res['mean_p_correct']:.4f}")
    print(f"  Mean P incorrect: {res['mean_p_incorrect']:.4f}")
    print("  Sample calibration bins (non-empty):")
    for br in res["bins"]:
        if br["count"] > 0:
            print(f"    {br['bin']}  n={br['count']:5d}  pred={br['mean_predicted']:.3f}  obs={br['observed_freq']:.3f}  gap={br['gap']:.3f}")
    print()

with open(OUT_DIR / "calibration_results.json", "w") as f:
    json.dump({"Protocol A": res_a, "Protocol B": res_b}, f, indent=2)

def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

with open(OUT_DIR / "calibration_report.md", "w") as f:
    f.write("# Phase 10: Attack Probability Calibration & Reliability\n\n")
    f.write("## Definition\n")
    f.write("P(Attack) = 1 - P(Benign class), derived from the 15-class softmax of the production K=1 classifier.\n\n")
    f.write("Ground truth is used **only after** inference to compare predicted probabilities against observed attack frequencies. No calibration transform is fit on the test set.\n\n")
    f.write("## ECE Formula\n```\nECE = sum_b (N_b / N) * |mean_pred_b - obs_freq_b|    (10 fixed equal-width bins over [0,1])\n```\n\n")
    for res in [res_a, res_b]:
        f.write(f"## {res['name']}\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Samples | {res['n_samples']} |\n")
        f.write(f"| Brier Score | {fmt(res['brier'])} |\n")
        f.write(f"| ECE | {fmt(res['ece'])} |\n")
        f.write(f"| Mean Abs Calibration Error | {fmt(res['mace'])} |\n")
        f.write(f"| P>=0.90 count | {res['high_conf_count']} |\n")
        f.write(f"| P>=0.90 precision | {fmt(res['high_conf_precision'])} |\n")
        f.write(f"| P>=0.90 recall | {fmt(res['high_conf_recall'])} |\n")
        f.write(f"| Mean P(Attack) correct | {fmt(res['mean_p_correct'])} |\n")
        f.write(f"| Mean P(Attack) incorrect | {fmt(res['mean_p_incorrect'])} |\n\n")
        f.write("### Calibration Bins\n\n| Bin | Count | Mean Predicted | Observed Freq | Gap |\n|---|---|---|---|---|\n")
        for br in res["bins"]:
            f.write(f"| {br['bin']} | {br['count']} | {fmt(br['mean_predicted'],3)} | {fmt(br['observed_freq'],3)} | {fmt(br['gap'],3)} |\n")
        f.write("\n")
    f.write("## Interpretation\nP(Attack) is the model's estimated probability that the next network window is non-Benign. ")
    f.write("It does NOT mean the predicted attack class is correct with that probability.\n")

print(f"Saved to {OUT_DIR}")
