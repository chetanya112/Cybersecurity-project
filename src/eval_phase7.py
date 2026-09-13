import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import joblib

SRC_DIR  = Path('src').resolve()
ROOT     = SRC_DIR.parent
DATA_DIR = ROOT / "data" / "processed" / "model_ready_v2_primary"
MDL_DIR  = ROOT / "models" / "phase4"
OUT_DIR  = ROOT / "results" / "phase7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def build_context(x_seq, pred_seq, k):
    if k == 1:
        return x_seq
    else:
        history_len = 10 - (k - 1)
        return np.concatenate([x_seq[10 - history_len:], pred_seq[:k-1, :]], axis=0)

data = np.load(DATA_DIR / "dataset.npz", allow_pickle=True)
X_test = data['X_test']
ylab_test = data['ylab_test']

with open(DATA_DIR / "metadata.json") as f:
    meta = json.load(f)
label_classes = [l.replace("Infilteration", "Infiltration") for l in meta["label_classes"]]
benign_id = label_classes.index("Benign")

# Normalize ylab_test spelling
ylab_test_clean = np.empty_like(ylab_test, dtype=object)
for i in range(ylab_test.shape[0]):
    for j in range(ylab_test.shape[1]):
        ylab_test_clean[i, j] = ylab_test[i, j].replace("Infilteration", "Infiltration")
ylab_test = ylab_test_clean

clf_path = MDL_DIR / "attack_classifier_protocolA_v1.pt"
fs_path = MDL_DIR / "state_transition_direct_v1.pt"
clf = AttackClassifierLSTM().to('cuda' if torch.cuda.is_available() else 'cpu')
clf.load_state_dict(torch.load(clf_path, map_location=next(clf.parameters()).device))
clf.eval()

fs_model = DirectFutureStateModel().to(next(clf.parameters()).device)
fs_model.load_state_dict(torch.load(fs_path, map_location=next(fs_model.parameters()).device))
fs_model.eval()

device = next(clf.parameters()).device

# 1. Group into Contiguous Blocks
blocks = []
curr_block = [0]
for i in range(1, len(X_test)):
    if np.allclose(X_test[i][:-1], X_test[i-1][1:], atol=1e-4):
        curr_block.append(i)
    else:
        blocks.append(curr_block)
        curr_block = [i]
blocks.append(curr_block)
print(f"Total test sequences: {len(X_test)}, Contiguous Blocks: {len(blocks)}")

# 2. Extract Episodes & Chronological Labels
episodes = [] # list of dicts
ep_id = 0

for b_idx, block in enumerate(blocks):
    # W_t labels for t in [1 ... len(block)+10]
    # At block[i], index i predicts W_{i+1} ... W_{i+10}.
    # We construct the entire ground truth timeline for this block.
    # length of timeline = len(block) + 10
    timeline_labels = ["Benign"] * (len(block) + 10)
    for idx_in_block, seq_idx in enumerate(block):
        targets = ylab_test[seq_idx]
        for step in range(10):
            timeline_labels[idx_in_block + step + 1] = targets[step] # offset by 1 because seq 0 predicts W1..W10

    # Now find episodes in this timeline
    in_ep = False
    ep_start = -1
    ep_class = None
    for t in range(len(timeline_labels)):
        lbl = timeline_labels[t]
        if lbl != "Benign":
            if not in_ep:
                in_ep = True
                ep_start = t
                ep_class = lbl
        else:
            if in_ep:
                ep_end = t - 1
                episodes.append({
                    "ep_id": ep_id, "block_idx": b_idx,
                    "onset_t": ep_start, "end_t": ep_end, "attack_class": ep_class,
                    "already_active": False
                })
                ep_id += 1
                in_ep = False
    if in_ep:
        episodes.append({
            "ep_id": ep_id, "block_idx": b_idx,
            "onset_t": ep_start, "end_t": len(timeline_labels)-1, "attack_class": ep_class,
            "already_active": False
        })
        ep_id += 1

print(f"Total Attack Episodes Found: {len(episodes)}")

# Determine if already active in observed history
for ep in episodes:
    # Episode onset is at timeline step onset_t.
    # What was the observed history?
    # For a forecast generated at sequence i (where i maps to t=i in timeline), 
    # observed history covers t-9 to t.
    # If the episode onset_t is <= 0, then the attack was active before our timeline even started recording W1!
    if ep['onset_t'] <= 9: # meaning it started within the first 9 steps, so at seq 0, the history W(-9)...W(0) might contain it.
        # Actually, let's trace exactly:
        # At seq 0, observed history is W(-9) to W(0). We don't know their labels.
        # But if onset is W(1) to W(9), it means at some sequence before onset, the history will contain it.
        # Wait, the rule is: "At forecast time t inspect the actual labels of the OBSERVED 10-window history: y(t-9)...y(t). If any observed history window is non-Benign, mark as 'already active'."
        # This check is done AT FORECAST TIME.
        pass

# 3. Generate Predictions for all sequences
# We will precalculate all predictions to quickly scan
all_preds = []
test_tns = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    fs_preds_all = fs_model(test_tns).cpu().numpy()

results = []
for k in [1, 3, 5, 10]:
    contexts = []
    for i in range(len(X_test)):
        ctx = build_context(X_test[i], fs_preds_all[i], k)
        contexts.append(ctx)
    ctx_t = torch.tensor(np.array(contexts), dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(clf(ctx_t), dim=1).cpu().numpy()
    pred_classes = np.argmax(probs, axis=1)
    p_attacks = 1.0 - probs[:, benign_id]
    
    for i in range(len(X_test)):
        results.append({
            "seq_idx": i,
            "k": k,
            "pred_class": label_classes[pred_classes[i]],
            "p_attack": p_attacks[i]
        })

# 4. Evaluate Episodes
eligible_episodes = 0
already_active = 0

horizon_stats = {1: {'det': 0, 'bin_det': 0, 'ext_det': 0, 'lead_times': [], 'bin_lead_times': [], 'ext_lead_times': []},
                 3: {'det': 0, 'bin_det': 0, 'ext_det': 0, 'lead_times': [], 'bin_lead_times': [], 'ext_lead_times': []},
                 5: {'det': 0, 'bin_det': 0, 'ext_det': 0, 'lead_times': [], 'bin_lead_times': [], 'ext_lead_times': []},
                 10:{'det': 0, 'bin_det': 0, 'ext_det': 0, 'lead_times': [], 'bin_lead_times': [], 'ext_lead_times': []}}

thresh_stats = {0.50: {'det': 0, 'lead_times': []},
                0.70: {'det': 0, 'lead_times': []},
                0.80: {'det': 0, 'lead_times': []},
                0.90: {'det': 0, 'lead_times': []},
                0.95: {'det': 0, 'lead_times': []}}

ep_records = []

for ep in episodes:
    b_idx = ep['block_idx']
    block = blocks[b_idx]
    onset_t = ep['onset_t']
    
    # Check if attack is already active at the START of the block.
    # If the episode onset_t is <= 0 (which means it's W_0 or earlier), it's already active.
    if onset_t <= 0:
        ep['already_active'] = True
        already_active += 1
        continue
        
    # We can only forecast using sequences strictly BEFORE onset_t.
    # A sequence at index i_in_block represents forecast time t = i_in_block.
    # Sequence i_in_block observes W(i_in_block - 9) ... W(i_in_block).
    # Its target W(i_in_block + K) must lie inside the episode [onset_t, end_t].
    # AND the forecast time t (i_in_block) MUST BE < onset_t.
    # AND none of the observed history windows can be non-benign!
    # History windows for seq i_in_block are t_hist in [i_in_block-9 ... i_in_block].
    # Since episode started at onset_t, and i_in_block < onset_t, 
    # history windows are naturally < onset_t.
    # BUT what if there was a PRIOR episode in this block?
    # We must check if timeline_labels[t_hist] has any attack.
    
    # Find valid forecast sequences for this episode
    valid_forecasts_for_ep = False
    
    # We iterate i_in_block from 0 to onset_t - 1
    # We must find the earliest successful forecast.
    
    ep_bin_lead_times = {1: None, 3: None, 5: None, 10: None}
    ep_ext_lead_times = {1: None, 3: None, 5: None, 10: None}
    ep_thresh_lead_times = {0.5: None, 0.7: None, 0.8: None, 0.9: None, 0.95: None}
    
    # We check each sequence before onset
    for i_in_block in range(onset_t):
        seq_idx = block[i_in_block]
        
        # Check if history is clean
        # history spans timeline index max(0, i_in_block-9) to i_in_block.
        # Wait, timeline array starts at W_1 (index 1).
        # Actually timeline_labels[0] is dummy "Benign".
        # Let's check ylab_test of prior sequences if we want perfectly aligned labels.
        # It's easier to check timeline_labels from max(1, i_in_block-8) up to i_in_block.
        history_clean = True
        for h_t in range(max(1, i_in_block-8), i_in_block+1):
            if timeline_labels[h_t] != "Benign":
                history_clean = False
                break
        
        if not history_clean:
            continue # history contains attack, invalid forecast point
            
        valid_forecasts_for_ep = True
        
        # Filter results for this seq_idx
        seq_res = [r for r in results if r['seq_idx'] == seq_idx]
        
        for r in seq_res:
            k = r['k']
            pred_class = r['pred_class']
            p_attack = r['p_attack']
            
            target_t = i_in_block + k
            
            # Is target inside episode?
            in_episode = (onset_t <= target_t <= ep['end_t'])
            
            # Binary Detection
            if pred_class != "Benign" and in_episode:
                if ep_bin_lead_times[k] is None:
                    ep_bin_lead_times[k] = (onset_t - i_in_block) * 30
                    
            # Exact Class Detection
            if pred_class == ep['attack_class'] and in_episode:
                if ep_ext_lead_times[k] is None:
                    ep_ext_lead_times[k] = (onset_t - i_in_block) * 30
                    
            # Threshold detection
            # "A threshold crossing counts only when P(Attack) >= threshold AND forecast is associated with an upcoming actual attack episode AND t < actual_attack_onset"
            # Being "associated" means the target W(t+K) is inside the episode?
            # Or does it mean at ANY horizon it detects it? 
            # We'll use the maximum P(Attack) across all valid K for this sequence?
            # Or we check if THIS specific prediction's P(Attack) >= thresh AND target_t is in episode.
            for thresh in [0.50, 0.70, 0.80, 0.90, 0.95]:
                if p_attack >= thresh and in_episode:
                    if ep_thresh_lead_times[thresh] is None:
                        ep_thresh_lead_times[thresh] = (onset_t - i_in_block) * 30

    if not valid_forecasts_for_ep:
        # e.g., the attack started at onset_t=1, so i_in_block=0 history is checked, maybe it's valid.
        # if valid_forecasts_for_ep is still false, it means history wasn't clean or onset was 0.
        ep['already_active'] = True
        already_active += 1
        continue
        
    eligible_episodes += 1
    
    # Record stats
    for k in [1, 3, 5, 10]:
        if ep_bin_lead_times[k] is not None:
            horizon_stats[k]['bin_det'] += 1
            horizon_stats[k]['bin_lead_times'].append(ep_bin_lead_times[k])
        if ep_ext_lead_times[k] is not None:
            horizon_stats[k]['ext_det'] += 1
            horizon_stats[k]['ext_lead_times'].append(ep_ext_lead_times[k])
            
    for thresh in [0.50, 0.70, 0.80, 0.90, 0.95]:
        if ep_thresh_lead_times[thresh] is not None:
            thresh_stats[thresh]['det'] += 1
            thresh_stats[thresh]['lead_times'].append(ep_thresh_lead_times[thresh])

    # Save to record
    ep_records.append({
        'episode_id': ep['ep_id'],
        'attack_class': ep['attack_class'],
        'binary_lead_times': ep_bin_lead_times,
        'exact_lead_times': ep_ext_lead_times,
        'threshold_lead_times': ep_thresh_lead_times
    })
    
    # Just print the first one as requested example
    if eligible_episodes == 1:
        print("\n--- Example Complete Episode ---")
        print(f"Attack onset index: {onset_t}")
        print(f"Actual class: {ep['attack_class']}")
        print(f"Binary Lead Times: {ep_bin_lead_times}")
        print(f"Exact Lead Times: {ep_ext_lead_times}")

print("\n=== PHASE 7 VERIFICATION ===")
print(f"Episodes total: {len(episodes)}")
print(f"Episodes eligible: {eligible_episodes}")
print(f"Episodes already active/invalid: {already_active}")
print(f"Episodes detected (any K): {len([e for e in ep_records if any(v is not None for v in e['binary_lead_times'].values())])}")

for k in [1, 3, 5, 10]:
    dr = horizon_stats[k]['bin_det'] / eligible_episodes if eligible_episodes > 0 else 0
    lts = horizon_stats[k]['bin_lead_times']
    mean_lt = np.mean(lts) if lts else 0
    med_lt = np.median(lts) if lts else 0
    print(f"\nK={k}:")
    print(f"Detection rate: {dr:.2%}")
    print(f"Mean lead time: {mean_lt:.1f}s")
    print(f"Median lead time: {med_lt:.1f}s")
    
for thresh in [0.50, 0.70, 0.80, 0.90, 0.95]:
    dr = thresh_stats[thresh]['det'] / eligible_episodes if eligible_episodes > 0 else 0
    lts = thresh_stats[thresh]['lead_times']
    med_lt = np.median(lts) if lts else 0
    print(f"\nThreshold {thresh:.2f}:")
    print(f"Detection rate: {dr:.2%}")
    print(f"Median lead time: {med_lt:.1f}s")

# Write Report
with open(OUT_DIR / "forecast_lead_time_report.md", "w") as f:
    f.write("# Phase 7: Forecast Lead Time Report\n\n")
    f.write("## Methodology\n")
    f.write("Forecast lead time measures how early CHRONEX predicts an impending attack. \n")
    f.write("A forecast is strictly penalized: Ground truth future states are never passed to the model. ")
    f.write("Lead time is calculated as (Actual Attack Onset - Forecast Timestamp) * 30 seconds. ")
    f.write("Forecasts are valid only if they occur BEFORE the attack onset, and the target window falls inside the actual attack episode.\n\n")
    
    f.write(f"Total Episodes: {len(episodes)}\n")
    f.write(f"Eligible for Early Warning: {eligible_episodes}\n")
    f.write(f"Excluded (Attack already active): {already_active}\n\n")

    f.write("### TABLE 1 — HORIZON PERFORMANCE (Binary)\n")
    f.write("| Horizon | Forecast Seconds | Episodes Evaluated | Episodes Detected | Detection Rate | Mean Lead Time | Median Lead Time | Max Lead Time |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for k in [1,3,5,10]:
        det = horizon_stats[k]['bin_det']
        dr = det / eligible_episodes if eligible_episodes > 0 else 0
        lts = horizon_stats[k]['bin_lead_times']
        f.write(f"| K={k} | {k*30}s | {eligible_episodes} | {det} | {dr:.1%} | {np.mean(lts) if lts else 0:.1f}s | {np.median(lts) if lts else 0:.1f}s | {np.max(lts) if lts else 0:.1f}s |\n")

    f.write("\n### TABLE 2 — RISK THRESHOLD\n")
    f.write("| P(Attack) Threshold | Episodes Detected | Detection Rate | Mean Lead Time | Median Lead Time | Max Lead Time |\n")
    f.write("|---|---|---|---|---|---|\n")
    for thresh in [0.5, 0.7, 0.8, 0.9, 0.95]:
        det = thresh_stats[thresh]['det']
        dr = det / eligible_episodes if eligible_episodes > 0 else 0
        lts = thresh_stats[thresh]['lead_times']
        f.write(f"| {thresh:.2f} | {det} | {dr:.1%} | {np.mean(lts) if lts else 0:.1f}s | {np.median(lts) if lts else 0:.1f}s | {np.max(lts) if lts else 0:.1f}s |\n")

    f.write("\n### TABLE 3 — ATTACK-VS-BENIGN VS EXACT CLASS\n")
    f.write("| Horizon | Binary Attack Detection Rate | Exact Class Detection Rate | Mean Binary Lead Time | Mean Exact-Class Lead Time |\n")
    f.write("|---|---|---|---|---|\n")
    for k in [1,3,5,10]:
        b_det = horizon_stats[k]['bin_det']
        e_det = horizon_stats[k]['ext_det']
        b_dr = b_det / eligible_episodes if eligible_episodes > 0 else 0
        e_dr = e_det / eligible_episodes if eligible_episodes > 0 else 0
        b_lts = horizon_stats[k]['bin_lead_times']
        e_lts = horizon_stats[k]['ext_lead_times']
        f.write(f"| K={k} | {b_dr:.1%} | {e_dr:.1%} | {np.mean(b_lts) if b_lts else 0:.1f}s | {np.mean(e_lts) if e_lts else 0:.1f}s |\n")

with open(OUT_DIR / "forecast_lead_time_results.json", "w") as f:
    json.dump({'eligible_episodes': eligible_episodes, 'records': ep_records}, f, indent=2)

print("\nResults and reports saved to results/phase7/")
