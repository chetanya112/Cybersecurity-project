# CHRONEX — SIH26153 Complete Project Reference
## AI-Based Network Attack Forecasting from Network Traffic Data
### (Use this document to resume work in a new chat/account if needed)

---

## 1. THE OFFICIAL PROBLEM STATEMENT (verbatim requirements)

**PS ID:** SIH26153 | **Org:** NTRO | **Category:** Software | **Theme:** Blockchain & Cybersecurity

### Core ask
Build a **World Model** — not a static classifier — that learns network traffic dynamics and forecasts attacker progression *before* compromise completes.

Traditional classifiers treat each flow in isolation → binary benign/malicious label. This discards temporal/causal structure (port-probe sequence, SYN-before-ACK-flood pattern, recon timing before lateral movement). **The PS explicitly wants dynamics learning: P(S_t+1 | S_t) — given current state, the probability distribution over future states.**

### Six mandatory technical requirements (from the PS text, exact)

1. **Represent network state** as feature vectors or graphs.
2. **Learn state-transition dynamics** using LSTM, Transformer, GNN, or equivalent — over time-windowed observations.
3. **Forecast future states** and estimate **probability of attacker progression** — via **K-step forward rollout** (feed predictions back in, simulate K steps ahead).
4. **Map predicted behaviour to MITRE ATT&CK phases** (Reconnaissance, Initial Access, Lateral Movement, C2, Exfiltration).
5. **Explainability** — SHAP values or attention weights. "Black-box outputs without interpretability are not acceptable."
6. **Two levels of traffic features required together, not optional:**
   - Flow-level (NetFlow/IPFIX): IP/port pairs, TCP flags, bytes/packets per flow, duration, IAT stats
   - Packet-level (PCAP-derived): TTL variance, TCP window size, fragment flags, payload size distribution, port-scan signatures, retransmission counts
   - PS's own reasoning: *"flow-level captures aggregate behaviour (SYN flood); packet-level exposes timing/sequencing (slow recon scan designed to evade flow-based thresholds)"*

### Required deliverables for evaluation
- Source code (GitHub/Drive link)
- README with setup instructions
- **Architecture Document (max 2 pages)**
- **Demo Video (max 2 minutes)**
- **Technical Presentation (max 5 slides)** — *different from the 6-slide idea-round PPT already built*
- Demo interface: accepts **PCAP or CSV**, runs inference fully **offline**, shows infiltration probability timeline + flagged flows + attack stage annotations
- **Benchmark comparison**: LSTM/world-model vs. Logistic Regression baseline — F1, precision, recall, false positive rate — "demonstrating measurable improvement"

### Approved datasets (per PS)
CIC-IDS-2018 (what we're using) ✅, CIC-IDS-2017, UNSW-NB15, CTU-13, CICIoT2023, LANL Auth Dataset, DARPA IDS datasets. Also references MITRE ATT&CK, CAPEC, CVE/NVD as knowledge-base resources.

---

## 2. WHAT'S ACTUALLY BUILT RIGHT NOW (verified, not assumed)

### Data pipeline — COMPLETE, solid, bug-audited
- `clean_data.py`: raw CIC-IDS-2018 (10 days, ~16.6M flows) → cleaned, deduplicated, consistent 70 numeric columns across all files. Fixed real bugs: identity-column mismatch across files, embedded duplicate headers, infinite-value handling, timestamp parsing artifacts (1970 dates), globally-constant columns dropped.
- `build_sequences.py`: flows → 30-second time windows → network state vectors → sequences of 10 windows. **Fixed bug**: `attack_ratio` was leaking the ground-truth label into model features (computed directly from `Label` column) — removed. Final feature count: **71** (70 traffic features + `flow_count`).
- `combine_and_resplit.py`: fixes a structural bug where chronological (by-day) splitting caused entire attack classes to be missing from train/val/test entirely. Now does **class-stratified splitting** — every class guaranteed present in all three splits (hard-fails loudly if a class has <3 sequences, rather than silently breaking). Train-only scaling (StandardScaler fit on train, applied to val/test — no leakage).
- All 15 real attack classes present: Benign, DDOS-HOIC, DDoS-LOIC-HTTP, DoS-Hulk, Bot, Infilteration, SSH-Bruteforce, DoS-GoldenEye, FTP-BruteForce, DoS-SlowHTTPTest, DoS-Slowloris, DDOS-LOIC-UDP, Brute Force-Web, Brute Force-XSS, SQL Injection.

### Model — PARTIAL, working but scoped narrower than the PS asks
- Binary LSTM (Benign vs. Attack — **collapsed from 15 classes**, not multiclass). Class-weighted loss, early stopping on val macro-F1.
- **Real test results (not fabricated):** Accuracy 0.89, Macro F1 0.85. Attack class: precision 0.77, recall 0.76, F1 0.76. Confusion matrix shows 83/340 real attacks missed (~24% miss rate) — an honest number to present, not a good one to hide.
- SHAP (`GradientExplainer`) working — produces real per-prediction top-8 feature attributions.
- MITRE mapping = a **hardcoded heuristic lookup table** (13 feature names → stage guess), not a learned mapping. Code comments honestly acknowledge this as a stopgap.
- Single-step forecasting only (`W1...W10 → W11`). **No K-step rollout implemented yet.**
- No baseline Logistic Regression comparison run yet.
- No packet-level features (flow-level only).

### PPT / Presentation — done for idea round
- 6-slide idea-round deck built into the official SIH template, all real numbers, no fabricated stats, correct citations. (Separate from the mandatory 5-slide **Technical Presentation** deliverable still needed for final evaluation.)

---

## 3. GAPS AGAINST THE PS — PRIORITIZED

### 🔴 Critical (currently zero honest answer if judges ask)
1. **K-step forward rollout** — the PS's literal core deliverable ("world model," "roll out K steps ahead," "time-series probability score"). Currently only single-step prediction exists. **This is the single most important thing left to build.**
2. **Packet-level features** — explicitly required "in combination," not optional. Scope realistically: one representative day's PCAP, extract TTL variance/window size/fragment flags/retransmissions via Scapy or PyShark, demonstrate the *combination* — full multi-day PCAP parsing isn't necessary to satisfy this.

### 🟠 High priority
3. **Baseline Logistic Regression comparison** — fast to add (~1 hour, reuse existing `X_train`/`y_train`, flatten sequences). Explicitly required by the PS. Do this early — if LSTM doesn't clearly beat baseline, better to know now.
4. **Real (not heuristic) MITRE stage mapping** — even a simple learned classifier trained on attack-type → MITRE-phase lookup would be more defensible than the current hardcoded dictionary. At minimum, be ready to explain the current approach honestly as an interim step.

### 🟡 Medium priority
5. Demo interface must accept **PCAP or CSV** (currently CSV-only planned).
6. Multiclass model (or a credible path back to it) — binary was a reasonable scoping call given class scarcity, but be ready to defend it explicitly, not present it as if it were multiclass.
7. Consolidate the exploratory notebook into one clean, reproducible script (PS explicitly asks for "reproducible training configuration"). Currently has duplicate diagnostic cells, repeated SHAP logic across cells, mixed Colab/local environment paths.

### ✅ Already fixed / non-issues
- `attack_ratio` leakage — fixed
- Chronological split class-omission bug — fixed via stratified resplit
- Train-only scaling — correct, no leakage
- **Security note (already actioned by user):** a GitHub token was previously found hardcoded in the notebook and pushed to a public repo — flagged for immediate revocation.

---

## 4. RECOMMENDED BUILD ORDER (given limited time before deadline)

1. Baseline Logistic Regression + real benchmark numbers (fast, low-risk, mandatory)
2. K-step rollout on top of existing binary LSTM (highest PS-alignment impact, doesn't require retraining from scratch — feed predictions back as input)
3. Packet-level feature extraction on one representative day (PCAP via Scapy/PyShark) — demonstrates PS-required "combination" of both feature levels
4. Improve MITRE mapping (even a simple trained lookup beats hardcoded heuristic)
5. Clean up notebook into one reproducible script/pipeline
6. Build the mandatory 5-slide Technical Presentation + 2-page Architecture Document + 2-minute Demo Video
7. Demo interface: extend to accept PCAP, show infiltration probability timeline + flagged flows + MITRE stage annotations, run fully offline

---

## 5. IDEAS TO STAND OUT FROM ~500 TEAMS (optional, only after the above is solid)

These are genuine differentiators, not padding — each ties directly back to something the PS already values, so none of them are "extra fluff," they're deeper execution of what's already asked for.

### Strongest candidates (high impact, realistic effort)

**A. Visualize the rollout curve itself, not just a single number.**
The PS asks for a "time-series probability score" over K steps. Most teams will likely just show one probability number per prediction. Showing an actual **rising risk curve across the K-step rollout** (like a stock chart trending toward "Attack") is a natural, visually compelling way to demonstrate you built the *actual* forecasting mechanism the PS describes — not just detection. This should be a priority deliverable regardless, but presenting it well is a differentiator.

**B. Baseline vs. LSTM side-by-side, on the same live example.**
Instead of burying the benchmark numbers in a report, show it live in the demo: same input sequence, baseline says X, LSTM says Y with a rollout curve and SHAP explanation. Makes the "measurable improvement" claim tangible instead of a table judges skim past.

**C. Auto-generated incident report per flagged sequence.**
A one-click "explain this alert" output — SHAP top features + MITRE stage + confidence + rollout trend — assembled into a clean, short summary (PDF or on-screen). This is genuine "SOC decision support" (exactly the PS's own framing), demonstrates the explainability requirement isn't just a slide, and is highly demoable in 2 minutes.

**D. Full-capture risk timeline ("network health over time") mode.**
Run the model across an entire day's traffic and plot forecasted risk as a continuous timeline, with real attack windows overlaid for comparison (ground truth vs. predicted). This visually proves the temporal-forecasting concept at scale, not just on cherry-picked single sequences — a strong "put your money where your mouth is" demo moment.

### Good but lower priority (do only if time remains)
- Alert triage/prioritization view — rank multiple simultaneous flagged windows by severity so an analyst knows what to look at first (reinforces "decision support, not autonomous action" framing)
- Lightweight adversarial robustness check — show what happens to the prediction under small input perturbations, ties to PS's "generalize to unseen patterns" language
- Graph visualization of communicating hosts alongside the LSTM output — PS mentions GNNs as an alternative architecture; even a visual graph layer (not a full GNN model) adds interpretive value and shows awareness of that approach

### Explicitly avoid
- Claiming autonomous response/attack-blocking (PS and your own framing consistently position this as decision support)
- Fabricated accuracy/performance numbers for anything not actually run
- Overpromising real-time/streaming deployment when the current system is offline/historical (fine to name as roadmap, not as done)

---

## 6. KEY NUMBERS TO REMEMBER (all verified real, safe to quote)

- Dataset: CIC-IDS-2018, 10 days, ~16.6M raw flows → ~15.57M cleaned
- Features: 71 per window (70 traffic + flow_count), 30-second windows, 10-window sequences (5-minute context)
- Current binary model: Accuracy 0.89, Macro F1 0.85, Attack recall 0.76 (misses ~24% of real attacks — state this honestly)
- 15 real attack classes present in the full dataset (list above)

---

## 7. IF STARTING A NEW CHAT — WHAT TO PASTE FIRST

Paste this entire document. Then say: "Continuing CHRONEX SIH26153 work — [specific task, e.g. 'help me write the K-step rollout code']." That gives full context without needing to re-explain the project from scratch.
