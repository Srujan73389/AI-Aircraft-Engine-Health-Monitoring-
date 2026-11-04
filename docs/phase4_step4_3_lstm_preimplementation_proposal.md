# Phase 4 – Step 4.3: Sequence Model (LSTM) Pre-Implementation Proposal

> [!IMPORTANT]
> **PROPOSAL STATUS: PRE-IMPLEMENTATION PLANNING ONLY — NOT IMPLEMENTED**  
> In accordance with the master project roadmap and user instructions, **no implementation code has been written, no packages have been installed, no models have been trained, and no datasets have been modified**.  
> This proposal specifies the technical architecture, data compatibility, evaluation protocol, and decision criteria for evaluating recurrent sequence models (LSTM) against the approved Step 4.2 XGBoost benchmark.

---

## 1. Master Project Roadmap & Architectural Context

### 1.1 Master Roadmap Specification
The master project roadmap ([Aircraft_Engine_AI_Project_Step_by_Step_Roadmap.pdf](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI_Project_Step_by_Step_Roadmap.pdf), Page 4) explicitly defines the scope and objective of Step 4.3:

* **Roadmap Step 4.3:** *"Evaluate a sequence model such as LSTM only if the available data and project time justify it."*
* **Digital Twin Model Set Architectural Role:**
  * **XGBoost RUL model:** Core tabular regression model predicting RUL from engineered sensor features and operating conditions.
  * **LSTM sequence model:** Optional sequence-based RUL prediction; **must be compared against XGBoost before retaining**.

### 1.2 The Reference Benchmark to Beat (Approved Step 4.2 XGBoost Winners)
The newly trained and audited XGBoost models from Step 4.2 establish the performance target:

| Subset | Winning Step 4.2 Model | Feature Representation & Tier | Feature Dim | Val Capped RMSE | Official Test RMSE | Official Test NASA Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FD001** | `XGBoost_Tuned_SummaryStats` | SummaryStats — Tier 1 (Core 14) | 70 | **12.75** | **13.48** | **268.06** |
| **FD002** | `XGBoost_FlattenedWindow` | Flattened — Tier 3 (Extended 24) | 720 | **19.33** | **30.05** | **14,159.01** |
| **FD003** | `XGBoost_Tuned_SummaryStats` | SummaryStats — Tier 3 (Extended 24) | 120 | **12.79** | **14.25** | **312.21** |
| **FD004** | `XGBoost_FlattenedWindow` | Flattened — Tier 3 (Extended 24) | 720 | **15.87** | **29.73** | **6,939.94** |

---

## 2. Input Data Compatibility & Pipeline Verification

### 2.1 Native 3D Tensor Ingestion
While linear models (Ridge) and decision trees (Random Forest, XGBoost) required flattening or summary statistics, **LSTM models consume 3D sequence tensors natively**:

* **Input Tensor Shape:** $(N, W, K) = (N, 30, K)$
  * Sequence window: $W = 30$ operational cycles.
  * Feature channels: $K \in \{14, 17, 24\}$ depending on the feature tier.
* **Available Sequence Arrays:**
  * **FD001:** Train = $(14020, 30, 24)$ | Val = $(3711, 30, 24)$ | Test Final = $(100, 30, 24)$
  * **FD002:** Train = $(36857, 30, 24)$ | Val = $(9362, 30, 24)$ | Test Final = $(259, 30, 24)$
  * **FD003:** Train = $(17857, 30, 24)$ | Val = $(3963, 30, 24)$ | Test Final = $(100, 30, 24)$
  * **FD004:** Train = $(42725, 30, 24)$ | Val = $(11303, 30, 24)$ | Test Final = $(248, 30, 24)$

### 2.2 Data Integrity & Anti-Leakage Confirmation
1. **Engine-Independent Splits:** Adheres strictly to [dataset/processed/train_val_split_manifest.json](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/dataset/processed/train_val_split_manifest.json). No engine appears in both training and validation sets (567 training engines, 142 validation engines).
2. **MinMax Scaling Isolation:** Feature scaling was fit strictly on training engines during Phase 3 ([models/scalers/scaler_manifest.json](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/scalers/scaler_manifest.json)); validation and test data were transformed without updating scaler statistics.
3. **Piecewise Capped Training Targets:** Target labels are capped at 125 cycles: $y_{\text{train}} = \min(\text{raw\_RUL}, 125)$.
4. **Test Isolation:** Official test sequences (`X_test_final_*.npy`) and uncapped ground-truth targets (`y_test_final_*.npy`) will be evaluated strictly in inference mode after model selection.
5. **Short Engine Tracking:** The `is_padded` indicator is preserved for subgroup evaluation (6 engines in FD002, 11 in FD004).

---

## 3. Proposed LSTM Architecture & Hyperparameters

To prevent overparameterization while capturing temporal degradation patterns, we propose PyTorch as the deep learning framework.

### 3.1 Primary Candidate: Stacked 2-Layer LSTM (`Stacked_LSTM`)
```
Input: (Batch_Size, 30, K)
  │
  ▼
LSTM Layer 1: hidden_size = 64, batch_first = True, return_sequences = True
  │  Output: (Batch_Size, 30, 64)
  ▼
Dropout: p = 0.20
  │
  ▼
LSTM Layer 2: hidden_size = 32, batch_first = True, return_sequences = False
  │  Output: (Batch_Size, 32) [Final time-step hidden state h_29]
  ▼
Dropout: p = 0.20
  │
  ▼
Dense Layer 1: Linear(32, 16) + ReLU
  │
  ▼
Output Regressor: Linear(16, 1) (Continuous scalar RUL prediction)
```

### 3.2 Secondary Candidate: Compact 1-Layer Single LSTM (`Single_LSTM`)
* **LSTM Layer:** `nn.LSTM(input_size=K, hidden_size=48, batch_first=True)`
* **Dropout:** $p = 0.20$
* **Dense Regressor:** `nn.Linear(48, 16)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.Linear(16, 1)`
* *Scope & Purpose:* Evaluated across all four C-MAPSS subsets (FD001, FD002, FD003, FD004) as an architecture-depth comparison against the 2-layer stacked network. It is not restricted to single-condition regimes (FD001 and FD003), but tests across all four subsets whether a shallower recurrent representation prevents overfitting or improves training dynamics.

### 3.3 Training Hyperparameters & Pre-Implementation Dependency Findings

| Hyperparameter / Dependency | Specification | Technical Justification & Pre-Implementation Status |
| :--- | :--- | :--- |
| **Framework Dependency** | PyTorch (`torch`) | **Pre-Implementation Finding Only:** Pre-implementation PyPI resolution confirmed that a compatible binary wheel (`torch==2.15.0`) is available on PyPI for Python 3.15 Windows AMD64. **PyTorch is NOT installed yet.** Formal installation and import verification will be executed only after explicit user implementation approval. |
| **Loss Function** | Mean Squared Error (`MSELoss`) | Directly aligns network backpropagation with the target RMSE metric. |
| **Optimizer** | `Adam` (Initial $\text{lr} = 0.001$) | Adaptive learning rate per parameter; accelerates convergence on non-stationary recurrent dynamics. |
| **Weight Decay** | $1 \times 10^{-4}$ ($L_2$ regularization) | Constrains recurrent weight norms and prevents exploding hidden-state magnitudes. |
| **LR Scheduler** | `ReduceLROnPlateau(factor=0.5, patience=4)` | Halves learning rate when validation capped RMSE plateaus for 4 epochs. |
| **Batch Size** | 128 | Balances gradient noise with execution speed on CPU/GPU. |
| **Max Epochs** | 60 | Sufficient epochs for convergence under early stopping. |
| **Early Stopping** | Patience = 12 epochs | Evaluated strictly on **validation capped RMSE**; restores best model state (`best_epoch`). |
| **Random Seed** | 42 | Set across PyTorch (`torch.manual_seed(42)`), NumPy (`np.random.seed(42)`), and DataLoader workers. |

---

## 4. Fair Evaluation Protocol & Decoupled Metrics

To maintain methodological consistency with Steps 4.1 and 4.2:

1. **Training Partition:** Fitted strictly on training engines using capped targets ($y_{\text{train}} \le 125$).
2. **Validation Partition (Model Selection):**
   * Primary metric: Capped validation RMSE ($y_{\text{val\_capped}} \le 125$).
   * Early stopping evaluates `val_capped_rmse`.
   * Model selection is determined **exclusively by minimum `val_capped_rmse`**.
3. **Diagnostic Validation:** Evaluated against uncapped RUL ($y_{\text{val\_uncapped}}$) to monitor degradation trajectory tracking.
4. **Official Test Partition (Benchmark):**
   * Evaluated **only once** after model selection.
   * Scored against original uncapped NASA ground truth (`y_test_final`).
   * **Zero Clipping Rule:** Model predictions are NOT clipped or clamped to 125.
5. **Metrics Calculated:**
   * RMSE, MAE, $R^2$, and official NASA asymmetric score:
     $$d = \hat{y} - y, \quad s = \begin{cases} e^{-d/13}-1, & d < 0 \\ e^{d/10}-1, & d \ge 0 \end{cases}$$
   * Subgroups: Full cohort, unpadded test engines, and padded short test engines.

---

## 5. Experiment Scope (16 Proposed Configurations)

To avoid excessive training latency while exploring feature representation and depth:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Proposed LSTM Experiment Matrix                       │
├────┬────────────────────┬─────────────────────┬──────────────┬──────────────┤
│ #  │ Model Variant      │ Feature Tier        │ Input Dim    │ Recurrent    │
├────┼────────────────────┼─────────────────────┼──────────────┼──────────────┤
│ 1  │ Stacked_LSTM       │ Tier1_Core14        │ (N, 30, 14)  │ 64 -> 32     │
│ 2  │ Stacked_LSTM       │ Tier2_Operational17 │ (N, 30, 17)  │ 64 -> 32     │
│ 3  │ Stacked_LSTM       │ Tier3_Extended24    │ (N, 30, 24)  │ 64 -> 32     │
│ 4  │ Single_LSTM        │ Tier3_Extended24    │ (N, 30, 24)  │ 48 (1-layer) │
└────┴────────────────────┴─────────────────────┴──────────────┴──────────────┘
Total Experiments: 4 configurations × 4 C-MAPSS subsets = 16 runs.
```

*Comparison Structure:*
- Configs 1–3 evaluate the impact of sensor selection (Core 14) vs. operating settings (Tier 2) vs. full channels (Tier 3) within the primary stacked LSTM.
- Config 4 evaluates the single-layer architecture (`Single_LSTM`) across all four subsets (FD001–FD004) to provide a systematic architecture-depth comparison against the 2-layer stacked network.

---

## 6. Proposed Deliverables & Directory Layout

Upon receiving user approval, the following deliverables will be produced:

1. **`models/lstm/run_lstm_models.py`**
   * Dedicated PyTorch script for data loading, model definition, training loops, validation early stopping, and metric evaluation.
2. **`models/lstm/lstm_manifest.json`**
   * Machine-readable manifest logging network architectures, epoch checkpoints, loss curves, and evaluation metrics.
3. **`reports/models/lstm_model_performance.csv`**
   * Tabular performance table recording all 16 configurations with decoupled validation and test metrics.
4. **`reports/models/lstm_per_engine_validation.csv`**
   * Engine-level validation diagnostic table across all 142 validation engines.
5. **`docs/phase4_step4_3_lstm_evaluation_report.md`**
   * Comprehensive evaluation report presenting head-to-head comparisons against the approved Step 4.2 XGBoost benchmark.
6. **`docs/phase4_step4_3_lstm_methodological_audit.md`**
   * Audit report verifying test isolation, convergence stability, and score additivity.

---

## 7. Strategic Recommendations & Core Questions

### Q1: Is LSTM technically suitable for our current sequence data?
* **Answer: YES.** The Phase 3 pipeline was designed around 3D sliding-window sequences of shape $(N, 30, 24)$ with stride $s=1$. Recurrent neural networks are naturally suited to ingest these sequences directly without the lossy statistical compression (SummaryStats) or dimensional expansion (FlattenedWindow) required by tree models.

### Q2: Can it be compared fairly against XGBoost?
* **Answer: YES.** Because both models ingest data derived from the identical Phase 3 scaling pipeline, the identical engine-level splits, identical piece-wise capped training targets ($y \le 125$), and are selected using the identical validation capped RMSE criterion, the comparison is completely direct, fair, and controlled.

### Q3: What are the main risks?
1. **Overparameterization & Overfitting:** For smaller subsets (FD001 and FD003, with 80 training engines), recurrent networks can easily overfit early run-to-failure cycles. This is mitigated by structural dropout ($0.20$), weight decay ($10^{-4}$), and early stopping.
2. **Computational Training Latency:** Unlike XGBoost which trains in seconds, recurrent networks on CPU require forward and backward passes across millions of sequence steps. Setting batch size to 128 and limiting maximum epochs to 60 with early stopping keeps training duration manageable.
3. **Multi-Condition Generalization:** In FD002 and FD004, operating condition transitions across 6 regimes can cause gradient turbulence. The learning rate scheduler (`ReduceLROnPlateau`) helps stabilize convergence.

### Q4: What configuration do you recommend?
* **Recommendation:** The **2-layer Stacked LSTM** (`nn.LSTM(64) -> nn.LSTM(32) -> Dense(16) -> Dense(1)`) with Adam optimizer, batch size 128, and early stopping on validation capped RMSE.

### Q5: What criteria should determine whether LSTM is retained in the final project?
Per the master project roadmap directive (*"Optional sequence-based RUL prediction; compare against XGBoost before retaining"*), model selection and architecture retention adhere strictly to the following 6-point protocol:

1. **Exclusive Validation Selection:** The winning LSTM configuration for each C-MAPSS subset (FD001, FD002, FD003, FD004) is selected **exclusively using validation capped RMSE** ($y \le 125$).
2. **Fair Validation Benchmark Comparison:** The selected winning LSTM model is compared directly with the approved Step 4.2 XGBoost winner for that subset **using validation capped RMSE**.
3. **Strict Test Set Prohibition:** Official test RMSE, test MAE, and test NASA asymmetric score are **never used to select, tune, or retain an architecture**.
4. **Isolated Single-Pass Test Evaluation:** Only after the winning LSTM configuration is locked and frozen will it be evaluated once on the official NASA test set.
5. **Independent Final Test Reporting:** Official test RMSE, MAE, and NASA asymmetric score will be reported for both LSTM and XGBoost as independent final comparison results in the project reports and manifests.
6. **Retention vs. Experimental Benchmark Decision:** If LSTM does not demonstrate a meaningful validation improvement over XGBoost on validation capped RMSE, **XGBoost will be retained as the primary production model** in the Digital Twin suite, and LSTM will be documented as an experimental sequential research benchmark.
