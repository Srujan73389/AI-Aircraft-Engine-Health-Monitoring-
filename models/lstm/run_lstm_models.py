"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 4: Machine Learning and Model Evaluation
Step 4.3: Train and Evaluate Sequence Models (LSTM)
=============================================================================

This script implements and quantitatively evaluates recurrent sequence models
(LSTM) across all four NASA C-MAPSS subsets (FD001, FD002, FD003, FD004) and
across three feature tiers (Tier 1 Core 14, Tier 2 Operational 17, Tier 3 Extended 24).

Total Configurations: Exactly 16 distinct model configurations
(4 per subset x 4 subsets):
  1. Stacked_LSTM - Tier 1 Core 14 (Input: 14 channels, 64 -> 32 units)
  2. Stacked_LSTM - Tier 2 Operational 17 (Input: 17 channels, 64 -> 32 units)
  3. Stacked_LSTM - Tier 3 Extended 24 (Input: 24 channels, 64 -> 32 units)
  4. Single_LSTM  - Tier 3 Extended 24 (Input: 24 channels, 48 units, 1-layer)

Strict Protocol:
  - Training target: Capped at 125 cycles: y_train = min(raw_RUL, 125)
  - Primary validation: Capped target: y_val_capped = min(raw_RUL, 125)
  - Additional diagnostic validation: Uncapped target: y_val_uncapped = raw_RUL
  - Official test benchmark: Uncapped NASA ground-truth RUL (y_test_final)
  - Raw model predictions are evaluated (never automatically clipped to 125)
  - Per-engine validation metrics aggregated across all 142 validation engines
  - Subgroup breakdown: Unpadded (690 engines) vs Padded (17 engines)
  - Exclusive winner selection criterion: Minimum validation capped RMSE
  - Zero test-set leakage: Test data is completely isolated until after selection
=============================================================================
"""

import os
import sys
import copy
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.scaling_pipeline import (
    TIER1_CORE_SENSORS,
    TIER2_MULTI_CONDITION,
    TIER3_EXTENDED_ALL,
)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
RUL_CAP = 125.0
RANDOM_STATE = 42

# Column index mappings from TIER3_EXTENDED_ALL
TIER_FEATURE_MAP = {
    "Tier1_Core14": {
        "features": TIER1_CORE_SENSORS,
        "indices": [TIER3_EXTENDED_ALL.index(c) for c in TIER1_CORE_SENSORS],
        "dim": 14,
    },
    "Tier2_Operational17": {
        "features": TIER2_MULTI_CONDITION,
        "indices": [TIER3_EXTENDED_ALL.index(c) for c in TIER2_MULTI_CONDITION],
        "dim": 17,
    },
    "Tier3_Extended24": {
        "features": TIER3_EXTENDED_ALL,
        "indices": [TIER3_EXTENDED_ALL.index(c) for c in TIER3_EXTENDED_ALL],
        "dim": 24,
    },
}


def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

class StackedLSTMRegressor(nn.Module):
    """
    2-Layer Stacked LSTM Regressor:
      - LSTM 1: input_dim -> hidden_dim1 (returns sequences)
      - Dropout(0.20)
      - LSTM 2: hidden_dim1 -> hidden_dim2 (returns last hidden state)
      - Dropout(0.20)
      - Linear(hidden_dim2, 16) -> ReLU()
      - Linear(16, 1)
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.20):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_dim1, hidden_size=hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim2, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, input_dim)
        out1, _ = self.lstm1(x)                # (B, 30, 64)
        out1 = self.dropout1(out1)
        out2, (hn, _) = self.lstm2(out1)        # out2: (B, 30, 32), hn: (1, B, 32)
        last_hidden = out2[:, -1, :]           # (B, 32)
        last_hidden = self.dropout2(last_hidden)
        h = self.relu(self.fc1(last_hidden))   # (B, 16)
        pred = self.out(h).squeeze(-1)         # (B,)
        return pred


class SingleLSTMRegressor(nn.Module):
    """
    1-Layer Single LSTM Regressor:
      - LSTM: input_dim -> hidden_dim (returns last hidden state)
      - Dropout(0.20)
      - Linear(hidden_dim, 16) -> ReLU()
      - Linear(16, 1)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 48, dropout: float = 0.20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, input_dim)
        out, _ = self.lstm(x)                  # (B, 30, 48)
        last_hidden = out[:, -1, :]            # (B, 48)
        last_hidden = self.dropout(last_hidden)
        h = self.relu(self.fc1(last_hidden))   # (B, 16)
        pred = self.out(h).squeeze(-1)         # (B,)
        return pred


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute official NASA asymmetric scoring function with overflow protection."""
    d = y_pred - y_true
    d_clipped = np.clip(d, -500.0, 500.0)
    s = np.where(d_clipped < 0, np.exp(-d_clipped / 13.0) - 1.0, np.exp(d_clipped / 10.0) - 1.0)
    return float(np.sum(s))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics (RMSE, MAE, R2, NASA Score)."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    nasa_score = compute_nasa_score(y_true, y_pred)
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "score": round(nasa_score, 2),
    }


# =============================================================================
# MODEL TRAINING & INFERENCE PIPELINE
# =============================================================================

def train_lstm_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_val_t: torch.Tensor,
    y_val_capped: np.ndarray,
    max_epochs: int = 60,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 12,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, dict]:
    """
    Train an LSTM model with MSE loss, Adam optimizer, ReduceLROnPlateau,
    and validation-capped-RMSE early stopping.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_capped_rmse": [],
        "lr": [],
    }

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(batch_losses))

        # Evaluate on validation set using capped targets
        model.eval()
        with torch.no_grad():
            val_preds_t = model(X_val_t.to(device))
            val_preds = val_preds_t.cpu().numpy()

        val_err = val_preds - y_val_capped
        val_rmse = float(np.sqrt(np.mean(val_err ** 2)))

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_capped_rmse"].append(round(val_rmse, 4))
        history["lr"].append(current_lr)

        # Learning rate schedule step
        scheduler.step(val_rmse)

        # Early stopping check on validation capped RMSE
        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    training_time = time.time() - t0

    # Restore best model state
    model.load_state_dict(best_model_weights)
    model.eval()

    train_summary = {
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "best_val_capped_rmse": round(best_val_rmse, 4),
        "training_time_sec": round(training_time, 2),
        "history": history,
    }
    return model, train_summary


def predict_in_batches(
    model: nn.Module,
    X_tensor: torch.Tensor,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Predict continuously across large tensors in batches to conserve memory."""
    model.eval()
    model.to(device)
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size].to(device)
            p = model(batch).cpu().numpy()
            preds_list.append(p)
    return np.concatenate(preds_list, axis=0)


# =============================================================================
# SYNTHETIC SANITY & SHAPE CHECK FUNCTION
# =============================================================================

def run_synthetic_pipeline_check() -> bool:
    """
    Execute syntax, data-loading, architecture shape tests, and a small
    forward-backward pass on synthetic tensors.
    """
    print("=" * 80)
    print("LSTM PIPELINE SANITY & SYNTHETIC FORWARD/BACKWARD CHECK")
    print("=" * 80)

    device = torch.device("cpu")
    set_seed(RANDOM_STATE)

    # 1. Test Stacked_LSTM shapes across Tiers 1-3
    for tier, info in TIER_FEATURE_MAP.items():
        k = info["dim"]
        dummy_x = torch.randn(8, 30, k, dtype=torch.float32)
        model = StackedLSTMRegressor(input_dim=k, hidden_dim1=64, hidden_dim2=32)
        out = model(dummy_x)
        assert out.shape == (8,), f"Shape mismatch in StackedLSTM for {tier}: got {out.shape}"
        print(f"[SHAPE CHECK] Stacked_LSTM ({tier:20s}, Dim={k:2d}): Input (8, 30, {k:2d}) -> Output {tuple(out.shape)} [OK]")

    # 2. Test Single_LSTM shapes on Tier 3
    k3 = TIER_FEATURE_MAP["Tier3_Extended24"]["dim"]
    dummy_x3 = torch.randn(8, 30, k3, dtype=torch.float32)
    model_single = SingleLSTMRegressor(input_dim=k3, hidden_dim=48)
    out_single = model_single(dummy_x3)
    assert out_single.shape == (8,), f"Shape mismatch in SingleLSTM: got {out_single.shape}"
    print(f"[SHAPE CHECK] Single_LSTM  ({'Tier3_Extended24':20s}, Dim={k3:2d}): Input (8, 30, {k3:2d}) -> Output {tuple(out_single.shape)} [OK]")

    # 3. Small synthetic forward-backward optimization check
    print("\n[OPTIMIZATION CHECK] Testing synthetic training step (Loss backprop & optimizer step)...")
    dummy_y = torch.tensor([125.0, 100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0], dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    preds = model(dummy_x)
    loss = criterion(preds, dummy_y)
    loss.backward()
    optimizer.step()
    print(f"  - Initial synthetic MSE Loss: {loss.item():.4f}")
    print("  - Backward pass and optimizer step completed with zero errors. [OK]")

    # 4. Data existence and tensor loading verification
    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "processed", "sequences")
    expected_engine_counts = {
        "FD001": {"train": 80, "val": 20, "test": 100},
        "FD002": {"train": 208, "val": 52, "test": 259},
        "FD003": {"train": 80, "val": 20, "test": 100},
        "FD004": {"train": 199, "val": 50, "test": 248},
    }

    print("\n[DATA VERIFICATION] Verifying Phase 3 sequence arrays & engine counts on disk:")
    for subset in SUBSETS:
        f_tr = os.path.join(seq_dir, f"X_train_{subset}.npy")
        f_val = os.path.join(seq_dir, f"X_validation_{subset}.npy")
        f_te = os.path.join(seq_dir, f"X_test_final_{subset}.npy")
        f_tr_meta = os.path.join(seq_dir, f"train_metadata_{subset}.csv")
        f_val_meta = os.path.join(seq_dir, f"validation_metadata_{subset}.csv")
        f_te_meta = os.path.join(seq_dir, f"test_final_metadata_{subset}.csv")

        assert os.path.exists(f_tr), f"Missing {f_tr}"
        assert os.path.exists(f_val), f"Missing {f_val}"
        assert os.path.exists(f_te), f"Missing {f_te}"
        assert os.path.exists(f_tr_meta), f"Missing {f_tr_meta}"
        assert os.path.exists(f_val_meta), f"Missing {f_val_meta}"
        assert os.path.exists(f_te_meta), f"Missing {f_te_meta}"

        sh_tr = np.load(f_tr, mmap_mode="r").shape
        sh_val = np.load(f_val, mmap_mode="r").shape
        sh_te = np.load(f_te, mmap_mode="r").shape

        df_tr_m = pd.read_csv(f_tr_meta)
        df_va_m = pd.read_csv(f_val_meta)
        df_te_m = pd.read_csv(f_te_meta)

        n_tr_units = len(df_tr_m["unit_number"].unique())
        n_va_units = len(df_va_m["unit_number"].unique())
        n_te_units = len(df_te_m["unit_number"].unique())

        exp = expected_engine_counts[subset]
        assert n_tr_units == exp["train"], f"Mismatch train engines {subset}: {n_tr_units} vs {exp['train']}"
        assert n_va_units == exp["val"], f"Mismatch val engines {subset}: {n_va_units} vs {exp['val']}"
        assert n_te_units == exp["test"], f"Mismatch test engines {subset}: {n_te_units} vs {exp['test']}"

        print(
            f"  - {subset}: Train={sh_tr} ({n_tr_units} eng), "
            f"Val={sh_val} ({n_va_units} eng), "
            f"TestFinal={sh_te} ({n_te_units} eng) [OK]"
        )

    print("\n" + "=" * 80)
    print("ALL PRE-EXECUTION PIPELINE CHECKS PASSED SUCCESSFULLY")
    print("=" * 80)
    return True


# =============================================================================
# MANIFEST METADATA GENERATORS
# =============================================================================

def build_project_configuration() -> dict:
    """Build standardized project configuration metadata for Step 4.6 compliance."""
    return {
        "global_random_seed": 42,
        "target_definition": {
            "target_name": "Remaining Useful Life (RUL)",
            "target_unit": "operational_cycles",
            "training_target_capping": 125.0,
            "primary_validation_capping": 125.0,
            "diagnostic_validation_target": "uncapped",
            "official_test_target": "uncapped",
            "prediction_policy": "raw_unclipped",
            "scoring_error_clipping": [-500.0, 500.0],
            "scoring_error_clipping_purpose": "numerical_protection_against_overflow",
        },
        "temporal_window": {
            "window_size": 30,
            "stride": 1,
            "short_test_engine_padding": "left-edge replication of initial observation (cycle t=1 repeated backwards)",
        },
        "feature_tiers": {
            "Tier1_Core14": list(TIER1_CORE_SENSORS),
            "Tier2_Operational17": list(TIER2_MULTI_CONDITION),
            "Tier3_Extended24": list(TIER3_EXTENDED_ALL),
        },
        "summary_statistics_definition": {
            "window_length": 30,
            "statistics": ["mean", "std", "min", "max", "ols_linear_trend_slope"],
            "formula_notes": "ols_linear_trend_slope computed over discrete time index t in [0, 29] via closed-form OLS formula: sum((t - t_mean)*(x - x_mean)) / sum((t - t_mean)**2)",
        },
    }


def get_environment_metadata() -> dict:
    """Dynamically extract environment and package versions for Step 4.6 compliance."""
    import platform
    import scipy
    import sklearn
    import xgboost
    import matplotlib

    return {
        "python_version": platform.python_version(),
        "python_build": platform.python_build()[0] + f" ({platform.python_build()[1]}) [{platform.python_compiler()}]",
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "recorded_for": "NASA C-MAPSS Phase 4 Roadmap Step 4.6 Compliance",
    }


# =============================================================================
# FULL 16-EXPERIMENT TRAINING RUNNER (READY FOR EXECUTION UPON APPROVAL)
# =============================================================================

def run_all_lstm_experiments():
    """
    Execute all 16 LSTM configurations across FD001–FD004.
    Will only be executed upon explicit user approval.
    """
    print("=" * 80)
    print("PHASE 4 - STEP 4.3: TRAIN AND EVALUATE LSTM SEQUENCE MODELS")
    print(f"Configurations: Exactly 16 (4 per subset x 4 subsets) | Random State: {RANDOM_STATE}")
    print("=" * 80)

    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "processed", "sequences")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "models")
    lstm_dir = os.path.join(PROJECT_ROOT, "models", "lstm")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(lstm_dir, exist_ok=True)

    device = torch.device("cpu")
    performance_records = []
    per_engine_records = []
    manifest_records = {}

    config_idx = 0
    start_total_time = time.time()

    for subset in SUBSETS:
        print("\n" + "#" * 80)
        print(f"SUBSET: {subset}")
        print("#" * 80)

        # 1. Load sequence arrays and metadata
        X_train_raw = np.load(os.path.join(seq_dir, f"X_train_{subset}.npy"))
        y_train_raw = np.load(os.path.join(seq_dir, f"y_train_{subset}.npy"))

        X_val_raw = np.load(os.path.join(seq_dir, f"X_validation_{subset}.npy"))
        y_val_raw = np.load(os.path.join(seq_dir, f"y_validation_{subset}.npy"))
        df_val_meta = pd.read_csv(os.path.join(seq_dir, f"validation_metadata_{subset}.csv"))

        X_test_final_raw = np.load(os.path.join(seq_dir, f"X_test_final_{subset}.npy"))
        y_test_final_raw = np.load(os.path.join(seq_dir, f"y_test_final_{subset}.npy"))
        df_test_final_meta = pd.read_csv(os.path.join(seq_dir, f"test_final_metadata_{subset}.csv"))

        # Target definitions
        y_train_capped = np.minimum(y_train_raw, RUL_CAP).astype(np.float32)
        y_val_capped = np.minimum(y_val_raw, RUL_CAP).astype(np.float32)
        y_val_uncapped = y_val_raw.astype(np.float32)
        y_test_final_uncapped = y_test_final_raw.astype(np.float32)

        val_units = df_val_meta["unit_number"].unique()
        num_val_engines = len(val_units)
        is_padded_test = df_test_final_meta["is_padded"].values
        last_val_indices = df_val_meta.groupby("unit_number").tail(1).index.values

        print(f"[{subset}] Loaded: Train={X_train_raw.shape}, Val={X_val_raw.shape} ({num_val_engines} engines), Test={X_test_final_raw.shape}")

        # Define 4 configurations for this subset
        configs = [
            ("Stacked_LSTM", "Tier1_Core14", 14, lambda k: StackedLSTMRegressor(input_dim=k, hidden_dim1=64, hidden_dim2=32)),
            ("Stacked_LSTM", "Tier2_Operational17", 17, lambda k: StackedLSTMRegressor(input_dim=k, hidden_dim1=64, hidden_dim2=32)),
            ("Stacked_LSTM", "Tier3_Extended24", 24, lambda k: StackedLSTMRegressor(input_dim=k, hidden_dim1=64, hidden_dim2=32)),
            ("Single_LSTM",  "Tier3_Extended24", 24, lambda k: SingleLSTMRegressor(input_dim=k, hidden_dim=48)),
        ]

        for m_variant, tier, dim, model_fn in configs:
            config_idx += 1
            col_indices = TIER_FEATURE_MAP[tier]["indices"]

            print(f"[{config_idx:02d}/16] [{subset}] Training {m_variant} | {tier} (Dim={dim})...", end=" ", flush=True)

            set_seed(RANDOM_STATE)

            # Slice feature channels
            X_tr_sub = torch.tensor(X_train_raw[:, :, col_indices], dtype=torch.float32)
            y_tr_sub = torch.tensor(y_train_capped, dtype=torch.float32)

            X_v_sub = torch.tensor(X_val_raw[:, :, col_indices], dtype=torch.float32)
            X_te_sub = torch.tensor(X_test_final_raw[:, :, col_indices], dtype=torch.float32)

            train_dataset = TensorDataset(X_tr_sub, y_tr_sub)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

            model = model_fn(dim)
            model, summary = train_lstm_model(
                model=model,
                train_loader=train_loader,
                X_val_t=X_v_sub,
                y_val_capped=y_val_capped,
                max_epochs=60,
                lr=0.001,
                weight_decay=0.0001,
                patience=12,
                device=device,
            )

            # Predictions
            val_preds = predict_in_batches(model, X_v_sub, batch_size=256, device=device)
            test_preds = predict_in_batches(model, X_te_sub, batch_size=256, device=device)

            # 1. Primary Validation (Capped Targets)
            m_val_capped = evaluate_predictions(y_val_capped, val_preds)

            # 2. Diagnostic Validation (Uncapped Targets)
            m_val_uncapped = evaluate_predictions(y_val_uncapped, val_preds)

            # 3. Per-Engine Validation Reporting
            engine_rmses = []
            engine_maes = []
            for u in val_units:
                u_mask = (df_val_meta["unit_number"] == u).values
                u_err = val_preds[u_mask] - y_val_capped[u_mask]
                e_rmse = float(np.sqrt(np.mean(u_err ** 2)))
                e_mae = float(np.mean(np.abs(u_err)))
                engine_rmses.append(e_rmse)
                engine_maes.append(e_mae)

                per_engine_records.append({
                    "subset": subset,
                    "model_variant": m_variant,
                    "feature_tier": tier,
                    "unit_number": int(u),
                    "windows_count": int(np.sum(u_mask)),
                    "per_engine_rmse_capped": round(e_rmse, 4),
                    "per_engine_mae_capped": round(e_mae, 4),
                })

            mean_engine_rmse = round(float(np.mean(engine_rmses)), 4)
            std_engine_rmse = round(float(np.std(engine_rmses)), 4)

            # Validation final-cycle RMSE
            val_final_preds = val_preds[last_val_indices]
            val_final_true = y_val_capped[last_val_indices]
            val_final_cycle_rmse = round(float(np.sqrt(np.mean((val_final_preds - val_final_true) ** 2))), 4)

            # 4. Official Test Benchmark (Uncapped NASA Ground Truth)
            m_test_full = evaluate_predictions(y_test_final_uncapped, test_preds)

            # Test subgroups
            unpadded_mask = (~is_padded_test)
            padded_mask = is_padded_test

            if np.any(unpadded_mask):
                m_test_unpadded = evaluate_predictions(
                    y_test_final_uncapped[unpadded_mask], test_preds[unpadded_mask]
                )
                test_unpadded_rmse = m_test_unpadded["rmse"]
                test_unpadded_score = m_test_unpadded["score"]
            else:
                test_unpadded_rmse = None
                test_unpadded_score = None

            if np.any(padded_mask):
                m_test_padded = evaluate_predictions(
                    y_test_final_uncapped[padded_mask], test_preds[padded_mask]
                )
                test_padded_rmse = m_test_padded["rmse"]
                test_padded_score = m_test_padded["score"]
            else:
                test_padded_rmse = "N/A"
                test_padded_score = "N/A"

            elapsed = summary["training_time_sec"]
            best_ep = summary["best_epoch"]
            tot_ep = summary["total_epochs"]

            print(f"Val(Cap) RMSE={m_val_capped['rmse']:5.2f}, R2={m_val_capped['r2']:6.3f} (BestEp={best_ep}/{tot_ep}) | Test(Uncap) RMSE={m_test_full['rmse']:5.2f}, Score={m_test_full['score']:8.1f} ({elapsed:.1f}s)")

            # Master tabular record
            performance_records.append({
                "subset": subset,
                "model_family": "LSTM",
                "model_variant": m_variant,
                "feature_tier": tier,
                "feature_dim": dim,
                "val_capped_rmse": m_val_capped["rmse"],
                "val_capped_mae": m_val_capped["mae"],
                "val_capped_r2": m_val_capped["r2"],
                "val_capped_score": m_val_capped["score"],
                "val_uncapped_rmse": m_val_uncapped["rmse"],
                "val_uncapped_mae": m_val_uncapped["mae"],
                "val_uncapped_r2": m_val_uncapped["r2"],
                "val_uncapped_score": m_val_uncapped["score"],
                "val_engines_count": num_val_engines,
                "val_mean_engine_rmse": mean_engine_rmse,
                "val_std_engine_rmse": std_engine_rmse,
                "val_final_cycle_rmse": val_final_cycle_rmse,
                "test_uncapped_rmse": m_test_full["rmse"],
                "test_uncapped_mae": m_test_full["mae"],
                "test_uncapped_score": m_test_full["score"],
                "test_unpadded_rmse": test_unpadded_rmse,
                "test_unpadded_score": test_unpadded_score,
                "test_padded_rmse": test_padded_rmse,
                "test_padded_score": test_padded_score,
                "best_epoch": best_ep,
                "total_epochs": tot_ep,
                "training_time_sec": elapsed,
            })

            manifest_records[f"{subset}_{m_variant}_{tier}"] = {
                "subset": subset,
                "model_family": "LSTM",
                "model_variant": m_variant,
                "feature_tier": tier,
                "feature_dim": dim,
                "architecture_params": {
                    "model_class": m_variant,
                    "input_dim": dim,
                    "hidden_dim": 48 if m_variant == "Single_LSTM" else [64, 32],
                    "dropout": 0.20,
                    "batch_size": 128,
                    "max_epochs": 60,
                    "early_stopping_patience": 12,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "random_seed": RANDOM_STATE,
                },
                "training_summary": summary,
                "val_capped_metrics": m_val_capped,
                "val_uncapped_metrics": m_val_uncapped,
                "test_uncapped_metrics": m_test_full,
                "val_mean_engine_rmse": mean_engine_rmse,
                "val_final_cycle_rmse": val_final_cycle_rmse,
            }

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"ALL 16 LSTM CONFIGURATIONS COMPLETED IN {total_time:.1f} SECONDS")
    print("=" * 80)

    # Save master performance CSV
    df_perf = pd.DataFrame(performance_records)
    perf_csv_path = os.path.join(reports_dir, "lstm_model_performance.csv")
    df_perf.to_csv(perf_csv_path, index=False)
    print(f"[SAVE] Master LSTM performance table: {perf_csv_path}")

    # Save per-engine validation CSV
    df_engine = pd.DataFrame(per_engine_records)
    engine_csv_path = os.path.join(reports_dir, "lstm_per_engine_validation.csv")
    df_engine.to_csv(engine_csv_path, index=False)
    print(f"[SAVE] Per-engine validation metrics: {engine_csv_path}")

    # Save manifest JSON with standardized Step 4.6 metadata
    manifest_path = os.path.join(lstm_dir, "lstm_manifest.json")
    final_manifest = {
        "_project_configuration": build_project_configuration(),
        **manifest_records,
        "_environment_metadata": get_environment_metadata(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)
    print(f"[SAVE] LSTM execution manifest: {manifest_path}")

    # Display summary of best LSTM model per subset (selected strictly by Validation Capped RMSE)
    print("\n" + "=" * 80)
    print("TOP PERFORMING LSTM MODEL PER SUBSET (SELECTED BY VALIDATION CAPPED RMSE):")
    print("=" * 80)
    for subset in SUBSETS:
        sub_df = df_perf[df_perf["subset"] == subset]
        best_row = sub_df.sort_values("val_capped_rmse").iloc[0]
        print(f"  [{subset}] Best: {best_row['model_variant']} ({best_row['feature_tier']})")
        print(f"         Val Capped RMSE: {best_row['val_capped_rmse']:.4f} | Official Test RMSE: {best_row['test_uncapped_rmse']:.4f} | Test Score: {best_row['test_uncapped_score']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        run_all_lstm_experiments()
    else:
        run_synthetic_pipeline_check()
