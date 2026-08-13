"""
BHO + multi-seed training for the supervised D-prediction baseline.

This is the supervised ceiling: a plain MLP trained directly on
Crank-fitted log10(D) labels (descriptors -> log10(D)), with no
physics involved. It establishes the upper bound on what indirect,
physics-mediated D recovery (the PINN) could achieve, since this
baseline has direct access to the target it is trying to predict.

Same structure as the other BHO scripts for consistency:
  - BHO objective: mean validation MSE across 5-fold CV (by formulation).
  - Checkpoint selection (BHO and final): best validation MSE.
  - Final retraining: fixed 80/20 train/test split (by formulation),
    best config retrained with 5 seeds, checkpoint selected via an
    internal validation split carved from the training set.
    MSE and MALE reported on the held-out test set.

Outputs (in OUTPUT_DIR):
    study.pkl
    best_config.json
    bho_log.csv
    best_model/
        seed_0/  weights.pt, metrics.json, training_curve.npy
        ...
        summary.json    mean +/- std across seeds
"""

import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================
DATA_DIR   = "../release_dataset_with_Crank_release.xlsx"
OUTPUT_DIR = "descriptornet_supervised_bho"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- BHO ---
BHO_N_TRIALS = 30
BHO_EPOCHS   = 1500
BHO_K_FOLDS  = 5
BHO_SEED     = 42

# --- Final retraining ---
FINAL_EPOCHS          = 2500
FINAL_SEEDS           = [0, 1, 2, 3, 4]
TEST_FRACTION         = 0.2
INTERNAL_VAL_FRACTION = 0.15


# ============================================================
# DATA LOADING
# ============================================================
def load_data(path):
    """
    One row per formulation: descriptors -> Crank-fitted D.
    No release profiles needed — this is direct supervised regression.
    """
    df = pd.read_excel(path)
    df = df.drop(columns=['Time', 'Release', 'Crank_Release'])
    df = df.drop_duplicates(subset=['Formulation Index']).reset_index(drop=True)

    fid = df['Formulation Index'].values
    X   = df.drop(columns=['Formulation Index', 'Crank_D']).values.astype(float)
    D   = df['Crank_D'].values.astype(float)

    print(f"Formulations: {len(fid)} | Descriptor columns: {X.shape[1]}")
    print(f"D range: [{D.min():.3e}, {D.max():.3e}]")

    return X, D, fid


# ============================================================
# NETWORK
# ============================================================
def get_activation(name):
    return {"tanh": nn.Tanh(), "silu": nn.SiLU(), "gelu": nn.GELU()}[name.lower()]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, depth, width, activation):
        super().__init__()
        layers = [nn.Linear(in_dim, width), get_activation(activation)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), get_activation(activation)]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CrankDPredictor(nn.Module):
    """Descriptors -> log10(D). Direct supervised regression, no physics."""
    def __init__(self, n_desc, depth, width, activation):
        super().__init__()
        self.mlp = MLP(n_desc, 1, depth, width, activation)

    def forward(self, x):
        return self.mlp(x)   # (B, 1) — log10(D)


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def eval_mse_male(model, X_t, y_log_t):
    """
    X_t      : (N, n_desc) descriptors, scaled
    y_log_t  : (N, 1) true log10(D)
    Returns (MSE on log10(D), MALE) — for this baseline they coincide
    in spirit, but MSE is squared-error (used for selection) and MALE
    is mean absolute error (used for reporting/comparison with PINN).
    """
    model.eval()
    pred_log = model(X_t)
    mse  = torch.mean((pred_log - y_log_t) ** 2).item()
    male = torch.mean(torch.abs(pred_log - y_log_t)).item()
    if not np.isfinite(mse):
        mse = float("inf")
    if not np.isfinite(male):
        male = float("inf")
    return mse, male


# ============================================================
# TRAINING LOOP — selects checkpoint by best validation MSE
# ============================================================
def train_one(model, train_loader, X_val_t, y_val_log_t, epochs, lr,
              weight_decay, log_every=50, prefix=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_mse     = float("inf")
    best_weights = None
    curve        = []

    for epoch in range(epochs):
        model.train()
        epoch_mse = 0.0
        n_batches = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = torch.mean((pred - y_b) ** 2)
            if torch.isnan(loss):
                return None, float("inf")
            loss.backward()
            optimizer.step()
            epoch_mse += loss.item()
            n_batches += 1

        epoch_mse /= max(n_batches, 1)
        val_mse, val_male = eval_mse_male(model, X_val_t, y_val_log_t)

        if val_mse < best_mse:
            best_mse     = val_mse
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0 or epoch == epochs - 1:
            curve.append((epoch, epoch_mse, val_mse, val_male))
            print(f"  {prefix}epoch {epoch:4d} | "
                  f"train MSE {epoch_mse:.4e} | val MSE {val_mse:.4e} "
                  f"(best {best_mse:.4e}) | val MALE {val_male:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return curve, best_mse


# ============================================================
# DATALOADER HELPER
# ============================================================
def make_loader(X_scaled, y_log, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_log,    dtype=torch.float32).view(-1, 1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# MAIN
# ============================================================
def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | Output: {out.resolve()}")

    X, D, fid = load_data(DATA_DIR)
    n_desc    = X.shape[1]
    y_log     = np.log10(D)
    all_fids  = fid   # one row per formulation already

    # ============================================================
    # BHO — 5-fold CV objective on mean validation MSE
    # ============================================================
    print(f"\nBHO: {BHO_N_TRIALS} trials | {BHO_K_FOLDS}-fold CV | {BHO_EPOCHS} epochs/fold")
    csv_path   = out / "bho_log.csv"
    fieldnames = ["trial", "val_mse_mean", "val_mse_std", "n_params", "elapsed_s",
                  "depth", "width", "activation", "lr", "batch_size", "weight_decay"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def objective(trial):
            config = {
                "depth":        trial.suggest_categorical("depth",        [2, 3, 4, 5, 6]),
                "width":        trial.suggest_categorical("width",        [32, 64, 128, 256]),
                "activation":   trial.suggest_categorical("activation",   ["tanh", "silu", "gelu"]),
                "lr":           trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "batch_size":   trial.suggest_categorical("batch_size",   [16, 32, 64]),
                "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3, 1e-2]),
            }

            tmp = CrankDPredictor(n_desc, config["depth"], config["width"], config["activation"])
            n_params = sum(p.numel() for p in tmp.parameters())

            print(f"\n--- Trial {trial.number} | {config} | {n_params:,} params ---")
            t0 = time.time()

            kf = KFold(n_splits=BHO_K_FOLDS, shuffle=True, random_state=BHO_SEED)
            fold_mses = []

            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(all_fids)):
                X_tr_raw, X_val_raw = X[tr_idx], X[val_idx]
                y_tr_log,  y_val_log  = y_log[tr_idx], y_log[val_idx]

                scaler       = StandardScaler()
                X_tr_scaled  = scaler.fit_transform(X_tr_raw)
                X_val_scaled = scaler.transform(X_val_raw)

                train_loader = make_loader(X_tr_scaled, y_tr_log,
                                           config["batch_size"], shuffle=True)
                X_val_t      = torch.tensor(X_val_scaled, dtype=torch.float32, device=DEVICE)
                y_val_log_t  = torch.tensor(y_val_log, dtype=torch.float32,
                                            device=DEVICE).view(-1, 1)

                torch.manual_seed(BHO_SEED)
                model = CrankDPredictor(n_desc, config["depth"], config["width"],
                                        config["activation"]).to(DEVICE)

                _, best_mse = train_one(
                    model, train_loader, X_val_t, y_val_log_t,
                    BHO_EPOCHS, config["lr"], config["weight_decay"],
                    log_every=BHO_EPOCHS, prefix=f"[T{trial.number} F{fold_idx}] ",
                )
                fold_mses.append(best_mse)
                print(f"  [T{trial.number}] fold {fold_idx} best val MSE = {best_mse:.4e}")

            elapsed = time.time() - t0
            fold_mses_clean = [m if np.isfinite(m) else 10.0 for m in fold_mses]
            mean_mse = float(np.mean(fold_mses_clean))
            std_mse  = float(np.std(fold_mses_clean))

            print(f"  Trial {trial.number} | CV val MSE = {mean_mse:.4e} ± {std_mse:.4e}"
                  f"  ({elapsed:.0f}s)")

            writer.writerow({
                "trial": trial.number,
                "val_mse_mean": mean_mse,
                "val_mse_std":  std_mse,
                "n_params":     n_params,
                "elapsed_s":    round(elapsed, 1),
                **config,
            })
            return mean_mse

        study = optuna.create_study(
            direction="minimize",
            study_name="supervised_descriptornet",
            sampler=optuna.samplers.TPESampler(seed=BHO_SEED),
        )
        study.optimize(objective, n_trials=BHO_N_TRIALS, show_progress_bar=False)

    with open(out / "study.pkl", "wb") as f:
        pickle.dump(study, f)

    best_config = study.best_trial.params
    print(f"\nBHO done. Best trial {study.best_trial.number}:")
    print(f"  CV val MSE = {study.best_value:.4e}")
    print(f"  config     = {best_config}")
    with open(out / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # ============================================================
    # FINAL RETRAINING — fixed 80/20 split, 5 seeds
    # ============================================================
    print(f"\nFinal retraining: {len(FINAL_SEEDS)} seeds x {FINAL_EPOCHS} epochs "
          f"(fixed {1-TEST_FRACTION:.0%}/{TEST_FRACTION:.0%} split)")

    idx_train, idx_test = train_test_split(
        np.arange(len(all_fids)), test_size=TEST_FRACTION, random_state=BHO_SEED
    )

    seed_results = []

    for seed in FINAL_SEEDS:
        run_dir = out / "best_model" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Internal validation split (from training rows), seed-dependent
        rng = np.random.default_rng(seed)
        tr_idx_shuffled = idx_train.copy()
        rng.shuffle(tr_idx_shuffled)
        n_ival          = max(1, int(len(idx_train) * INTERNAL_VAL_FRACTION))
        ival_idx        = tr_idx_shuffled[:n_ival]
        actual_tr_idx   = tr_idx_shuffled[n_ival:]

        # Scaler fit on actual training rows only
        scaler             = StandardScaler()
        X_actual_tr_scaled = scaler.fit_transform(X[actual_tr_idx])
        X_ival_scaled       = scaler.transform(X[ival_idx])
        X_test_scaled       = scaler.transform(X[idx_test])

        train_loader = make_loader(X_actual_tr_scaled, y_log[actual_tr_idx],
                                   best_config["batch_size"], shuffle=True)
        X_ival_t     = torch.tensor(X_ival_scaled, dtype=torch.float32, device=DEVICE)
        y_ival_log_t = torch.tensor(y_log[ival_idx], dtype=torch.float32,
                                    device=DEVICE).view(-1, 1)
        X_test_t     = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
        y_test_log_t = torch.tensor(y_log[idx_test], dtype=torch.float32,
                                    device=DEVICE).view(-1, 1)

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = CrankDPredictor(n_desc, best_config["depth"], best_config["width"],
                                best_config["activation"]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Seed {seed} | {n_params:,} params")

        t0 = time.time()
        curve, _ = train_one(
            model, train_loader, X_ival_t, y_ival_log_t,
            FINAL_EPOCHS, best_config["lr"], best_config.get("weight_decay", 0.0),
            log_every=100, prefix=f"[S{seed}] ",
        )
        elapsed = time.time() - t0

        test_mse, test_male = eval_mse_male(model, X_test_t, y_test_log_t)
        print(f"  Seed {seed} | Test MSE={test_mse:.4e} | Test MALE={test_male:.4f}  "
              f"({elapsed:.0f}s)")

        torch.save(model.state_dict(), run_dir / "weights.pt")
        metrics = {
            "seed": seed, "test_MSE": test_mse, "test_MALE": test_male,
            "epochs": FINAL_EPOCHS, "elapsed_s": round(elapsed, 1),
            "config": best_config,
        }
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        if curve is not None:
            np.save(run_dir / "training_curve.npy", np.array(curve, dtype=object))
        seed_results.append({"seed": seed, "MSE": test_mse, "MALE": test_male})

    mses  = [r["MSE"]  for r in seed_results if np.isfinite(r["MSE"])]
    males = [r["MALE"] for r in seed_results if np.isfinite(r["MALE"])]
    summary = {
        "best_config":  best_config,
        "bho_cv_mse":   study.best_value,
        "seed_results": seed_results,
        "MSE_mean":     float(np.mean(mses)),
        "MSE_std":      float(np.std(mses)),
        "MALE_mean":    float(np.mean(males)),
        "MALE_std":     float(np.std(males)),
    }
    with open(out / "best_model" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Test MSE  (log10 D): {summary['MSE_mean']:.4e} +/- {summary['MSE_std']:.4e}")
    print(f"Test MALE (log10 D): {summary['MALE_mean']:.4f} +/- {summary['MALE_std']:.4f}")
    print(f"Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()