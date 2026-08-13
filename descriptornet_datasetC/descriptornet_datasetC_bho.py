"""
BHO + multi-seed training for DescriptorNet on Dataset B.

Simplified structure:
  - BHO objective: mean validation MSE (release) across 5-fold CV.
  - Checkpoint selection (BHO and final): best validation MSE.
  - Final retraining: fixed 80/20 train/test split (by formulation),
    best config retrained with 5 seeds. Checkpoint selected via an
    internal validation split carved from the training set (test set
    untouched during training). MSE and MALE reported on test set.

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
DATA_DIR     = "../plga_dataset/release_dataset_even.xlsx"
NET2_WEIGHTS = "../physicsnet/physicsnet_pretrained.pt"
NET2_CONFIG  = "../physicsnet/physicsnet_bho/best_config.json"
OUTPUT_DIR   = "descriptornet_datasetC_bho"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESCRIPTOR_COLS = [
    'Drug MW', 'Drug TPSA', 'Drug LogP', 'Polymer MW', 'LA/GA',
    'Initial Drug-to-Polymer Ratio', 'Particle Size',
    'Drug Loading Capacity', 'Drug Encapsulation Efficiency',
    'Solubility Enhancer Concentration'
]

LOG10_D_INIT = -10.0   # empirically found to work better than data mean

# --- BHO ---
BHO_N_TRIALS = 30
BHO_EPOCHS   = 200       # per fold per trial
BHO_K_FOLDS  = 5
BHO_SEED     = 42

# --- Final retraining ---
FINAL_EPOCHS          = 1000
FINAL_SEEDS           = [0, 1, 2, 3, 4]
TEST_FRACTION         = 0.2
INTERNAL_VAL_FRACTION = 0.15

# --- Fixed knobs ---
N_RHO_INTEGRATION = 30
GRAD_CLIP         = 1.0


# ============================================================
# DATA LOADING
# ============================================================
def load_data(path):
    df = pd.read_excel(path)
    df['t_s']  = df['Time'] * 86400.0
    df['R_cm'] = df['Particle Size'] / 2.0 / 1e4

    X = df[DESCRIPTOR_COLS].values.astype(float)
    t = df['t_s'].values.astype(float)
    R = df['R_cm'].values.astype(float)
    c = df['Release'].values.astype(float)

    df_part = (df[['Formulation Index', 'Crank_D', 'R_cm'] + DESCRIPTOR_COLS]
               .drop_duplicates(subset=['Formulation Index'])
               .reset_index(drop=True))
    X_part   = df_part[DESCRIPTOR_COLS].values.astype(float)
    D_part   = df_part['Crank_D'].values.astype(float) * 1e4   # -> cm^2/s
    fid      = df_part['Formulation Index'].values
    row_fids = df['Formulation Index'].values

    print(f"Rows: {len(df)} | Formulations: {len(fid)}")
    print(f"D range: [{D_part.min():.3e}, {D_part.max():.3e}] cm^2/s")
    print(f"R range: [{R.min():.3e}, {R.max():.3e}] cm")
    print(f"t range: [{t.min():.1f}, {t.max():.1f}] s")

    return X, t, R, c, X_part, D_part, fid, row_fids


# ============================================================
# NETWORKS
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


class DescriptorNet(nn.Module):
    def __init__(self, n_desc, depth, width, activation, log10_D_init):
        super().__init__()
        self.mlp = MLP(n_desc, 1, depth, width, activation)
        nn.init.constant_(self.mlp.net[-1].bias, log10_D_init)

    def forward(self, desc):
        return self.mlp(desc).squeeze(-1)


class PhysicsNet(nn.Module):
    def __init__(self, depth, width, activation):
        super().__init__()
        self.mlp = MLP(2, 1, depth, width, activation)

    def forward(self, rho, Fo):
        x   = torch.cat([rho**2, Fo], dim=1)
        raw = self.mlp(x)
        return (1.0 - rho**2) * raw


# ============================================================
# INTEGRATION
# ============================================================
def integrate_release(net2, Fo, rho_grid):
    B     = Fo.shape[0]
    n_rho = rho_grid.shape[0]
    Fo_exp  = Fo.view(B, 1).expand(B, n_rho).reshape(-1, 1)
    rho_exp = rho_grid.view(1, n_rho).expand(B, n_rho).reshape(-1, 1)
    u   = net2(rho_exp, Fo_exp).view(B, n_rho)
    M_t = torch.trapz(u * rho_grid**2, rho_grid, dim=1)
    return torch.clamp(1.0 - 3.0 * M_t, min=0.0, max=1.0)


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def eval_release_mse(net1, net2, loader, rho_grid):
    """Validation/test MSE on release — used for checkpoint selection and BHO."""
    net1.eval()
    total_mse = 0.0
    n_batches = 0
    for desc_b, t_b, R_b, c_b in loader:
        desc_b, t_b, R_b, c_b = [x.to(DEVICE) for x in [desc_b, t_b, R_b, c_b]]
        D_pred  = 10.0 ** net1(desc_b)
        Fo      = torch.clamp(D_pred * t_b / R_b**2, 0.0, 1.5)
        release = integrate_release(net2, Fo, rho_grid)
        mse     = torch.mean((release - c_b)**2).item()
        if not np.isfinite(mse):
            return float("inf")
        total_mse += mse
        n_batches += 1
    return total_mse / max(n_batches, 1)


@torch.no_grad()
def eval_male(net1, X_part_t, D_part_t):
    """Diagnostic only — never used for selection."""
    net1.eval()
    D_pred = torch.clamp(10.0 ** net1(X_part_t), min=1e-30)
    valid  = D_part_t > 0
    if valid.sum() == 0:
        return float("inf")
    male = torch.mean(
        torch.abs(torch.log10(D_pred[valid]) - torch.log10(D_part_t[valid]))
    ).item()
    return male if np.isfinite(male) else float("inf")


# ============================================================
# LOAD FROZEN PHYSICSNET
# ============================================================
def load_frozen_net2():
    with open(NET2_CONFIG) as f:
        cfg = json.load(f)
    net2 = PhysicsNet(cfg["depth"], cfg["width"], cfg["activation"]).to(DEVICE)
    net2.load_state_dict(torch.load(NET2_WEIGHTS, map_location=DEVICE))
    for p in net2.parameters():
        p.requires_grad_(False)
    net2.eval()
    print(f"PhysicsNet (FROZEN): depth={cfg['depth']} width={cfg['width']} "
          f"act={cfg['activation']} | "
          f"{sum(p.numel() for p in net2.parameters()):,} params")
    return net2


# ============================================================
# TRAINING LOOP — selects checkpoint by best validation MSE
# ============================================================
def train_one(net1, net2, loader, val_loader, epochs, rho_grid,
              log_every=50, prefix=""):
    optimizer = torch.optim.Adam(
        net1.parameters(),
        lr=net1._lr,
        weight_decay=net1._weight_decay,
    )
    best_mse     = float("inf")
    best_weights = None
    curve        = []

    for epoch in range(epochs):
        net1.train()
        epoch_mse = 0.0

        for desc_b, t_b, R_b, c_b in loader:
            desc_b, t_b, R_b, c_b = [x.to(DEVICE) for x in [desc_b, t_b, R_b, c_b]]
            optimizer.zero_grad()
            D_pred  = 10.0 ** net1(desc_b)
            Fo      = torch.clamp(D_pred * t_b / R_b**2, 0.0, 1.5)
            release = integrate_release(net2, Fo, rho_grid)
            loss    = torch.mean((release - c_b)**2)
            if torch.isnan(loss):
                return None, float("inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net1.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_mse += loss.item()

        epoch_mse /= len(loader)
        val_mse = eval_release_mse(net1, net2, val_loader, rho_grid)

        if val_mse < best_mse:
            best_mse     = val_mse
            best_weights = {k: v.clone() for k, v in net1.state_dict().items()}

        if epoch % log_every == 0 or epoch == epochs - 1:
            curve.append((epoch, epoch_mse, val_mse))
            print(f"  {prefix}epoch {epoch:4d} | "
                  f"train MSE {epoch_mse:.4e} | val MSE {val_mse:.4e} "
                  f"(best {best_mse:.4e})")

    if best_weights is not None:
        net1.load_state_dict(best_weights)

    return curve, best_mse


# ============================================================
# DATALOADER HELPER
# ============================================================
def make_loader(X_scaled, row_mask, t_arr, R_arr, c_arr, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X_scaled,        dtype=torch.float32),
        torch.tensor(t_arr[row_mask], dtype=torch.float32),
        torch.tensor(R_arr[row_mask], dtype=torch.float32),
        torch.tensor(c_arr[row_mask], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# MAIN
# ============================================================
def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | Output: {out.resolve()}")

    X, t, R, c, X_part, D_part, fid, row_fids = load_data(DATA_DIR)
    n_desc   = X.shape[1]
    all_fids = np.unique(fid)

    log10_D_init = LOG10_D_INIT
    print(f"log10(D) init bias: {log10_D_init:.3f}")

    rho_grid = torch.linspace(0, 1, N_RHO_INTEGRATION, device=DEVICE)
    net2     = load_frozen_net2()

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
                "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "batch_size":   trial.suggest_categorical("batch_size",   [32, 64, 128]),
                "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3, 1e-2]),
            }

            tmp = DescriptorNet(n_desc, config["depth"], config["width"],
                                config["activation"], log10_D_init)
            n_params = sum(p.numel() for p in tmp.parameters())

            print(f"\n--- Trial {trial.number} | {config} | {n_params:,} params ---")
            t0 = time.time()

            kf = KFold(n_splits=BHO_K_FOLDS, shuffle=True, random_state=BHO_SEED)
            fold_mses = []

            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(all_fids)):
                tr_fids  = all_fids[tr_idx]
                val_fids = all_fids[val_idx]

                tr_row_mask  = np.isin(row_fids, tr_fids)
                val_row_mask = np.isin(row_fids, val_fids)

                scaler       = StandardScaler()
                X_tr_scaled  = scaler.fit_transform(X[tr_row_mask])
                X_val_scaled = scaler.transform(X[val_row_mask])

                loader     = make_loader(X_tr_scaled,  tr_row_mask,  t, R, c,
                                         config["batch_size"], shuffle=True)
                val_loader = make_loader(X_val_scaled, val_row_mask, t, R, c,
                                         config["batch_size"], shuffle=False)

                torch.manual_seed(BHO_SEED)
                net1 = DescriptorNet(n_desc, config["depth"], config["width"],
                                     config["activation"], log10_D_init).to(DEVICE)
                net1._lr           = config["lr"]
                net1._weight_decay = config["weight_decay"]

                _, best_mse = train_one(
                    net1, net2, loader, val_loader, BHO_EPOCHS, rho_grid,
                    log_every=BHO_EPOCHS,
                    prefix=f"[T{trial.number} F{fold_idx}] ",
                )
                fold_mses.append(best_mse)
                print(f"  [T{trial.number}] fold {fold_idx} best val MSE = {best_mse:.4e}")

            elapsed = time.time() - t0
            fold_mses_clean = [m if np.isfinite(m) else 1.0 for m in fold_mses]
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
            study_name="descriptornet_datasetB",
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

    fids_train, fids_test = train_test_split(
        all_fids, test_size=TEST_FRACTION, random_state=BHO_SEED
    )

    seed_results = []

    for seed in FINAL_SEEDS:
        run_dir = out / "best_model" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Internal validation split (from training particles), seed-dependent
        rng = np.random.default_rng(seed)
        tr_fids_shuffled = fids_train.copy()
        rng.shuffle(tr_fids_shuffled)
        n_ival            = max(1, int(len(fids_train) * INTERNAL_VAL_FRACTION))
        ival_fids         = tr_fids_shuffled[:n_ival]
        actual_tr_fids    = tr_fids_shuffled[n_ival:]

        actual_tr_row_mask = np.isin(row_fids, actual_tr_fids)
        ival_row_mask      = np.isin(row_fids, ival_fids)
        test_row_mask      = np.isin(row_fids, fids_test)

        # Scaler fit on actual training rows only
        scaler             = StandardScaler()
        X_actual_tr_scaled = scaler.fit_transform(X[actual_tr_row_mask])
        X_ival_scaled      = scaler.transform(X[ival_row_mask])
        X_test_scaled      = scaler.transform(X[test_row_mask])

        loader      = make_loader(X_actual_tr_scaled, actual_tr_row_mask, t, R, c,
                                  best_config["batch_size"], shuffle=True)
        ival_loader = make_loader(X_ival_scaled, ival_row_mask, t, R, c,
                                  best_config["batch_size"], shuffle=False)
        test_loader = make_loader(X_test_scaled, test_row_mask, t, R, c,
                                  best_config["batch_size"], shuffle=False)

        # Per-particle tensors for MALE diagnostic on test set
        test_part_mask = np.isin(fid, fids_test)
        X_test_part_t = torch.tensor(scaler.transform(X_part[test_part_mask]),
                                     dtype=torch.float32, device=DEVICE)
        D_test_part_t = torch.tensor(D_part[test_part_mask],
                                     dtype=torch.float32, device=DEVICE)

        torch.manual_seed(seed)
        np.random.seed(seed)

        net1 = DescriptorNet(n_desc, best_config["depth"], best_config["width"],
                             best_config["activation"], log10_D_init).to(DEVICE)
        net1._lr           = best_config["lr"]
        net1._weight_decay = best_config.get("weight_decay", 0.0)
        n_params = sum(p.numel() for p in net1.parameters())
        print(f"\n  Seed {seed} | {n_params:,} params")

        t0 = time.time()
        curve, _ = train_one(
            net1, net2, loader, ival_loader, FINAL_EPOCHS, rho_grid,
            log_every=100, prefix=f"[S{seed}] ",
        )
        elapsed = time.time() - t0

        test_mse  = eval_release_mse(net1, net2, test_loader, rho_grid)
        test_male = eval_male(net1, X_test_part_t, D_test_part_t)
        print(f"  Seed {seed} | Test MSE={test_mse:.4e} | Test MALE={test_male:.4f}  "
              f"({elapsed:.0f}s)")

        torch.save(net1.state_dict(), run_dir / "weights.pt")
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
    print(f"Test MSE:  {summary['MSE_mean']:.4e} +/- {summary['MSE_std']:.4e}")
    print(f"Test MALE: {summary['MALE_mean']:.4f} +/- {summary['MALE_std']:.4f}")
    print(f"Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()