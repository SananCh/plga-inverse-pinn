"""
BHO + multi-seed training for DescriptorNet on Dataset A.

Two-stage workflow:
    1. BHO  -- Optuna finds the best DescriptorNet architecture/LR
               using a validation split carved from the training set.
               Objective: release RMSE on the validation particles.
    2. Final -- Best config retrained from scratch with multiple seeds
               on the full training set, evaluated on the held-out test set.

Outputs (in OUTPUT_DIR):
    study.pkl
    best_config.json
    bho_log.csv
    best_model/
        seed_0/
            weights.pt
            metrics.json       -- RMSE + MALE on test set
            training_curve.npy
        ...
        summary.json           -- mean +/- std across seeds
"""

import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR    = "../datasetA_training_data"
TEST_DIR     = "../datasetA_testing_data"
NET2_WEIGHTS = "physicsnet/physicsnet_pretrained.pt"
OUTPUT_DIR   = "descriptornet_datasetA_bho_with_Crank"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Network 2 (frozen) — loaded from PhysicsNet BHO best_config.json ---
NET2_CONFIG = "../physicsnet/physicsnet_bho/best_config.json"

# --- BHO ---
BHO_N_TRIALS  = 50
BHO_EPOCHS    = 100        # Dataset A converges fast — no need for long BHO trials
VAL_FRACTION  = 0.15      # fraction of training particles held out for BHO val
BHO_SEED      = 42

# --- Final retraining ---
FINAL_EPOCHS  = 200
FINAL_SEEDS   = [0, 1, 2, 3, 4]

# --- Fixed training knobs (not searched) ---
N_RHO_INTEGRATION = 30
BATCH_SIZE        = 48
GRAD_CLIP         = 1.0
N_DESC            = 3     # Dataset A: [eps, tau_ort, log10(D0)]


# ============================================================
# DATA
# ============================================================
def load_dataset(data_dir):
    c     = np.load(f"{data_dir}/c_release_matrix.npy")
    t     = np.load(f"{data_dir}/t_release_matrix.npy")
    D0    = np.load(f"{data_dir}/D0_array.npy")
    eps   = np.load(f"{data_dir}/epsilon_array.npy")
    tau   = np.load(f"{data_dir}/tau_ort_array.npy")
    D_eff = np.load(f"{data_dir}/Deff_array.npy")
    R     = np.load(f"{data_dir}/R_array.npy")
    return {"c": c, "t": t, "D0": D0, "eps": eps,
            "tau": tau, "D_eff": D_eff, "R": R,
            "N": c.shape[0], "T": c.shape[1]}


def make_tensors(ds, indices, desc_mean, desc_std):
    desc_raw  = np.stack([ds["eps"][indices],
                          ds["tau"][indices],
                          np.log10(ds["D0"][indices])], axis=1)
    desc_norm = (desc_raw - desc_mean) / desc_std
    return {
        "desc":  torch.tensor(desc_norm,           dtype=torch.float32, device=DEVICE),
        "t":     torch.tensor(ds["t"][indices],    dtype=torch.float32, device=DEVICE),
        "c":     torch.tensor(ds["c"][indices],    dtype=torch.float32, device=DEVICE),
        "R":     torch.tensor(ds["R"][indices],    dtype=torch.float32, device=DEVICE),
        "D_eff": torch.tensor(ds["D_eff"][indices],dtype=torch.float32, device=DEVICE),
        "N":     len(indices),
        "T":     ds["T"],
    }


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
# BRIDGE + INTEGRATION
# ============================================================
def crank_release_torch(Fo, n_terms=200):
    n = torch.arange(1, n_terms + 1, device=Fo.device, dtype=Fo.dtype)
    exponent = -(n**2) * (torch.pi**2) * Fo.unsqueeze(-1)   # (N, T, M)
    series = torch.sum(torch.exp(exponent) / n**2, dim=-1)
    return 1.0 - (6.0 / torch.pi**2) * series

def predicted_release(net2, D_eff, t_abs, R):
    N, T = t_abs.shape
    Fo = D_eff.view(N, 1) * t_abs / R.view(N, 1) ** 2
    return crank_release_torch(Fo)
    
# ============================================================
# EVALUATION (no grad, per-particle)
# ============================================================
@torch.no_grad()
def evaluate(net1, net2, split):
    net1.eval()
    N   = split["N"]
    T   = split["T"]
    nb  = (N + BATCH_SIZE - 1) // BATCH_SIZE

    male_sum = 0.0
    mse_sum  = 0.0

    for b in range(nb):
        sl       = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
        D_pred   = 10.0 ** net1(split["desc"][sl])
        D_true   = split["D_eff"][sl]
        male_sum += torch.sum(
            torch.abs(torch.log10(D_pred) - torch.log10(D_true))
        ).item()

        rel_pred  = predicted_release(net2, D_pred,
                                      split["t"][sl], split["R"][sl])
        mse_sum += torch.mean((rel_pred - split["c"][sl])**2).item()

    male = male_sum / N
    mse  = mse_sum / nb
    return male, mse


# ============================================================
# TRAINING LOOP
# ============================================================
def train_one(net1, net2, train_split, epochs, lr, batch_size=BATCH_SIZE,
              val_split=None, log_every=10, trial_num=None):
    """
    Train DescriptorNet. Prints train RMSE (and val RMSE if val_split given)
    every log_every epochs so progress is visible.
    """
    optimizer = torch.optim.Adam(net1.parameters(), lr=lr)
    N         = train_split["N"]
    n_batches = (N + batch_size - 1) // batch_size
    curve     = []
    prefix    = f"[Trial {trial_num}] " if trial_num is not None else ""

    for epoch in range(epochs):
        net1.train()
        perm       = torch.randperm(N, device=DEVICE)
        epoch_mse  = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            optimizer.zero_grad()

            D_pred   = 10.0 ** net1(train_split["desc"][idx])
            rel_pred = predicted_release(net2, D_pred,
                                         train_split["t"][idx],
                                         train_split["R"][idx])
            loss = torch.mean((rel_pred - train_split["c"][idx])**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net1.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_mse += loss.item()

        epoch_mse /= n_batches
        train_mse = epoch_mse

        if epoch % log_every == 0 or epoch == epochs - 1:
            if val_split is not None:
                _, val_mse = evaluate(net1, net2, val_split)
                curve.append((epoch, train_mse, val_mse))
                print(f"  {prefix}epoch {epoch:4d} | "
                      f"train MSE {train_mse:.4e} | "
                      f"val MSE {val_mse:.4e}")
            else:
                curve.append((epoch, train_mse))
                print(f"  {prefix}epoch {epoch:4d} | train MSE {train_mse:.4e}")

    return curve


# ============================================================
# LOAD FROZEN PHYSICSNET
# ============================================================
def load_frozen_net2():
    with open(NET2_CONFIG) as f:
        net2_cfg = json.load(f)
    net2 = PhysicsNet(
        depth=net2_cfg["depth"],
        width=net2_cfg["width"],
        activation=net2_cfg["activation"],
    ).to(DEVICE)
    state = torch.load(NET2_WEIGHTS, map_location=DEVICE)
    net2.load_state_dict(state)
    for p in net2.parameters():
        p.requires_grad_(False)
    net2.eval()
    n2_params = sum(p.numel() for p in net2.parameters())
    print(f"PhysicsNet (FROZEN): {n2_params:,} params | "
          f"depth={net2_cfg['depth']} width={net2_cfg['width']} "
          f"act={net2_cfg['activation']}")
    return net2


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def make_objective(net2, bho_train, bho_val, log10_D_init, csv_writer):
    def objective(trial):
        config = {
            "depth":      trial.suggest_categorical("depth",      [2, 3, 4, 5, 6]),
            "width":      trial.suggest_categorical("width",      [32, 64, 128, 256]),
            "activation": trial.suggest_categorical("activation", ["tanh", "silu", "gelu"]),
            "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }

        torch.manual_seed(BHO_SEED)
        net1 = DescriptorNet(
            n_desc=N_DESC,
            depth=config["depth"],
            width=config["width"],
            activation=config["activation"],
            log10_D_init=log10_D_init,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in net1.parameters())

        print(f"\n--- Trial {trial.number} | {config} | {n_params:,} params ---")
        t0 = time.time()

        try:
            train_one(net1, net2, bho_train,
                      epochs=BHO_EPOCHS, lr=config["lr"],
                      batch_size=config["batch_size"],
                      val_split=bho_val, log_every=10,
                      trial_num=trial.number)
            _, val_mse = evaluate(net1, net2, bho_val)
        except Exception as e:
            print(f"  FAILED: {e}")
            return float("inf")

        elapsed = time.time() - t0
        print(f"  val MSE = {val_mse:.4e}  ({elapsed:.0f}s)")

        csv_writer.writerow({
            "trial": trial.number, "val_mse": val_mse,
            "n_params": n_params, "elapsed_s": round(elapsed, 1),
            **config,
        })
        return val_mse

    return objective


# ============================================================
# MAIN
# ============================================================
def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output: {out.resolve()}")

    # --- Load data ---
    train_raw = load_dataset(TRAIN_DIR)
    test_raw  = load_dataset(TEST_DIR)
    N_train   = train_raw["N"]
    print(f"Train particles: {N_train} | Test particles: {test_raw['N']}")

    # Descriptor normalisation stats from full training set
    desc_raw   = np.stack([train_raw["eps"], train_raw["tau"],
                           np.log10(train_raw["D0"])], axis=1)
    desc_mean  = desc_raw.mean(axis=0)
    desc_std   = desc_raw.std(axis=0)
    log10_D_init = float(np.mean(np.log10(train_raw["D_eff"])))
    print(f"log10(D) init bias: {log10_D_init:.3f}")

    # BHO validation split (carved from training set)
    rng        = np.random.default_rng(BHO_SEED)
    all_idx    = np.arange(N_train)
    rng.shuffle(all_idx)
    n_val      = int(N_train * VAL_FRACTION)
    val_idx    = all_idx[:n_val]
    bho_tr_idx = all_idx[n_val:]
    print(f"BHO train: {len(bho_tr_idx)} | BHO val: {len(val_idx)}")

    bho_train = make_tensors(train_raw, bho_tr_idx, desc_mean, desc_std)
    bho_val   = make_tensors(train_raw, val_idx,    desc_mean, desc_std)
    test      = make_tensors(test_raw,  np.arange(test_raw["N"]),
                             desc_mean, desc_std)

    # Full training set (used for final seed runs)
    full_train = make_tensors(train_raw, all_idx, desc_mean, desc_std)

    # Load frozen PhysicsNet
    net2 = None

    # ---- BHO ----
    print(f"\nStarting BHO: {BHO_N_TRIALS} trials x {BHO_EPOCHS} epochs")
    csv_path  = out / "bho_log.csv"
    fieldnames = ["trial", "val_mse", "n_params", "elapsed_s",
                  "depth", "width", "activation", "lr", "batch_size"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        study = optuna.create_study(
            direction="minimize",
            study_name="descriptornet_datasetA",
            sampler=optuna.samplers.TPESampler(seed=BHO_SEED),
        )
        study.optimize(
            make_objective(net2, bho_train, bho_val, log10_D_init, writer),
            n_trials=BHO_N_TRIALS,
            show_progress_bar=False,
        )

    with open(out / "study.pkl", "wb") as f:
        pickle.dump(study, f)

    best_config = study.best_trial.params
    print(f"\nBHO done. Best trial {study.best_trial.number}:")
    print(f"  val MSE  = {study.best_value:.4e}")
    print(f"  config   = {best_config}")

    with open(out / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # ---- Final retraining with multiple seeds ----
    print(f"\nRetraining best config with {len(FINAL_SEEDS)} seeds "
          f"x {FINAL_EPOCHS} epochs...")
    seed_results = []

    for seed in FINAL_SEEDS:
        seed_dir = out / "best_model" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        net1 = DescriptorNet(
            n_desc=N_DESC,
            depth=best_config["depth"],
            width=best_config["width"],
            activation=best_config["activation"],
            log10_D_init=log10_D_init,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in net1.parameters())
        print(f"\n  Seed {seed} | {n_params:,} params")

        t0    = time.time()
        curve = train_one(net1, net2, full_train,
                          epochs=FINAL_EPOCHS,
                          lr=best_config["lr"],
                          batch_size=best_config["batch_size"],
                          val_split=test,
                          log_every=1)
        elapsed = time.time() - t0

        male, mse = evaluate(net1, net2, test)
        print(f"  Test MALE={male:.4f}  MSE={mse:.4e}  ({elapsed:.0f}s)")

        torch.save(net1.state_dict(), seed_dir / "weights.pt")

        metrics = {
            "seed": seed, "test_MALE": male, "test_MSE": mse,
            "epochs": FINAL_EPOCHS, "elapsed_s": round(elapsed, 1),
            "config": best_config,
        }
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        np.save(seed_dir / "training_curve.npy", np.array(curve))
        seed_results.append({"seed": seed, "MALE": male, "MSE": mse})

    males = [r["MALE"] for r in seed_results]
    mses  = [r["MSE"]  for r in seed_results]

    summary = {
        "best_config":   best_config,
        "bho_val_mse":   study.best_value,
        "seed_results":  seed_results,
        "MALE_mean":     float(np.mean(males)),
        "MALE_std":      float(np.std(males)),
        "MSE_mean":      float(np.mean(mses)),
        "MSE_std":       float(np.std(mses)),
    }
    with open(out / "best_model" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"MALE: {summary['MALE_mean']:.4f} +/- {summary['MALE_std']:.4f}")
    print(f"MSE:  {summary['MSE_mean']:.4e} +/- {summary['MSE_std']:.4e}")
    print(f"All outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()