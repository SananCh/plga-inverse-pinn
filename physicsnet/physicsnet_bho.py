"""
Bayesian Hyperparameter Optimisation for PhysicsNet pre-training.

Runs Optuna to find the best architecture and training hyperparameters
for the physics surrogate network. Every trial is logged with full
details. After BHO, the best configuration is retrained with multiple
seeds for the final reported result.

Outputs (in BHO_OUTPUT_DIR):
    study.pkl                    -- full Optuna study (all trials)
    best_config.json             -- best hyperparameters found
    bho_log.csv                  -- trial-by-trial log
    best_model/
        seed_0/weights.pt
        seed_0/metrics.json
        seed_0/training_curve.npy
        seed_1/ ...
        ...
        summary.json             -- mean +/- std across seeds
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import optuna
import pickle
import csv
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BHO_OUTPUT_DIR = "physicsnet_bho"
BHO_N_TRIALS   = 50
BHO_EPOCHS     = 5000      # epochs per BHO trial (shorter than final)
FINAL_EPOCHS   = 10000     # epochs for final seed runs
FINAL_SEEDS    = [0, 1, 2, 3, 4]

FO_MAX         = 1.5
GRAD_CLIP      = 1.0
N_COLLOCATION  = 4000
N_IC           = 2000

# Fixed seed for BHO trials (reproducible trial-level results)
BHO_SEED = 42

# Verification grid (held-out, never used in training)
VERIFY_RHO = np.linspace(0.02, 0.98, 40)
VERIFY_FO  = np.linspace(0.02, FO_MAX, 40)


# ============================================================
# SEARCH SPACE
# ============================================================
def sample_hyperparameters(trial):
    return {
        "depth":      trial.suggest_categorical("depth",      [3, 4, 5, 6, 7]),
        "width":      trial.suggest_categorical("width",      [64, 128, 192, 256]),
        "activation": trial.suggest_categorical("activation", ["tanh", "silu", "softplus"]),
        "lr":         trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "w_ic":       trial.suggest_categorical("w_ic",       [1, 5, 10, 20, 50]),
        "rho_bias":   trial.suggest_categorical("rho_bias",   [1, 2, 3, 4]),
        "fo_bias":    trial.suggest_categorical("fo_bias",    [1, 2, 3, 4]),
    }


# ============================================================
# NETWORK
# ============================================================
def get_activation(name):
    acts = {"tanh": nn.Tanh(), "silu": nn.SiLU(),
            "gelu": nn.GELU(), "softplus": nn.Softplus()}
    return acts[name.lower()]


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


class PhysicsNet(nn.Module):
    def __init__(self, depth, width, activation):
        super().__init__()
        self.mlp = MLP(2, 1, depth, width, activation)

    def forward(self, rho, Fo):
        x   = torch.cat([rho**2, Fo], dim=1)
        raw = self.mlp(x)
        return (1.0 - rho**2) * raw


# ============================================================
# COLLOCATION SAMPLING (inline, no external dependency)
# ============================================================
def _exp_warp(u, strength):
    if strength <= 1e-8:
        return u
    s = torch.tensor(strength, dtype=u.dtype, device=u.device)
    return (torch.exp(strength * u) - 1.0) / torch.expm1(s)


def sample_collocation(n, fo_max, device, rho_bias=2.0, fo_bias=2.0):
    u_rho = torch.rand(n, 1, device=device)
    u_fo  = torch.rand(n, 1, device=device)
    Fo  = _exp_warp(u_fo,  fo_bias) * fo_max
    rho = 1.0 - _exp_warp(u_rho, rho_bias)
    return rho.requires_grad_(True), Fo.requires_grad_(True)


# ============================================================
# LOSSES
# ============================================================
def pde_residual(net, rho_bias, fo_bias):
    device = next(net.parameters()).device
    rho, Fo = sample_collocation(N_COLLOCATION, FO_MAX, device,
                                 rho_bias=rho_bias, fo_bias=fo_bias)
    u = net(rho, Fo)
    du_dFo   = torch.autograd.grad(u,  Fo,  torch.ones_like(u),
                                   create_graph=True)[0]
    du_drho  = torch.autograd.grad(u,  rho, torch.ones_like(u),
                                   create_graph=True)[0]
    d2u_drho2 = torch.autograd.grad(du_drho, rho, torch.ones_like(du_drho),
                                    create_graph=True)[0]
    laplacian = d2u_drho2 + (2.0 / (rho + 1e-6)) * du_drho
    return torch.mean((du_dFo - laplacian) ** 2)


def ic_loss_fn(net):
    device = next(net.parameters()).device
    rho = torch.rand(N_IC, 1, device=device)
    Fo  = torch.zeros(N_IC, 1, device=device)
    u   = net(rho, Fo)
    return torch.mean((u - 1.0) ** 2)


# ============================================================
# VERIFICATION
# ============================================================
def crank_u(rho, Fo, n_terms=170):
    rho = np.asarray(rho, dtype=float)
    Fo  = np.asarray(Fo,  dtype=float)
    out = np.zeros_like(rho)
    for n in range(1, n_terms + 1):
        out += ((-1.0) ** (n + 1) / n) * np.sin(n * np.pi * rho) \
               * np.exp(-(n ** 2) * np.pi ** 2 * Fo)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (2.0 / (np.pi * rho)) * out
    limit0 = 2.0 * sum(((-1.0) ** (n + 1)) * np.exp(-(n ** 2) * np.pi ** 2 * Fo)
                       for n in range(1, n_terms + 1))
    u = np.where(rho < 1e-9, limit0, u)
    return np.clip(u, 0.0, 1.0)


def build_verify_grid():
    RR, FF = np.meshgrid(VERIFY_RHO, VERIFY_FO)
    u_true  = crank_u(RR.ravel(), FF.ravel())
    rho_t   = torch.tensor(RR.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    fo_t    = torch.tensor(FF.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    return rho_t, fo_t, u_true


def verify(net, rho_t, fo_t, u_true):
    net.eval()
    with torch.no_grad():
        u_net = net(rho_t, fo_t).cpu().numpy().ravel()
    err = np.abs(u_net - u_true)
    return float(err.mean()), float(err.max())


# ============================================================
# TRAINING LOOP
# ============================================================
def train(config, epochs, seed, verbose=False):
    """
    Train one PhysicsNet with given config and seed.
    Returns: (mean_err, max_err, training_curve)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = PhysicsNet(
        depth=config["depth"],
        width=config["width"],
        activation=config["activation"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    rho_t, fo_t, u_true = build_verify_grid()

    curve = []   # (epoch, pde, ic) logged every 250 epochs

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()

        loss_pde = pde_residual(net, config["rho_bias"], config["fo_bias"])
        loss_ic  = ic_loss_fn(net)
        loss     = loss_pde + config["w_ic"] * loss_ic

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()

        if epoch % 250 == 0 or epoch == epochs - 1:
            curve.append((epoch, loss_pde.item(), loss_ic.item()))
            if verbose:
                print(f"  epoch {epoch:5d} | pde {loss_pde.item():.3e} "
                      f"| ic {loss_ic.item():.3e}")

    mean_err, max_err = verify(net, rho_t, fo_t, u_true)
    return mean_err, max_err, curve, net


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def make_objective(csv_writer, trial_start_times):
    def objective(trial):
        config = sample_hyperparameters(trial)
        n_params = sum(
            p.numel() for p in
            PhysicsNet(config["depth"], config["width"],
                       config["activation"]).parameters()
        )

        print(f"\n--- Trial {trial.number} ---")
        print(f"  Config: {config}")
        print(f"  Params: {n_params:,}")

        t0 = time.time()
        try:
            mean_err, max_err, curve, _ = train(
                config, epochs=BHO_EPOCHS, seed=BHO_SEED, verbose=False
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            return float("inf")

        elapsed = time.time() - t0
        print(f"  mean|err|={mean_err:.4e}  max|err|={max_err:.4e}  "
              f"time={elapsed:.0f}s")

        # Log to CSV
        csv_writer.writerow({
            "trial":      trial.number,
            "mean_err":   mean_err,
            "max_err":    max_err,
            "n_params":   n_params,
            "elapsed_s":  round(elapsed, 1),
            **config,
        })

        # Report intermediate value for Optuna pruning (optional)
        trial.report(mean_err, step=BHO_EPOCHS)

        return mean_err   # minimise mean verification error

    return objective


# ============================================================
# MAIN
# ============================================================
def main():
    out = Path(BHO_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"BHO: {BHO_N_TRIALS} trials x {BHO_EPOCHS} epochs each")
    print(f"Final retraining: {len(FINAL_SEEDS)} seeds x {FINAL_EPOCHS} epochs")
    print(f"Output directory: {out.resolve()}")
    print("=" * 70)

    # --- BHO ---
    csv_path = out / "bho_log.csv"
    fieldnames = ["trial", "mean_err", "max_err", "n_params", "elapsed_s",
                  "depth", "width", "activation", "lr", "w_ic",
                  "rho_bias", "fo_bias"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        study = optuna.create_study(
            direction="minimize",
            study_name="physicsnet_bho",
            sampler=optuna.samplers.TPESampler(seed=BHO_SEED),
        )
        study.optimize(
            make_objective(writer, {}),
            n_trials=BHO_N_TRIALS,
            show_progress_bar=True,
        )

    # Save study
    with open(out / "study.pkl", "wb") as f:
        pickle.dump(study, f)

    best = study.best_trial
    best_config = best.params
    best_config["w_ic"]     = best_config.get("w_ic", 1)
    best_config["rho_bias"] = best_config.get("rho_bias", 2)
    best_config["fo_bias"]  = best_config.get("fo_bias", 2)

    print("\n" + "=" * 70)
    print(f"BHO complete. Best trial: {best.number}")
    print(f"  mean|err| = {best.value:.4e}")
    print(f"  Config: {best_config}")

    with open(out / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # --- Final retraining with multiple seeds ---
    print("\n" + "=" * 70)
    print(f"Retraining best config with {len(FINAL_SEEDS)} seeds...")

    seed_results = []

    for seed in FINAL_SEEDS:
        seed_dir = out / "best_model" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Seed {seed}:")
        mean_err, max_err, curve, net = train(
            best_config, epochs=FINAL_EPOCHS, seed=seed, verbose=True
        )
        print(f"  Final: mean|err|={mean_err:.4e}  max|err|={max_err:.4e}")

        # Save weights
        torch.save(net.state_dict(), seed_dir / "weights.pt")

        # Save metrics
        metrics = {
            "seed":     seed,
            "mean_err": mean_err,
            "max_err":  max_err,
            "config":   best_config,
            "epochs":   FINAL_EPOCHS,
        }
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save training curve
        np.save(seed_dir / "training_curve.npy", np.array(curve))

        seed_results.append({"seed": seed, "mean_err": mean_err, "max_err": max_err})

    # Summary across seeds
    mean_errs = [r["mean_err"] for r in seed_results]
    max_errs  = [r["max_err"]  for r in seed_results]

    summary = {
        "best_config":       best_config,
        "bho_best_mean_err": best.value,
        "seed_results":      seed_results,
        "mean_err_mean":     float(np.mean(mean_errs)),
        "mean_err_std":      float(np.std(mean_errs)),
        "max_err_mean":      float(np.mean(max_errs)),
        "max_err_std":       float(np.std(max_errs)),
    }

    with open(out / "best_model" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"mean|err| across seeds: "
          f"{summary['mean_err_mean']:.4e} +/- {summary['mean_err_std']:.4e}")
    print(f"max|err|  across seeds: "
          f"{summary['max_err_mean']:.4e} +/- {summary['max_err_std']:.4e}")
    print(f"All outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()