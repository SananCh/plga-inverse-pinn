"""
Retrain PhysicsNet from the best BHO configuration and save weights.

Reads best_config.json produced by physicsnet_bho.py, trains one
PhysicsNet with a chosen seed, verifies against Crank, and saves
the weights to a .pt file ready for importing into the inverse pipeline.

Usage:
    python physicsnet_retrain.py
    python physicsnet_retrain.py --config path/to/best_config.json
                                 --seed 0
                                 --epochs 10000
                                 --out physicsnet_pretrained.pt
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# CONFIG DEFAULTS
# ============================================================
DEFAULT_CONFIG = "physicsnet_bho/best_config.json"
DEFAULT_OUT    = "physicsnet_pretrained.pt"
DEFAULT_SEED   = 0
DEFAULT_EPOCHS = 10000
FO_MAX         = 1.5
GRAD_CLIP      = 1.0
N_COLLOCATION  = 4000
N_IC           = 2000
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# NETWORK
# ============================================================
def get_activation(name):
    return {"tanh": nn.Tanh(), "silu": nn.SiLU(),
            "gelu": nn.GELU(), "softplus": nn.Softplus()}[name.lower()]


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
# COLLOCATION SAMPLING
# ============================================================
def _exp_warp(u, strength):
    if strength <= 1e-8:
        return u
    s = torch.tensor(strength, dtype=u.dtype, device=u.device)
    return (torch.exp(strength * u) - 1.0) / torch.expm1(s)


def sample_collocation(n, fo_max, device, rho_bias=2.0, fo_bias=2.0):
    u_rho = torch.rand(n, 1, device=device)
    u_fo  = torch.rand(n, 1, device=device)
    Fo    = _exp_warp(u_fo,  fo_bias) * fo_max
    rho   = 1.0 - _exp_warp(u_rho, rho_bias)
    return rho.requires_grad_(True), Fo.requires_grad_(True)


# ============================================================
# LOSSES
# ============================================================
def pde_residual(net, rho_bias, fo_bias):
    device = next(net.parameters()).device
    rho, Fo = sample_collocation(N_COLLOCATION, FO_MAX, device,
                                 rho_bias=rho_bias, fo_bias=fo_bias)
    u = net(rho, Fo)
    du_dFo    = torch.autograd.grad(u, Fo, torch.ones_like(u),
                                    create_graph=True)[0]
    du_drho   = torch.autograd.grad(u, rho, torch.ones_like(u),
                                    create_graph=True)[0]
    d2u_drho2 = torch.autograd.grad(du_drho, rho, torch.ones_like(du_drho),
                                    create_graph=True)[0]
    laplacian = d2u_drho2 + (2.0 / (rho + 1e-6)) * du_drho
    return torch.mean((du_dFo - laplacian) ** 2)


def ic_loss_fn(net, w_ic):
    device = next(net.parameters()).device
    rho = torch.rand(N_IC, 1, device=device)
    Fo  = torch.zeros(N_IC, 1, device=device)
    u   = net(rho, Fo)
    return w_ic * torch.mean((u - 1.0) ** 2)


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


def verify(net):
    rho_vals = np.linspace(0.02, 0.98, 40)
    fo_vals  = np.linspace(0.02, FO_MAX, 40)
    RR, FF   = np.meshgrid(rho_vals, fo_vals)
    u_true   = crank_u(RR.ravel(), FF.ravel())
    rho_t = torch.tensor(RR.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    fo_t  = torch.tensor(FF.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    net.eval()
    with torch.no_grad():
        u_net = net(rho_t, fo_t).cpu().numpy().ravel()
    err = np.abs(u_net - u_true)
    return float(err.mean()), float(err.max())


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--out",    default=DEFAULT_OUT)
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Device:  {DEVICE}")
    print(f"Config:  {config}")
    print(f"Seed:    {args.seed}")
    print(f"Epochs:  {args.epochs}")
    print(f"Output:  {args.out}")
    print("-" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build network
    net = PhysicsNet(
        depth=config["depth"],
        width=config["width"],
        activation=config["activation"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"PhysicsNet: {n_params:,} parameters")

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    rho_bias = config.get("rho_bias", 2)
    fo_bias  = config.get("fo_bias",  2)
    w_ic     = config.get("w_ic",     1)

    # Training loop
    for epoch in range(args.epochs):
        net.train()
        optimizer.zero_grad()

        loss_pde = pde_residual(net, rho_bias, fo_bias)
        loss_ic  = ic_loss_fn(net, w_ic)
        loss     = loss_pde + loss_ic

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()

        if epoch % 500 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:6d} | "
                  f"pde {loss_pde.item():.3e} | "
                  f"ic {loss_ic.item():.3e}")

    # Verify and save
    print("-" * 60)
    mean_err, max_err = verify(net)
    print(f"Verification vs Crank: mean|err|={mean_err:.4e}  max|err|={max_err:.4e}")

    torch.save(net.state_dict(), args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()