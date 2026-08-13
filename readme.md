# Inverse PINN for Drug Release Prediction from PLGA Microspheres

Code and data for the paper *"Physics Informed Neural Networks for drug release
prediction from PLGA particles"* (submitted to the International Journal of
Pharmaceutics).

Archived at https://doi.org/10.5281/zenodo.21924123

The framework couples two networks: **PhysicsNet**, which learns the universal
dimensionless solution of Fickian diffusion in a sphere from the PDE residual
alone, and **DescriptorNet**, which maps formulation descriptors to effective
diffusivity. Coupled through the Fourier number, they predict full release
profiles and recover effective diffusivity without diffusivity labels.

## Repository structure

```
plga_dataset/                  Source and derived data (see Data section)
physicsnet/                    PhysicsNet pre-training, BHO, retraining, weights
datasetA_generator.ipynb       Synthetic benchmark generator (Dataset A)
datasetB_generator.ipynb       Idealized real dataset generator (Dataset B)
datasetC_generator.ipynb       Real dataset preparation (Dataset C)
datasetA_training_data/        Generated Dataset A arrays (.npy)
datasetA_testing_data/
descriptornet_datasetA/        DescriptorNet BHO + retraining, per dataset
descriptornet_datasetB/
descriptornet_datasetC/
descriptornet_supervised/      Directly supervised diffusivity baseline
fit_Crank.ipynb                Crank analytical fits to measured profiles
plots/                         Paper figures (PNG + PDF)
LICENSE                        MIT license (code)
```

Each `*_bho/` folder contains the Optuna log (`bho_log.csv`), the study object
(`study.pkl`), the selected configuration (`best_config.json`), and
`best_model/` with per-seed weights, metrics, and training curves.

## Data

- `mp_dataset_initial_formulation.xlsx` and `mp_dataset_processed.xlsx` are from
  the dataset compiled by **Bao et al.**, *A dataset on formulation parameters
  and characteristics of drug-loaded PLGA microparticles*, Sci. Data (2025),
  https://doi.org/10.1038/s41597-025-04621-9 — 321 drug–polymer formulations.
  These files are redistributed here under CC BY 4.0, with attribution to the
  original authors.
- `crank_fit.xlsx` — effective diffusivities obtained by fitting Crank's
  analytical solution to each measured profile (this work).
- `release_dataset_even.xlsx` / `release_dataset_with_Crank_release_even.xlsx`
  — measured and Crank-idealized profiles resampled on an even time grid
  (this work); these are the inputs for Datasets B and C.

## Reproducing the pipeline

1. **PhysicsNet pre-training** — `physicsnet/physicsnet_pretrain.py`
   (pre-trained weights included: `physicsnet_pretrained.pt`).
2. **Dataset generation** — `datasetA_generator.ipynb`,
   `datasetB_generator.ipynb`, `datasetC_generator.ipynb`, and
   `fit_Crank.ipynb` for the Crank fits.
3. **Hyperparameter optimization** — `*_bho.py` in each model folder
   (Optuna, TPE sampler).
4. **Final retraining and evaluation** — `*_retraining.ipynb` in each model
   folder, using the configuration in `best_config.json` over five seeds.

Trained on a single NVIDIA H100; full pipeline requires under two hours of
GPU time.

## Requirements

Python 3.10+, PyTorch, NumPy, pandas, scikit-learn, Optuna, openpyxl,
matplotlib.

## License

Code is released under the MIT License (see `LICENSE`). The redistributed Bao
et al. data files are covered by CC BY 4.0, as noted above.