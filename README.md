# Rep2Rep-Shuffle: Self-Supervised MRI Denoising via Random Disjoint Repetition Subsets

[![Journal](https://img.shields.io/badge/MRM-10.1002%2Fmrm.70155-blue.svg)](https://doi.org/10.1002/mrm.70155)
[![arXiv](https://img.shields.io/badge/arXiv-2504.17698-b31b1b.svg)](https://arxiv.org/abs/2504.17698)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **Rep2Rep-Shuffle**, an advanced self-supervised MRI denoising framework that leverages random disjoint subsets of repeated acquisitions for robust, noise-adaptive representation learning.

---

## Overview

While standard Rep2Rep utilizes fixed repetition pairing ($x_0 \to x_1$), **Rep2Rep-Shuffle** generalizes repetition learning by dynamically partitioning repeated acquisitions for each slice and epoch into random disjoint non-empty subsets:
- **Input Group ($\mathcal{A}$)**: A random non-empty subset of repetitions averaged to form the noisy input.
- **Target Group ($\mathcal{B}$)**: A disjoint non-empty subset ($\mathcal{A} \cap \mathcal{B} = \emptyset$) averaged to form the supervisory target.
- **Unused Repetitions**: Remaining repetitions may remain unassigned during the draw, ensuring uniform sampling over all ordered pairs of non-empty disjoint subsets.

This random permutation and grouping prevents model memorization, improves generalization across varying SNR levels, and operates without clean ground-truth images.

```
       All N Repetitions {x_0, x_1, ..., x_{N-1}}
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
  Subset A (Input)            Subset B (Target)
  \bar{x}_A = Avg(A)          \bar{x}_B = Avg(B)  [A ∩ B = ∅]
       │                           │
       ▼                           │
[ CDLNet 2D/3D ]                   │
 (Noise: \sigma / √|A|)            │
       │                           │
       ▼                           │
 Denoised Estimate (\hat{x})       │
       │                           │
       └────────► Loss: MSE(\hat{x}, \bar{x}_B)
```

---

## Key Features

- **Random Disjoint Subset Sampling**: Dynamically draws disjoint input and target groups per slice to maximize data utilization and SNR diversity.
- **Noise-Adaptive CDLNet**: Spatially varying threshold modulation scaled dynamically by group cardinality: $\sigma_{\mathcal{A}} = \sigma / \sqrt{|\mathcal{A}|}$.
- **True Complex-Valued Architectures**: Native support for complex operations with `CDLNet2D` (`Conv2d` / `ConvTranspose2d`) and `CDLNet3D` (`Conv3d` / `ConvTranspose3d`).
- **Flexible Data Formats**: Direct loading of `.mat` (MATLAB v5/v7.3), `.h5`/`.hdf5` (HDF5), and `.npz` files.
- **Zero Inter-project Dependencies**: Completely self-contained implementation with no dependency on other repositories.

---

## Installation

### Prerequisites
- Python $\ge$ 3.9
- PyTorch $\ge$ 2.1 (with CUDA recommended)

### Install via pip (Editable Mode)

```bash
git clone https://github.com/RapidMRI-NYU/Rep2Rep-Shuffle.git
cd Rep2Rep-Shuffle
pip install -e .
```

To install test dependencies:
```bash
pip install -e ".[test]"
pytest
```

---

## Data Format

Input dataset files (`.mat`, `.h5`, `.hdf5`, `.npz`) provide coil-combined complex images and spatial noise maps:

| Array Key | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `coil_combined` | `complex64` | `(N, S, H, W)` | Primary key. Coil-combined complex image-domain repetitions ($N \ge 2$, slices $S$, height $H$, width $W$). |
| `sigma_map` | `float32` | `(S, H, W)` or `(N, S, H, W)` | Real noise standard deviation map in the same intensity units as the images. |

> **Backward Compatibility**: The data loader automatically searches for `coil_combined`, then `images`, and finally falls back to legacy `kspace`. Existing files using the historical `kspace` key continue to work seamlessly without modification.

### Sampling Strategy
For each slice during training:
1. Every repetition is assigned one of three labels: `input` ($\mathcal{A}$), `target` ($\mathcal{B}$), or `unused`, each with probability $1/3$.
2. The assignment is repeated if either $\mathcal{A}$ or $\mathcal{B}$ is empty.
3. Input volume is computed as the mean over $\mathcal{A}$, target volume as the mean over $\mathcal{B}$.
4. Input noise map is propagated as $\sigma / \sqrt{|\mathcal{A}|}$ (or root-sum-square over $|\mathcal{A}|$ for per-repetition maps).
5. The average of all $N$ repetitions is computed as `reference` and used strictly for evaluation metrics (NMSE, PSNR) during validation—**never during training**.

---

## Optional Multicoil Preprocessing

If starting from fully sampled multicoil k-space data, the included preprocessing utility handles IFFT, ESPIRiT/Hamming-filtered sensitivity estimation, Biorthogonal wavelet noise estimation, and adaptive scaling:

```bash
# Reads raw k-space of shape (N, S, C, H, W) and outputs coil-combined complex images
python preprocess.py raw_multicoil.h5 processed_data.mat --input-key kspace --output-key coil_combined --device cuda
```

**Preprocessing Steps:**
1. Centered orthonormal inverse 2D Fast Fourier Transform ($\text{IFFT2c}$).
2. Sensitivity map estimation from repetition-averaged coil images using a 2D Hamming filter (exponent 32).
3. Noise covariance / variance estimation from unfiltered repetition 0 using the `bior4.4` wavelet high-frequency sub-band ($HH$).
4. Adaptive coil combination and 99.9th percentile intensity normalization.
5. Saves `coil_combined` along with a legacy `kspace` alias for full backwards compatibility.

---

## Training

Edit configuration files in `configs/` (`configs/cdl2d.json` or `configs/cdl3d.json`) and run:

```bash
# Train 2D Complex CDLNet with Shuffle Strategy
python train.py configs/cdl2d.json

# Train 3D Complex CDLNet with Shuffle Strategy
python train.py configs/cdl3d.json
```

### Resume Training
Resume from a specific checkpoint using `--resume`:
```bash
python train.py configs/cdl3d.json --resume runs/rep2rep_shuffle_cdl3d/checkpoint_epoch_000050.pt
```

### CDLNet Model Parameters
| Parameter | Paper Notation | Meaning | Example (2D / 3D) |
| :--- | :--- | :--- | :--- |
| `K` | $K$ | Number of unrolled iterative shrinkage steps / network stages | `30` |
| `M` | $M$ | Number of learned convolutional dictionary filters | `507` |
| `P` | $P$ | Filter kernel / patch size | `9` (2D) / `[3, 9, 9]` (3D) |
| `s` | $s$ | Convolution stride for multiscale representation | `2` (2D) / `[1, 2, 2]` (3D) |
| `t0` | $t_0$ | Initial soft-threshold bias | `0.0` |
| `adaptive` | — | Adaptively scale threshold by spatial noise map $\sigma$ | `true` |

### Configuration Structure
```json
{
  "seed": 0,
  "device": "cuda",
  "output": "runs/rep2rep_shuffle_cdl3d",
  "model": {
    "name": "CDLNet3D",
    "K": 30,
    "M": 507,
    "P": [3, 9, 9],
    "s": [1, 2, 2],
    "t0": 0.0,
    "adaptive": true,
    "init": true
  },
  "data": {
    "train": ["/path/to/train_data"],
    "val": ["/path/to/val_data"],
    "image_key": "coil_combined",
    "sigma_key": "sigma_map",
    "crop_size": 128,
    "slab_depth": 12
  },
  "train": {
    "epochs": 100000,
    "batch_size": 1,
    "val_batch_size": 1,
    "num_workers": 4,
    "lr": 1e-5,
    "clip_grad": 0.5,
    "val_every": 10,
    "save_every": 50,
    "scheduler": {
      "name": "cosine",
      "T_max": 4000,
      "eta_min": 1e-6
    }
  }
}
```
> **Note on Epochs**: Because training samples one random slab per subject file each epoch, `epochs` is typically set high (e.g. `100000` for small datasets) to ensure thorough coverage across volume slices. Adjust according to your dataset size.

---

## Inference

Run inference on single or batch files:

```bash
# Average all repetitions and denoise
python infer.py runs/rep2rep_shuffle_cdl3d/checkpoint_epoch_000200.pt input_case.mat output_denoised.mat --device cuda

# Denoise a specific subset of repetitions
python infer.py runs/rep2rep_shuffle_cdl3d/checkpoint_epoch_000200.pt input_case.mat output_denoised.mat --reps 0,2,4 --device cuda
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{janjusevicRep2Rep2026,
  author    = {Janju{\v{s}}evi{\'c}, Nikola and Chen, Jingjia and Ginocchio, Luke and Bruno, Mary and Huang, Yuhui and Wang, Yao and Chandarana, Hersh and Feng, Li},
  title     = {Self-Supervised Noise Adaptive {MRI} Denoising via Repetition to Repetition ({Rep2Rep}) Learning},
  journal   = {Magnetic Resonance in Medicine},
  volume    = {95},
  number    = {3},
  pages     = {1619--1633},
  year      = {2026},
  doi       = {10.1002/mrm.70155}
}

@article{janjusevicRep2Rep2025arxiv,
  author    = {Janju{\v{s}}evi{\'c}, Nikola and Chen, Jingjia and Ginocchio, Luke and Bruno, Mary and Huang, Yuhui and Wang, Yao and Chandarana, Hersh and Feng, Li},
  title     = {Self-Supervised Noise Adaptive {MRI} Denoising via Repetition to Repetition ({Rep2Rep}) Learning},
  journal   = {arXiv preprint arXiv:2504.17698},
  year      = {2025},
  doi       = {10.48550/arXiv.2504.17698}
}

@article{janjusevicCDLNet2022,
  author    = {Janju{\v{s}}evi{\'c}, Nikola and Khalilian-Gourtani, Amirhossein and Wang, Yao},
  title     = {{CDLNet}: Noise-Adaptive Convolutional Dictionary Learning Network for Blind Denoising and Demosaicing},
  journal   = {IEEE Open Journal of Signal Processing},
  volume    = {3},
  pages     = {196--211},
  year      = {2022},
  doi       = {10.1109/OJSP.2022.3172842}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE). Third-party acknowledgments can be found in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

