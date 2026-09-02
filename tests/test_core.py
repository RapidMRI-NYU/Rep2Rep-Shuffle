import json

import numpy as np
import torch
from scipy.io import savemat

from rep2rep_shuffle.data import Rep2RepShuffleDataset, read_case, sample_disjoint_groups, sigma_for_average
from rep2rep_shuffle.engine import infer, train
from rep2rep_shuffle.model import CDLNet2D, CDLNet3D
from rep2rep_shuffle.preprocess import fft2c, preprocess_multicoil


def make_case(path, repetitions=4, slices=3, height=16, width=18, key="coil_combined"):
    rng = np.random.default_rng(4)
    base = rng.normal(size=(slices, height, width)) + 1j * rng.normal(size=(slices, height, width))
    images = np.stack([base + 0.05 * index for index in range(repetitions)]).astype(np.complex64)
    sigma = np.stack([
        np.full((slices, height, width), 0.01 * (index + 1), np.float32)
        for index in range(repetitions)
    ])
    savemat(path, {key: images, "sigma_map": sigma})
    return images, sigma


def test_per_slice_disjoint_pair_and_reference(tmp_path):
    path = tmp_path / "case.mat"
    images, sigma = make_case(path)
    dataset = Rep2RepShuffleDataset(path, slab_depth=2, crop_size=(12, 14), training=False, seed=0)
    sample = dataset[0]
    rng = np.random.default_rng(np.random.SeedSequence([0, 0, 0, 0]))
    expected_input = []
    expected_target = []
    expected_sigma = []
    for slice_index in range(2):
        input_indices, target_indices = sample_disjoint_groups(images.shape[0], rng)
        expected_input.append(images[input_indices, slice_index].mean(0)[2:14, 2:16])
        expected_target.append(images[target_indices, slice_index].mean(0)[2:14, 2:16])
        expected_sigma.append(sigma_for_average(sigma, input_indices, slice_index)[2:14, 2:16])
    np.testing.assert_allclose(sample["input"].numpy(), np.stack(expected_input))
    np.testing.assert_allclose(sample["target"].numpy(), np.stack(expected_target))
    np.testing.assert_allclose(sample["reference"].numpy(), images[:, :2, 2:14, 2:16].mean(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sample["sigma"].numpy(), np.stack(expected_sigma))


def test_legacy_kspace_fallback(tmp_path):
    path = tmp_path / "legacy_case.mat"
    images, sigma = make_case(path, key="kspace")
    loaded, loaded_sigma = read_case(path)
    np.testing.assert_allclose(loaded, images)
    np.testing.assert_allclose(loaded_sigma, sigma)
    dataset = Rep2RepShuffleDataset(path, depth=2, crop_size=(12, 14), training=False, seed=0)
    sample = dataset[0]
    assert sample["input"].shape == (2, 12, 14)


def test_random_groups_are_nonempty_disjoint_and_can_leave_reps_unused():
    found_unused = False
    for seed in range(100):
        input_indices, target_indices = sample_disjoint_groups(6, np.random.default_rng(seed))
        assert input_indices.size > 0 and target_indices.size > 0
        assert np.intersect1d(input_indices, target_indices).size == 0
        found_unused |= np.union1d(input_indices, target_indices).size < 6
    assert found_unused


def test_shared_and_per_rep_sigma():
    shared = np.full((2, 3, 4), 6.0, np.float32)
    np.testing.assert_allclose(sigma_for_average(shared, [0, 2, 3]), shared / np.sqrt(3))
    per_rep = np.stack([shared, shared * 2, shared * 3])
    expected = np.sqrt(shared ** 2 + (shared * 3) ** 2) / 2
    np.testing.assert_allclose(sigma_for_average(per_rep, [0, 2]), expected)


def test_model_shapes():
    model_2d = CDLNet2D(K=2, M=4, P=3, s=2, t0=0.01, init=False)
    image_2d = torch.randn(2, 1, 17, 19, dtype=torch.complex64)
    sigma_2d = torch.ones(2, 1, 9, 10)
    assert model_2d(image_2d, sigma_2d)[0].shape == image_2d.shape
    model_3d = CDLNet3D(K=2, M=4, P=(3, 3, 3), s=(1, 2, 2), t0=0.01, init=False)
    image_3d = torch.randn(1, 1, 5, 17, 19, dtype=torch.complex64)
    sigma_3d = torch.ones(1, 1, 5, 9, 10)
    assert model_3d(image_3d, sigma_3d)[0].shape == image_3d.shape


def test_fully_sampled_preprocess():
    rng = np.random.default_rng(5)
    repetitions, slices, coils, size = 2, 1, 3, 16
    image = rng.normal(size=(repetitions, slices, size, size)) + 1j * rng.normal(size=(repetitions, slices, size, size))
    phase = np.exp(1j * np.linspace(0, 1, coils))[:, None, None]
    coil_images = image[:, :, None] * phase[None, None]
    coil_images += 0.01 * (rng.normal(size=coil_images.shape) + 1j * rng.normal(size=coil_images.shape))
    kspace = fft2c(torch.from_numpy(coil_images.astype(np.complex64))).numpy()
    combined, sigma, scale = preprocess_multicoil(kspace, hamming_strength=32, device="cpu")
    assert combined.shape == (repetitions, slices, size, size)
    assert sigma.shape == (slices, size, size)
    assert np.all(np.isfinite(combined))
    assert np.all(np.isfinite(sigma)) and np.all(sigma >= 0)
    assert scale > 0


def test_training_keeps_periodic_checkpoints(tmp_path):
    case = tmp_path / "case.mat"
    make_case(case, height=12, width=12)
    output = tmp_path / "run"
    configuration = {
        "seed": 2,
        "device": "cpu",
        "output": str(output),
        "model": {"name": "CDLNet2D", "K": 2, "M": 3, "P": 3, "s": 2, "t0": 0.01, "adaptive": True, "init": False},
        "data": {"train": [str(case)], "val": [str(case)], "crop_size": 8, "slab_depth": 1},
        "train": {
            "epochs": 1,
            "batch_size": 1,
            "val_batch_size": 1,
            "num_workers": 0,
            "lr": 1e-4,
            "clip_grad": 0.5,
            "val_every": 1,
            "save_every": 50,
            "scheduler": {"name": "cosine", "T_max": 1, "eta_min": 0.0},
        },
    }
    train(configuration)
    assert (output / "checkpoint_epoch_000000.pt").is_file()
    assert (output / "checkpoint_epoch_000001.pt").is_file()
    record = json.loads((output / "history.jsonl").read_text().splitlines()[-1])
    assert "reference_nmse" in record["val"]
    inference = tmp_path / "inference.mat"
    infer(output / "checkpoint_epoch_000001.pt", case, inference, indices="0,1", device="cpu")
    assert inference.is_file()
    loaded, loaded_sigma = read_case(case)
    assert loaded.shape == (4, 3, 12, 12)
    assert loaded_sigma.shape == loaded.shape
