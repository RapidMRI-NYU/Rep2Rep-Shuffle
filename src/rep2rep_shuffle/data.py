import re
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader, Dataset


EXTENSIONS = {".mat", ".h5", ".hdf5", ".npz"}


def _decode_complex(array):
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return array
    if array.dtype.names:
        names = array.dtype.names
        if "real" in names and "imag" in names:
            return array["real"] + 1j * array["imag"]
        if len(names) >= 2:
            return array[names[0]] + 1j * array[names[1]]
    if array.ndim and array.shape[-1] == 2:
        return array[..., 0] + 1j * array[..., 1]
    return array


def _h5_array(handle, key):
    node = handle[key]
    if isinstance(node, h5py.Group):
        if "real" not in node or "imag" not in node:
            raise ValueError(f"Unsupported HDF5 group at key {key!r}")
        return np.asarray(node["real"]) + 1j * np.asarray(node["imag"])
    return _decode_complex(np.asarray(node))


def read_array(path, key):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        try:
            values = loadmat(path, variable_names=[key])
            if key not in values:
                raise KeyError(f"Missing key {key!r} in {path}")
            return _decode_complex(values[key])
        except NotImplementedError:
            with h5py.File(path, "r") as handle:
                return _h5_array(handle, key)
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            return _h5_array(handle, key)
    if suffix == ".npz":
        with np.load(path) as values:
            if key not in values:
                raise KeyError(f"Missing key {key!r} in {path}")
            return _decode_complex(values[key])
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def save_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        savemat(path, arrays)
    elif suffix == ".npz":
        np.savez_compressed(path, **arrays)
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "w") as handle:
            for key, value in arrays.items():
                handle.create_dataset(key, data=value)
    else:
        raise ValueError("Output must end in .mat, .npz, .h5, or .hdf5")


def _natural_key(path):
    return [int(value) if value.isdigit() else value.lower() for value in re.split(r"(\d+)", str(path))]


def discover_files(locations):
    if isinstance(locations, (str, Path)):
        locations = [locations]
    files = []
    for location in locations:
        path = Path(location).expanduser()
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in EXTENSIONS)
        else:
            raise FileNotFoundError(path)
    files = sorted(set(item.resolve() for item in files), key=_natural_key)
    if not files:
        raise ValueError(f"No supported data files found in {locations}")
    return files


def validate_case(images, sigma, source="array"):
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (N,S,H,W), got {images.shape} in {source}")
    if images.shape[0] < 2:
        raise ValueError(f"At least two repetitions are required in {source}")
    if not np.iscomplexobj(images):
        raise TypeError(f"Images must be complex-valued in {source}")
    if sigma.ndim == 3 and sigma.shape != images.shape[1:]:
        raise ValueError(f"Shared sigma map shape {sigma.shape} does not match {images.shape[1:]} in {source}")
    if sigma.ndim == 4 and sigma.shape != images.shape:
        raise ValueError(f"Per-repetition sigma map shape {sigma.shape} does not match {images.shape} in {source}")
    if sigma.ndim not in {3, 4}:
        raise ValueError(f"Expected sigma map with shape (S,H,W) or (N,S,H,W), got {sigma.shape}")
    if np.iscomplexobj(sigma) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError(f"Sigma map must contain finite non-negative real values in {source}")
    if not np.all(np.isfinite(images)):
        raise ValueError(f"Images contain non-finite values in {source}")


DEFAULT_IMAGE_KEYS = ("coil_combined", "images", "kspace")


def read_first_available(path, candidate_keys):
    path = Path(path)
    if isinstance(candidate_keys, str):
        candidate_keys = [candidate_keys]
    checked = []
    for key in candidate_keys:
        try:
            return read_array(path, key), key
        except KeyError:
            checked.append(key)
    raise KeyError(f"None of candidate keys {list(candidate_keys)} were found in {path}. Checked: {checked}")


def read_case(path, image_key="coil_combined", sigma_key="sigma_map"):
    candidate_image_keys = [image_key] if image_key else []
    for fallback in DEFAULT_IMAGE_KEYS:
        if fallback not in candidate_image_keys:
            candidate_image_keys.append(fallback)
    raw_images, _ = read_first_available(path, candidate_image_keys)
    images = _decode_complex(raw_images).astype(np.complex64, copy=False)
    sigma = np.asarray(read_array(path, sigma_key), dtype=np.float32)
    validate_case(images, sigma, path)
    return images, sigma


def sigma_for_average(sigma, indices, slice_index=None):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Repetition group cannot be empty")
    if sigma.ndim == 4:
        selected = sigma[indices] if slice_index is None else sigma[indices, slice_index]
        return np.sqrt(np.sum(selected.astype(np.float32) ** 2, axis=0)) / indices.size
    selected = sigma if slice_index is None else sigma[slice_index]
    return selected.astype(np.float32) / np.sqrt(indices.size)


def _crop_shape(crop_size, height, width):
    if crop_size is None:
        return height, width
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    crop_h, crop_w = map(int, crop_size)
    if crop_h < 1 or crop_w < 1 or crop_h > height or crop_w > width:
        raise ValueError(f"Invalid crop {(crop_h, crop_w)} for image {(height, width)}")
    return crop_h, crop_w


def sample_disjoint_groups(num_repetitions, rng):
    if num_repetitions < 2:
        raise ValueError("At least two repetitions are required")
    while True:
        labels = rng.integers(0, 3, size=num_repetitions)
        input_indices = np.flatnonzero(labels == 1)
        target_indices = np.flatnonzero(labels == 2)
        if input_indices.size and target_indices.size:
            return input_indices, target_indices


class Rep2RepShuffleDataset(Dataset):
    def __init__(
        self,
        locations,
        depth=1,
        crop_size=128,
        training=True,
        seed=0,
        image_key="coil_combined",
        sigma_key="sigma_map",
        slab_depth=None,
    ):
        self.files = discover_files(locations)
        if slab_depth is not None:
            depth = slab_depth
        self.depth = depth
        self.crop_size = crop_size
        self.training = bool(training)
        self.seed = int(seed)
        self.image_key = image_key
        self.sigma_key = sigma_key
        self.epoch = 0

    def __len__(self):
        return len(self.files)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng(self, index):
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, int(self.training)]))

    def __getitem__(self, index):
        images, sigma = read_case(self.files[index], self.image_key, self.sigma_key)
        _, slices, height, width = images.shape
        depth = slices if self.depth is None else min(int(self.depth), slices)
        rng = self._rng(index)
        start = int(rng.integers(0, slices - depth + 1)) if self.training else (slices - depth) // 2
        crop_h, crop_w = _crop_shape(self.crop_size, height, width)
        if self.training:
            top = int(rng.integers(0, height - crop_h + 1))
            left = int(rng.integers(0, width - crop_w + 1))
        else:
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2
        region = np.s_[top:top + crop_h, left:left + crop_w]
        input_volume = []
        target_volume = []
        input_sigma = []
        for slice_index in range(start, start + depth):
            input_indices, target_indices = sample_disjoint_groups(images.shape[0], rng)
            input_volume.append(images[input_indices, slice_index].mean(axis=0)[region])
            target_volume.append(images[target_indices, slice_index].mean(axis=0)[region])
            input_sigma.append(sigma_for_average(sigma, input_indices, slice_index)[region])
        input_volume = np.stack(input_volume)
        target_volume = np.stack(target_volume)
        input_sigma = np.stack(input_sigma)
        reference = images[:, start:start + depth, region[0], region[1]].mean(axis=0)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(input_volume, dtype=np.complex64)),
            "target": torch.from_numpy(np.ascontiguousarray(target_volume, dtype=np.complex64)),
            "sigma": torch.from_numpy(np.ascontiguousarray(input_sigma, dtype=np.float32)),
            "reference": torch.from_numpy(np.ascontiguousarray(reference, dtype=np.complex64)),
        }


def make_loader(dataset, batch_size, num_workers, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(shuffle),
        generator=generator,
    )
