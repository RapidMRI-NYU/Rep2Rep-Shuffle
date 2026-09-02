import argparse
from pathlib import Path

import numpy as np
import pywt
import torch
import torch.nn.functional as F

from .data import read_array, save_arrays


def ifft2c(kspace):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def fft2c(image):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def center_crop(image, size):
    if size is None:
        return image
    if isinstance(size, int):
        size = (size, size)
    crop_h, crop_w = map(int, size)
    height, width = image.shape[-2:]
    if crop_h > height or crop_w > width or crop_h < 1 or crop_w < 1:
        raise ValueError(f"Invalid crop {(crop_h, crop_w)} for {(height, width)}")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[..., top:top + crop_h, left:left + crop_w]


def hamming_filter(coil_images, strength=32.0):
    height, width = coil_images.shape[-2:]
    window_h = torch.hamming_window(height, periodic=False, device=coil_images.device)
    window_w = torch.hamming_window(width, periodic=False, device=coil_images.device)
    window = torch.outer(window_h, window_w)
    window = (window / window.max()).pow(float(strength))
    return ifft2c(fft2c(coil_images) * window)


def estimate_sensitivity(coil_images, patch_size=5):
    coils, height, width = coil_images.shape
    patches = F.unfold(
        coil_images.unsqueeze(0),
        kernel_size=(patch_size, patch_size),
        padding=patch_size // 2,
    )
    patches = patches.reshape(1, coils, patch_size * patch_size, -1).permute(0, 3, 1, 2)
    covariance = patches @ patches.conj().transpose(-1, -2)
    _, vectors = torch.linalg.eigh(covariance)
    sensitivity = vectors[0, :, :, -1]
    anchor = coil_images.abs().sum(dim=(-2, -1)).argmax()
    sensitivity = sensitivity * torch.exp(-1j * torch.angle(sensitivity[:, anchor])).unsqueeze(-1)
    return sensitivity.reshape(height, width, coils).permute(2, 0, 1).contiguous()


def _hh_filter(device):
    filters = torch.tensor(pywt.Wavelet("bior4.4").filter_bank, dtype=torch.float32, device=device)[:2]
    first = torch.cat([filters[:1], filters[:1], filters[1:], filters[1:]])
    second = torch.cat([filters, filters])
    bank = torch.einsum("ki,kj->kij", first, second).unsqueeze(0).flip(-2, -1)
    return bank.transpose(0, 1)[3:4]


def estimate_noise_covariance(coil_images):
    if min(coil_images.shape[-2:]) < 10:
        raise ValueError("Noise-map estimation requires spatial dimensions of at least 10")
    kernel = _hh_filter(coil_images.device)
    real = F.conv2d(coil_images.real.unsqueeze(1), kernel, stride=2).squeeze(1)
    imag = F.conv2d(coil_images.imag.unsqueeze(1), kernel, stride=2).squeeze(1)
    coefficients = torch.complex(real, imag).flatten(1)
    coefficients = coefficients - coefficients.mean(dim=1, keepdim=True)
    return coefficients @ coefficients.conj().T / (coefficients.shape[1] - 1)


def combine_coils(coil_images, sensitivity):
    denominator = sensitivity.abs().square().sum(dim=0).clamp_min(1e-16)
    return (sensitivity.conj().unsqueeze(0) * coil_images).sum(dim=1) / denominator


def compute_sigma_map(sensitivity, covariance):
    denominator = sensitivity.abs().square().sum(dim=0).clamp_min(1e-16)
    weights = sensitivity / denominator
    variance = torch.einsum("cyx,cd,dyx->yx", weights.conj(), covariance, weights).real
    return variance.clamp_min(0).sqrt()


def preprocess_multicoil(kspace, crop_size=None, hamming_strength=32.0, use_hamming=True, device="auto"):
    kspace = np.asarray(kspace)
    if kspace.ndim != 5 or not np.iscomplexobj(kspace):
        raise ValueError(f"Expected complex fully sampled k-space (N,S,C,H,W), got {kspace.shape}")
    if kspace.shape[0] < 2:
        raise ValueError("At least two repetitions are required")
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    tensor = torch.from_numpy(np.ascontiguousarray(kspace, dtype=np.complex64)).to(selected_device)
    repetitions, slices = tensor.shape[:2]
    combined = []
    sigma_maps = []
    with torch.no_grad():
        for slice_index in range(slices):
            coil_images = center_crop(ifft2c(tensor[:, slice_index]), crop_size)
            covariance = estimate_noise_covariance(coil_images[0])
            sensitivity_input = coil_images.mean(dim=0)
            if use_hamming:
                sensitivity_input = hamming_filter(sensitivity_input, hamming_strength)
            sensitivity = estimate_sensitivity(sensitivity_input)
            combined.append(combine_coils(coil_images, sensitivity).cpu())
            sigma_maps.append(compute_sigma_map(sensitivity, covariance).cpu())
    images = torch.stack(combined, dim=1).numpy().astype(np.complex64)
    sigma = torch.stack(sigma_maps, dim=0).numpy().astype(np.float32)
    scale = float(np.percentile(np.abs(images), 99.9))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid 99.9th-percentile normalization scale: {scale}")
    return images / scale, sigma / scale, scale


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create coil-combined repetitions and noise maps from fully sampled multicoil k-space")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--input-key", default="kspace", help="Array key for input multicoil k-space")
    parser.add_argument("--output-key", default="coil_combined", help="Array key for output coil-combined complex images")
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--hamming-strength", type=float, default=32.0)
    parser.add_argument("--no-hamming", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    raw = read_array(args.input, args.input_key)
    images, sigma, scale = preprocess_multicoil(
        raw,
        crop_size=args.crop_size,
        hamming_strength=args.hamming_strength,
        use_hamming=not args.no_hamming,
        device=args.device,
    )
    save_data = {
        args.output_key: images,
        "sigma_map": sigma,
        "scale": np.float64(scale),
        "hamming_strength": np.float32(args.hamming_strength),
    }
    if args.output_key != "kspace":
        save_data["kspace"] = images
    save_arrays(Path(args.output), save_data)


if __name__ == "__main__":
    main()
