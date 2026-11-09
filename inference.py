import torch

from training import prepare_sigma


@torch.inference_mode()
def denoise(model, image, sigma_map=None):
    if image.ndim != 4 or not image.is_complex():
        raise ValueError("Expected a complex image shaped [B, C, H, W]")
    model.eval()
    sigma = None if sigma_map is None else prepare_sigma(sigma_map, image, model)
    estimate, _ = model(image, sigma)
    return estimate
