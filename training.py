import torch
import torch.nn.functional as F


def prepare_sigma(sigma, image, model):
    conv = model.A[0]
    size = []
    for length, kernel, stride, padding, dilation in zip(
        image.shape[2:], conv.kernel_size, conv.stride, conv.padding, conv.dilation
    ):
        padded = ((length + stride - 1) // stride) * stride
        size.append((padded + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)
    if tuple(sigma.shape[2:]) != tuple(size):
        sigma = F.interpolate(sigma.float(), size=size, mode="bilinear", align_corners=False)
    return sigma * 255.0


def train_step(model, optimizer, input_image, target_image, sigma_map, clip_grad=0.5):
    sigma = prepare_sigma(sigma_map, input_image, model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    estimate, _ = model(input_image, sigma)
    loss = (estimate - target_image).abs().square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
    optimizer.step()
    model.project()
    return loss.detach()


def train_epoch(model, optimizer, batches, make_pair, clip_grad=0.5):
    losses = []
    for repetitions, sigma_map in batches:
        input_image, target_image, input_sigma = make_pair(repetitions, sigma_map)
        losses.append(train_step(model, optimizer, input_image, target_image, input_sigma, clip_grad))
    if not losses:
        raise ValueError("Training batches are empty")
    return torch.stack(losses).mean()
