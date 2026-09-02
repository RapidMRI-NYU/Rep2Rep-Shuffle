import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from tqdm import tqdm

from .data import Rep2RepShuffleDataset, make_loader, read_case, save_arrays, sigma_for_average
from .model import build_model


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def model_dimension(model):
    return 2 if isinstance(model.A[0], torch.nn.Conv2d) else 3


def _reshape(tensor, dimensions, complex_value=False):
    dtype = torch.complex64 if complex_value else torch.float32
    tensor = tensor.to(dtype=dtype)
    if dimensions == 2:
        batch, depth, height, width = tensor.shape
        return tensor.reshape(batch * depth, 1, height, width)
    return tensor.unsqueeze(1)


def prepare_batch(batch, model, device):
    dimensions = model_dimension(model)
    return {
        "input": _reshape(batch["input"], dimensions, True).to(device, non_blocking=True),
        "target": _reshape(batch["target"], dimensions, True).to(device, non_blocking=True),
        "sigma": _reshape(batch["sigma"], dimensions).to(device, non_blocking=True),
        "reference": _reshape(batch["reference"], dimensions, True).to(device, non_blocking=True),
    }


def prepare_sigma(sigma, image, model):
    convolution = model.A[0]
    target = []
    for length, kernel, stride, padding, dilation in zip(
        image.shape[2:],
        convolution.kernel_size,
        convolution.stride,
        convolution.padding,
        convolution.dilation,
    ):
        padded = ((length + stride - 1) // stride) * stride
        target.append((padded + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)
    if tuple(sigma.shape[2:]) != tuple(target):
        mode = "bilinear" if len(target) == 2 else "trilinear"
        sigma = F.interpolate(sigma, size=target, mode=mode, align_corners=False)
    return sigma * 255.0


def complex_mse(estimate, target):
    return (estimate - target).abs().square().mean()


def _slice_metrics(estimate, reference):
    estimate = estimate.detach().abs().float().cpu().numpy()
    reference = reference.detach().abs().float().cpu().numpy()
    estimate = estimate.reshape(-1, *estimate.shape[-2:])
    reference = reference.reshape(-1, *reference.shape[-2:])
    psnr_values = []
    ssim_values = []
    for current, desired in zip(estimate, reference):
        data_range = float(desired.max() - desired.min())
        if data_range <= 0:
            continue
        error = float(np.mean((current - desired) ** 2))
        psnr_values.append(float("inf") if error == 0 else 20 * math.log10(data_range / math.sqrt(error)))
        window = min(7, desired.shape[-2], desired.shape[-1])
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            ssim_values.append(float(structural_similarity(desired, current, data_range=data_range, win_size=window)))
    return psnr_values, ssim_values


def run_epoch(loader, model, device, optimizer=None, clip_grad=None, description="train"):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    elements = 0
    error_energy = 0.0
    reference_energy = 0.0
    psnr_values = []
    ssim_values = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in tqdm(loader, desc=description, leave=False):
            batch = prepare_batch(raw_batch, model, device)
            sigma = prepare_sigma(batch["sigma"], batch["input"], model)
            if training:
                optimizer.zero_grad(set_to_none=True)
            estimate, _ = model(batch["input"], sigma)
            loss = complex_mse(estimate, batch["target"])
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                optimizer.step()
                model.project()
            count = batch["target"].numel()
            loss_sum += float(loss.item()) * count
            elements += count
            if not training:
                difference = estimate - batch["reference"]
                error_energy += float(difference.abs().square().sum().item())
                reference_energy += float(batch["reference"].abs().square().sum().item())
                current_psnr, current_ssim = _slice_metrics(estimate, batch["reference"])
                psnr_values.extend(current_psnr)
                ssim_values.extend(current_ssim)
    result = {"loss": loss_sum / max(elements, 1)}
    if not training:
        result.update({
            "reference_nmse": error_energy / max(reference_energy, 1e-16),
            "reference_psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
            "reference_ssim": float(np.mean(ssim_values)) if ssim_values else float("nan"),
        })
    return result


def _scheduler(optimizer, configuration):
    name = configuration.get("name", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configuration["T_max"]),
            eta_min=float(configuration.get("eta_min", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(configuration["step_size"]),
            gamma=float(configuration["gamma"]),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def save_checkpoint(path, epoch, model, optimizer, scheduler, configuration):
    torch.save(
        {
            "epoch": int(epoch),
            "method": "rep2rep-shuffle",
            "model_configuration": configuration["model"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def train(configuration, resume=None):
    seed = int(configuration.get("seed", 0))
    seed_everything(seed)
    device = select_device(configuration.get("device", "auto"))
    model_configuration = dict(configuration["model"])
    if resume:
        model_configuration["init"] = False
    model = build_model(model_configuration).to(device)
    train_configuration = configuration["train"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_configuration["lr"]))
    scheduler = _scheduler(optimizer, train_configuration["scheduler"])
    start_epoch = 1
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
    data_configuration = configuration["data"]
    dataset_options = {
        "depth": data_configuration.get("slab_depth", data_configuration.get("depth", 1)),
        "crop_size": data_configuration.get("crop_size", 128),
        "seed": seed,
        "image_key": data_configuration.get("image_key", "coil_combined"),
        "sigma_key": data_configuration.get("sigma_key", "sigma_map"),
    }
    train_dataset = Rep2RepShuffleDataset(data_configuration["train"], training=True, **dataset_options)
    val_dataset = Rep2RepShuffleDataset(data_configuration["val"], training=False, **dataset_options)
    train_loader = make_loader(
        train_dataset,
        train_configuration.get("batch_size", 1),
        train_configuration.get("num_workers", 4),
        True,
        seed,
    )
    val_loader = make_loader(
        val_dataset,
        train_configuration.get("val_batch_size", 1),
        train_configuration.get("num_workers", 4),
        False,
        seed + 1,
    )
    output = Path(configuration["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.glob("checkpoint_epoch_*.pt")):
        raise FileExistsError(f"Checkpoint files already exist in {output}")
    with (output / "config.json").open("w") as handle:
        json.dump(configuration, handle, indent=2)
    if not resume:
        save_checkpoint(output / "checkpoint_epoch_000000.pt", 0, model, optimizer, scheduler, configuration)
    epochs = int(train_configuration["epochs"])
    val_every = int(train_configuration.get("val_every", 10))
    save_every = int(train_configuration.get("save_every", 50))
    if val_every < 1 or save_every < 1:
        raise ValueError("val_every and save_every must be positive")
    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        train_result = run_epoch(
            train_loader,
            model,
            device,
            optimizer=optimizer,
            clip_grad=train_configuration.get("clip_grad"),
            description=f"train {epoch}",
        )
        scheduler.step()
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_result}
        if epoch % val_every == 0 or epoch == epochs:
            val_dataset.set_epoch(epoch)
            record["val"] = run_epoch(val_loader, model, device, description=f"val {epoch}")
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=True) + "\n")
        print(json.dumps(record, allow_nan=True))
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                output / f"checkpoint_epoch_{epoch:06d}.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                configuration,
            )
    return model


def _load_inference_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_configuration = dict(state["model_configuration"])
    model_configuration["init"] = False
    model = build_model(model_configuration).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def _parse_indices(value, count):
    if value is None:
        return np.arange(count, dtype=np.int64)
    indices = np.asarray([int(item) for item in value.split(",")], dtype=np.int64)
    if indices.size == 0 or np.unique(indices).size != indices.size or np.any(indices < 0) or np.any(indices >= count):
        raise ValueError(f"Invalid repetition indices for N={count}: {value}")
    return indices


def infer(checkpoint, input_path, output_path, indices=None, image_key="kspace", sigma_key="sigma_map", device="auto", batch_size=8):
    selected_device = select_device(device)
    model, state = _load_inference_model(checkpoint, selected_device)
    images, sigma = read_case(input_path, image_key, sigma_key)
    indices = _parse_indices(indices, images.shape[0])
    input_average = images[indices].mean(axis=0).astype(np.complex64)
    input_sigma = sigma_for_average(sigma, indices).astype(np.float32)
    dimensions = model_dimension(model)
    with torch.no_grad():
        if dimensions == 2:
            outputs = []
            for start in range(0, input_average.shape[0], int(batch_size)):
                image = torch.from_numpy(input_average[start:start + batch_size]).unsqueeze(1).to(selected_device)
                sigma_batch = torch.from_numpy(input_sigma[start:start + batch_size]).unsqueeze(1).to(selected_device)
                estimate, _ = model(image, prepare_sigma(sigma_batch, image, model))
                outputs.append(estimate.squeeze(1).cpu())
            denoised = torch.cat(outputs).numpy()
        else:
            image = torch.from_numpy(input_average).unsqueeze(0).unsqueeze(0).to(selected_device)
            sigma_batch = torch.from_numpy(input_sigma).unsqueeze(0).unsqueeze(0).to(selected_device)
            estimate, _ = model(image, prepare_sigma(sigma_batch, image, model))
            denoised = estimate[0, 0].cpu().numpy()
    save_arrays(
        output_path,
        {
            "denoised": denoised.astype(np.complex64),
            "input_average": input_average,
            "sigma_map": input_sigma,
            "rep_indices": indices,
            "checkpoint_epoch": np.int64(state["epoch"]),
        },
    )


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train Rep2Rep with per-slice random disjoint repetition subsets")
    parser.add_argument("config")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    with Path(args.config).open() as handle:
        configuration = json.load(handle)
    train(configuration, args.resume)


def infer_main(argv=None):
    parser = argparse.ArgumentParser(description="Denoise averaged complex MRI repetitions")
    parser.add_argument("checkpoint")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--reps", help="Comma-separated zero-based repetition indices; default uses all")
    parser.add_argument("--image-key", default="coil_combined", help="Key for image-domain data (default: 'coil_combined', auto-falls back to 'kspace')")
    parser.add_argument("--sigma-key", default="sigma_map")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    infer(
        args.checkpoint,
        args.input,
        args.output,
        indices=args.reps,
        image_key=args.image_key,
        sigma_key=args.sigma_key,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    train_main()
