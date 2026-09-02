import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_soft_threshold(x, threshold, eps=1e-16):
    return x * F.relu(x.abs() - threshold) / (x.abs() + eps)


def _tuple(value, dimensions):
    if isinstance(value, int):
        return (value,) * dimensions
    value = tuple(int(item) for item in value)
    if len(value) != dimensions:
        raise ValueError(f"Expected {dimensions} values, got {value}")
    return value


def _axis_padding(length, stride):
    total = (-length) % stride
    return total // 2, total - total // 2


def _preprocess_2d(x, stride):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = x - mean
    sh, sw = _tuple(stride, 2)
    top, bottom = _axis_padding(x.shape[-2], sh)
    left, right = _axis_padding(x.shape[-1], sw)
    pad = (left, right, top, bottom)
    if any(pad):
        x = F.pad(x, pad, mode="reflect")
    return x, mean, pad


def _preprocess_3d(x, stride):
    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    x = x - mean
    sd, sh, sw = _tuple(stride, 3)
    front, back = _axis_padding(x.shape[-3], sd)
    top, bottom = _axis_padding(x.shape[-2], sh)
    left, right = _axis_padding(x.shape[-1], sw)
    pad = (left, right, top, bottom, front, back)
    if any(pad):
        x = F.pad(x, pad, mode="reflect")
    return x, mean, pad


def _unpad_2d(x, pad):
    left, right, top, bottom = pad
    return x[..., top:x.shape[-2] - bottom, left:x.shape[-1] - right]


def _unpad_3d(x, pad):
    left, right, top, bottom, front, back = pad
    return x[
        ...,
        front:x.shape[-3] - back,
        top:x.shape[-2] - bottom,
        left:x.shape[-1] - right,
    ]


def _power_method(operator, vector, iterations=200):
    eigenvalue = None
    for _ in range(iterations):
        vector = operator(vector)
        vector = vector / vector.norm().clamp_min(1e-16)
        eigenvalue = (vector.conj() * operator(vector)).sum().real
    value = float(eigenvalue.item())
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid spectral estimate: {value}")
    return value


def _unit_ball(weight, dimensions):
    norm = torch.linalg.vector_norm(weight, dim=dimensions, keepdim=True)
    return weight * torch.clamp(norm.reciprocal(), max=1.0)


class CDLNet2D(nn.Module):
    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0.0, adaptive=True, init=True):
        super().__init__()
        kernel = _tuple(P, 2)
        stride = _tuple(s, 2)
        padding = tuple((item - 1) // 2 for item in kernel)
        output_padding = tuple(item - 1 for item in stride)
        self.A = nn.ModuleList([
            nn.Conv2d(C, M, kernel, stride=stride, padding=padding, bias=False, dtype=torch.complex64)
            for _ in range(K)
        ])
        self.B = nn.ModuleList([
            nn.ConvTranspose2d(
                M, C, kernel, stride=stride, padding=padding,
                output_padding=output_padding, bias=False, dtype=torch.complex64,
            )
            for _ in range(K)
        ])
        self.D = self.B[0]
        self.t = nn.Parameter(float(t0) * torch.ones(K, 2, M, 1, 1))
        weight = torch.randn(M, C, *kernel, dtype=torch.complex64)
        for analysis, synthesis in zip(self.A, self.B):
            analysis.weight.data.copy_(weight)
            synthesis.weight.data.copy_(weight.conj())
        if init:
            with torch.no_grad():
                estimate = _power_method(
                    lambda value: self.D(self.A[0](value)),
                    torch.rand(1, C, 128, 128, dtype=torch.complex64),
                )
                scale = math.sqrt(estimate)
                for analysis, synthesis in zip(self.A, self.B):
                    analysis.weight.data.div_(scale)
                    synthesis.weight.data.div_(scale)
        self.K = int(K)
        self.M = int(M)
        self.P = kernel
        self.s = stride
        self.adaptive = bool(adaptive)

    def forward(self, image, sigma=None):
        image, mean, pad = _preprocess_2d(image, self.s)
        factor = 0 if sigma is None or not self.adaptive else sigma / 255.0
        coefficients = complex_soft_threshold(self.A[0](image), self.t[0, :1] + factor * self.t[0, 1:2])
        for layer in range(1, self.K):
            residual = self.B[layer](coefficients) - image
            coefficients = complex_soft_threshold(
                coefficients - self.A[layer](residual),
                self.t[layer, :1] + factor * self.t[layer, 1:2],
            )
        output = _unpad_2d(self.D(coefficients), pad) + mean
        return output, coefficients

    @torch.no_grad()
    def project(self):
        self.t.clamp_(min=0)
        for analysis, synthesis in zip(self.A, self.B):
            analysis.weight.copy_(_unit_ball(analysis.weight, (2, 3)))
            synthesis.weight.copy_(_unit_ball(synthesis.weight, (2, 3)))


class CDLNet3D(nn.Module):
    def __init__(self, K=3, M=64, P=(3, 7, 7), s=(1, 1, 1), C=1, t0=0.0, adaptive=True, init=True, init_depth=12):
        super().__init__()
        kernel = _tuple(P, 3)
        stride = _tuple(s, 3)
        padding = tuple((item - 1) // 2 for item in kernel)
        output_padding = tuple(item - 1 for item in stride)
        self.A = nn.ModuleList([
            nn.Conv3d(C, M, kernel, stride=stride, padding=padding, bias=False, dtype=torch.complex64)
            for _ in range(K)
        ])
        self.B = nn.ModuleList([
            nn.ConvTranspose3d(
                M, C, kernel, stride=stride, padding=padding,
                output_padding=output_padding, bias=False, dtype=torch.complex64,
            )
            for _ in range(K)
        ])
        self.D = self.B[0]
        self.t = nn.Parameter(float(t0) * torch.ones(K, 2, M, 1, 1, 1))
        weight = torch.randn(M, C, *kernel, dtype=torch.complex64)
        for analysis, synthesis in zip(self.A, self.B):
            analysis.weight.data.copy_(weight)
            synthesis.weight.data.copy_(weight.conj())
        if init:
            with torch.no_grad():
                estimate = _power_method(
                    lambda value: self.D(self.A[0](value)),
                    torch.rand(1, C, int(init_depth), 128, 128, dtype=torch.complex64),
                )
                scale = math.sqrt(estimate)
                for analysis, synthesis in zip(self.A, self.B):
                    analysis.weight.data.div_(scale)
                    synthesis.weight.data.div_(scale)
        self.K = int(K)
        self.M = int(M)
        self.P = kernel
        self.s = stride
        self.adaptive = bool(adaptive)

    def forward(self, image, sigma=None):
        image, mean, pad = _preprocess_3d(image, self.s)
        factor = 0 if sigma is None or not self.adaptive else sigma / 255.0
        coefficients = complex_soft_threshold(self.A[0](image), self.t[0, :1] + factor * self.t[0, 1:2])
        for layer in range(1, self.K):
            residual = self.B[layer](coefficients) - image
            coefficients = complex_soft_threshold(
                coefficients - self.A[layer](residual),
                self.t[layer, :1] + factor * self.t[layer, 1:2],
            )
        output = _unpad_3d(self.D(coefficients), pad) + mean
        return output, coefficients

    @torch.no_grad()
    def project(self):
        self.t.clamp_(min=0)


CDLNet_C = CDLNet2D
CDLNet3D_C = CDLNet3D


def build_model(configuration):
    configuration = dict(configuration)
    name = configuration.pop("name")
    if name in {"CDLNet2D", "CDLNet_C"}:
        return CDLNet2D(**configuration)
    if name in {"CDLNet3D", "CDLNet3D_C"}:
        return CDLNet3D(**configuration)
    raise ValueError(f"Unknown model: {name}")
