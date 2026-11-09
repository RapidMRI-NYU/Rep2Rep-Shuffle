import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_soft_threshold(x, threshold, eps=1e-16):
    return x * F.relu(x.abs() - threshold) / (x.abs() + eps)


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    value = tuple(int(item) for item in value)
    if len(value) != 2:
        raise ValueError("Expected a pair")
    return value


def _unit_ball(weight):
    norm = torch.linalg.vector_norm(weight, dim=(2, 3), keepdim=True)
    return weight * torch.clamp(norm.reciprocal(), max=1.0)


class CDLNet2D(nn.Module):
    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0.0, adaptive=True):
        super().__init__()
        kernel, stride = _pair(P), _pair(s)
        padding = tuple((size - 1) // 2 for size in kernel)
        output_padding = tuple(size - 1 for size in stride)
        self.A = nn.ModuleList([
            nn.Conv2d(C, M, kernel, stride, padding, bias=False, dtype=torch.complex64)
            for _ in range(K)
        ])
        self.B = nn.ModuleList([
            nn.ConvTranspose2d(
                M, C, kernel, stride, padding, output_padding,
                bias=False, dtype=torch.complex64,
            )
            for _ in range(K)
        ])
        self.D = self.B[0]
        self.t = nn.Parameter(float(t0) * torch.ones(K, 2, M, 1, 1))
        weight = torch.randn(M, C, *kernel, dtype=torch.complex64)
        with torch.no_grad():
            for analysis, synthesis in zip(self.A, self.B):
                analysis.weight.copy_(weight)
                synthesis.weight.copy_(weight.conj())
            self.project()
        self.K, self.s, self.adaptive = int(K), stride, bool(adaptive)

    def forward(self, image, sigma=None):
        mean = image.mean(dim=(1, 2, 3), keepdim=True)
        image = image - mean
        sh, sw = self.s
        pad_h, pad_w = (-image.shape[-2]) % sh, (-image.shape[-1]) % sw
        top, left = pad_h // 2, pad_w // 2
        pad = (left, pad_w - left, top, pad_h - top)
        if any(pad):
            image = F.pad(image, pad, mode="reflect")

        factor = sigma / 255.0 if sigma is not None and self.adaptive else 0
        code = complex_soft_threshold(self.A[0](image), self.t[0, :1] + factor * self.t[0, 1:2])
        for index in range(1, self.K):
            residual = self.B[index](code) - image
            code = complex_soft_threshold(
                code - self.A[index](residual),
                self.t[index, :1] + factor * self.t[index, 1:2],
            )

        output = self.D(code)
        h_end = output.shape[-2] - pad[3] if pad[3] else None
        w_end = output.shape[-1] - pad[1] if pad[1] else None
        output = output[..., pad[2]:h_end, pad[0]:w_end] + mean
        return output, code

    @torch.no_grad()
    def project(self):
        self.t.clamp_(min=0)
        for analysis, synthesis in zip(self.A, self.B):
            analysis.weight.copy_(_unit_ball(analysis.weight))
            synthesis.weight.copy_(_unit_ball(synthesis.weight))
