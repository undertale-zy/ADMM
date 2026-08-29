"""Differentiable loss families used across ADMM experiment rounds 1--36."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from admm_unrolled import complex_from_channels, fast_forward_operator


def normalized_nmse(prediction: Tensor, target: Tensor) -> Tensor:
    numerator = torch.sum(torch.abs(prediction - target) ** 2, dim=(-2, -1))
    denominator = torch.sum(torch.abs(target) ** 2, dim=(-2, -1)).clamp_min(1e-8)
    return torch.mean(numerator / denominator)


def _complex_inputs(
    prediction: Tensor, target: Tensor, measurements: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    predicted = complex_from_channels(prediction)
    target_image = complex_from_channels(target)
    observed = complex_from_channels(measurements)
    shape = (int(observed.shape[-2]), int(observed.shape[-1]))
    predicted_echo = fast_forward_operator(predicted, shape)
    clean_echo = fast_forward_operator(target_image, shape)
    return predicted, target_image, observed, predicted_echo, clean_echo


def background_penalty(
    prediction: Tensor,
    target: Tensor,
    *,
    include_l1: bool,
) -> Tensor:
    support = (torch.abs(target) > 1e-6).to(prediction.real.dtype)[:, None]
    protected = F.max_pool2d(support, kernel_size=3, stride=1, padding=1)[:, 0]
    background = 1.0 - protected
    target_energy = torch.sum(torch.abs(target) ** 2, dim=(-2, -1)).clamp_min(1e-8)
    l2 = torch.mean(
        torch.sum(torch.abs(prediction) ** 2 * background, dim=(-2, -1))
        / target_energy
    )
    if not include_l1:
        return l2
    target_rms = torch.sqrt(torch.mean(torch.abs(target) ** 2, dim=(-2, -1))).clamp_min(1e-8)
    l1 = torch.mean(
        torch.mean(torch.abs(prediction) * background, dim=(-2, -1))
        / target_rms
    )
    return l2 + l1


def observed_echo_loss(
    prediction: Tensor,
    target: Tensor,
    measurements: Tensor,
    *,
    echo_weight: float,
    sparse_weight: float = 0.0,
    background_weight: float = 0.0,
    background_l1: bool = False,
) -> dict[str, Tensor]:
    predicted, target_image, observed, predicted_echo, _ = _complex_inputs(
        prediction, target, measurements
    )
    image = normalized_nmse(predicted, target_image)
    echo = normalized_nmse(predicted_echo, observed)
    sparse = torch.mean(torch.abs(predicted))
    background = background_penalty(
        predicted, target_image, include_l1=background_l1
    ) if background_weight else image.new_zeros(())
    total = image + echo_weight * echo + sparse_weight * sparse + background_weight * background
    return {
        "image": image,
        "echo": echo,
        "observed_residual": echo,
        "sparse": sparse,
        "background": background,
        "total": total,
    }


def noise_aware_loss(
    prediction: Tensor,
    target: Tensor,
    measurements: Tensor,
    *,
    clean_weight: float,
    discrepancy_weight: float = 0.1,
    background_weight: float = 0.1,
    support_weight: float = 0.0,
) -> dict[str, Tensor]:
    predicted, target_image, observed, predicted_echo, clean_echo = _complex_inputs(
        prediction, target, measurements
    )
    image = normalized_nmse(predicted, target_image)
    clean = normalized_nmse(predicted_echo, clean_echo)
    observed_residual = normalized_nmse(predicted_echo, observed)
    observed_energy = torch.sum(torch.abs(observed) ** 2, dim=(-2, -1)).clamp_min(1e-8)
    true_noise = torch.sum(torch.abs(observed - clean_echo) ** 2, dim=(-2, -1)) / observed_energy
    predicted_noise = torch.sum(torch.abs(predicted_echo - observed) ** 2, dim=(-2, -1)) / observed_energy
    discrepancy = torch.mean((predicted_noise - true_noise) ** 2)
    background = background_penalty(predicted, target_image, include_l1=True)

    support = (torch.abs(target_image) > 1e-6).to(predicted.real.dtype)[:, None]
    protected = F.max_pool2d(support, kernel_size=3, stride=1, padding=1)[:, 0]
    target_energy = torch.sum(torch.abs(target_image) ** 2, dim=(-2, -1)).clamp_min(1e-8)
    support_energy = torch.sum(
        torch.abs(predicted) ** 2 * protected, dim=(-2, -1)
    ) / target_energy
    support_loss = torch.mean(torch.relu(0.05 - support_energy)) if support_weight else image.new_zeros(())
    total = (
        image
        + clean_weight * clean
        + discrepancy_weight * discrepancy
        + background_weight * background
        + support_weight * support_loss
    )
    return {
        "image": image,
        "clean_echo": clean,
        "echo": clean,
        "observed_residual": observed_residual,
        "true_noise_residual": torch.mean(true_noise),
        "discrepancy": discrepancy,
        "background": background,
        "support": support_loss,
        "total": total,
    }


def detached_metrics(losses: Mapping[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


__all__ = [
    "background_penalty",
    "detached_metrics",
    "noise_aware_loss",
    "normalized_nmse",
    "observed_echo_loss",
]
