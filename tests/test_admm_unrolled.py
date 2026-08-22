from __future__ import annotations

import numpy as np
import pytest
import torch

from admm_unrolled import (
    UnrolledADMM,
    complex_from_channels,
    complex_soft_threshold,
    complex_to_channels,
    fast_adjoint_operator,
    fast_forward_operator,
)
from train_admm_unrolled import compute_loss
from synthetic_isar_dataset import SyntheticISARDataset, generate_sparse_isar_sample


def test_channel_round_trip_and_operator_shapes() -> None:
    torch.manual_seed(2)
    image = torch.randn(3, 2, 10, 8)
    complex_image = complex_from_channels(image)
    torch.testing.assert_close(complex_to_channels(complex_image), image)
    echo = fast_forward_operator(complex_image, (6, 5))
    assert echo.shape == (3, 6, 5)
    recovered_shape = fast_adjoint_operator(echo, (10, 8)).shape
    assert recovered_shape == (3, 10, 8)


def test_fft_adjoint_inner_product() -> None:
    torch.manual_seed(3)
    image = torch.randn(10, 8, dtype=torch.complex64)
    echo = torch.randn(6, 5, dtype=torch.complex64)
    lhs = torch.sum(torch.conj(fast_forward_operator(image, (6, 5))) * echo)
    rhs = torch.sum(torch.conj(image) * fast_adjoint_operator(echo, (10, 8)))
    torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-5)


def test_network_forward_has_valid_parameters_and_gradients() -> None:
    torch.manual_seed(4)
    model = UnrolledADMM((10, 8), (6, 5), num_layers=3)
    measurements = torch.randn(2, 2, 6, 5, requires_grad=True)
    output = model(measurements)
    assert output.shape == (2, 2, 10, 8)
    assert torch.isfinite(output).all()
    loss = output.square().mean()
    loss.backward()
    parameters = model.parameters_per_layer
    for value in (parameters.c, parameters.tau, parameters.beta):
        assert torch.all(value > 0)
        assert torch.isfinite(value).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_zero_measurements_are_stable() -> None:
    model = UnrolledADMM((8, 6), (4, 3), num_layers=4)
    output = model(torch.zeros(2, 2, 4, 3))
    assert torch.equal(output, torch.zeros_like(output))


def test_complex_soft_threshold() -> None:
    values = torch.tensor([0.0 + 0.0j, 3.0 + 4.0j, 1.0j])
    result = complex_soft_threshold(values, 2.0)
    expected = torch.tensor([0.0 + 0.0j, 1.8 + 2.4j, 0.0 + 0.0j])
    torch.testing.assert_close(result, expected)


def test_dataset_is_deterministic_and_has_distinct_scenes() -> None:
    first = SyntheticISARDataset(3, image_shape=(16, 8), measurement_shape=(8, 4), seed=10)
    second = SyntheticISARDataset(3, image_shape=(16, 8), measurement_shape=(8, 4), seed=10)
    for index in range(3):
        measurements_a, target_a = first[index]
        measurements_b, target_b = second[index]
        torch.testing.assert_close(measurements_a, measurements_b)
        torch.testing.assert_close(target_a, target_b)
    assert not torch.equal(first[0][1], first[1][1])
    assert first[0][0].shape == (2, 8, 4)
    assert first[0][1].shape == (2, 16, 8)


def test_sample_rejects_invalid_scatterer_limits() -> None:
    with pytest.raises(ValueError, match="scatterer limits"):
        generate_sparse_isar_sample(
            (8, 6), (4, 3), rng=np.random.default_rng(1), min_scatterers=0
        )


def test_tiny_training_updates_parameters_and_reduces_loss() -> None:
    torch.manual_seed(9)
    dataset = SyntheticISARDataset(
        4,
        image_shape=(8, 4),
        measurement_shape=(4, 2),
        seed=9,
    )
    measurements = torch.stack([dataset[index][0] for index in range(4)])
    targets = torch.stack([dataset[index][1] for index in range(4)])
    model = UnrolledADMM((8, 4), (4, 2), num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    with torch.no_grad():
        initial_loss = float(compute_loss(model(measurements), targets, measurements)[0])
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_loss(model(measurements), targets, measurements)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(compute_loss(model(measurements), targets, measurements)[0])
    assert final_loss < initial_loss
