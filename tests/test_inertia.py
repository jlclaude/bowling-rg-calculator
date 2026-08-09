"""Tests for SI inertia-tensor primitives."""

import numpy as np
import pytest
from pydantic import ValidationError

from bowling_rg.inertia import (
    build_undrilled_inertia_tensor,
    cylinder_centroid_inertia,
    rotate_inertia_tensor,
    translate_inertia_tensor,
)
from bowling_rg.models import BallSpec


def ball_spec() -> BallSpec:
    return BallSpec(
        manufacturer="Example",
        model="Control",
        nominal_weight_lb=15,
        gross_mass_g=6803.9,
        low_rg_in=2.49,
        total_diff=0.045,
        intermediate_diff=0.012,
        core_type="asymmetric",
    )


def test_ball_spec_rejects_intermediate_diff_above_total() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        BallSpec(
            manufacturer="Example",
            model="Invalid",
            nominal_weight_lb=15,
            gross_mass_g=6803.9,
            low_rg_in=2.49,
            total_diff=0.045,
            intermediate_diff=0.05,
            core_type="asymmetric",
        )


def test_undrilled_tensor_uses_si_principal_moments() -> None:
    ball = ball_spec()
    expected_rg_m = np.array([2.49, 2.502, 2.535]) * 0.0254
    expected = (ball.gross_mass_g / 1000.0) * expected_rg_m**2
    np.testing.assert_allclose(np.diag(build_undrilled_inertia_tensor(ball)), expected)


def test_parallel_axis_translation() -> None:
    mass = 2.0
    centroidal = cylinder_centroid_inertia(mass, 0.1, 0.5, (0, 0, 1))
    displacement = np.array([0.3, -0.2, 0.1])
    expected_offset = mass * (
        np.dot(displacement, displacement) * np.identity(3)
        - np.outer(displacement, displacement)
    )
    translated = translate_inertia_tensor(centroidal, mass, displacement)
    np.testing.assert_allclose(translated, centroidal + expected_offset)


def test_tensor_rotation_preserves_eigenvalues() -> None:
    tensor = np.diag([1.0, 2.0, 4.0])
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = rotate_inertia_tensor(tensor, rotation)
    np.testing.assert_allclose(np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(tensor))
