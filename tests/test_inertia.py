"""Tests for generic rigid-body inertia mathematics."""

import numpy as np
import pytest
from pydantic import ValidationError

from bowling_rg.inertia import (
    build_undrilled_inertia_tensor,
    grams_to_kilograms,
    inches_to_meters,
    meters_to_inches,
    moment_to_rg,
    parallel_axis_shift,
    principal_properties,
    principal_tensor,
    rg_to_moment,
    rotate_tensor,
    solid_cylinder_centroid_tensor,
    translate_tensor,
)
from bowling_rg.models import BallSpec


def ball_spec() -> BallSpec:
    """Return a representative valid ball without interpreting its RG fields."""
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


def z_rotation(angle_rad: float) -> np.ndarray:
    """Return a proper rotation about Z for test inputs."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def test_unit_conversion_helpers() -> None:
    assert grams_to_kilograms(2500) == pytest.approx(2.5)
    assert inches_to_meters(1) == pytest.approx(0.0254)
    assert meters_to_inches(0.0254) == pytest.approx(1.0)


def test_rg_moment_round_trip() -> None:
    mass_kg = 6.8
    original_rg_m = 0.0635
    assert moment_to_rg(rg_to_moment(mass_kg, original_rg_m), mass_kg) == pytest.approx(
        original_rg_m
    )


def test_principal_diagonal_tensor_returns_original_eigenvalues() -> None:
    tensor = principal_tensor(0.03, 0.01, 0.02)
    properties = principal_properties(tensor, 5.0)
    np.testing.assert_allclose(properties.eigenvalues, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(
        properties.rg_values_m, np.sqrt(np.array([0.01, 0.02, 0.03]) / 5.0)
    )


def test_rotating_tensor_preserves_eigenvalues() -> None:
    tensor = principal_tensor(0.01, 0.02, 0.04)
    rotated = rotate_tensor(tensor, z_rotation(np.deg2rad(37)))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(tensor), atol=1e-14
    )


def test_parallel_axis_theorem_matches_hand_calculation() -> None:
    shift = parallel_axis_shift(2.0, (3.0, 0.0, 0.0))
    np.testing.assert_allclose(shift, np.diag([0.0, 18.0, 18.0]))


def test_z_aligned_cylinder_has_expected_components() -> None:
    mass, radius, length = 2.0, 0.1, 0.5
    tensor = solid_cylinder_centroid_tensor(mass, radius, length, (0, 0, 1))
    i_axis = 0.5 * mass * radius**2
    i_perpendicular = mass * (3 * radius**2 + length**2) / 12
    np.testing.assert_allclose(
        tensor, np.diag([i_perpendicular, i_perpendicular, i_axis])
    )


def test_rotating_cylinder_axis_preserves_eigenvalues() -> None:
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    z_tensor = solid_cylinder_centroid_tensor(2.0, 0.1, 0.5, (0, 0, 1))
    angled_tensor = solid_cylinder_centroid_tensor(2.0, 0.1, 0.5, axis)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(angled_tensor), np.linalg.eigvalsh(z_tensor), atol=1e-14
    )


def test_translate_then_restore_with_same_center_reference() -> None:
    centroidal = principal_tensor(0.01, 0.02, 0.03)
    displacement = np.array([0.2, -0.1, 0.05])
    translated = translate_tensor(centroidal, 4.0, displacement)
    restored = translated - parallel_axis_shift(4.0, displacement)
    np.testing.assert_allclose(restored, centroidal, atol=1e-15)


@pytest.mark.parametrize(
    "invalid_rotation",
    [
        np.diag([1.0, 1.0, 2.0]),
        np.diag([1.0, 1.0, -1.0]),
        np.identity(2),
    ],
)
def test_invalid_rotation_matrices_raise_errors(invalid_rotation: np.ndarray) -> None:
    with pytest.raises(ValueError, match="rotation"):
        rotate_tensor(principal_tensor(1, 2, 3), invalid_rotation)


def test_non_symmetric_inertia_tensor_raises_error() -> None:
    tensor = np.array([[1.0, 0.1, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    with pytest.raises(ValueError, match="symmetric"):
        principal_properties(tensor, 2.0)


def test_ball_spec_conversion_remains_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="has not been specified"):
        build_undrilled_inertia_tensor(ball_spec())


def test_ball_spec_validation_remains_intact() -> None:
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
