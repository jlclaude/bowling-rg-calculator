"""Tests for drilling-input validation and the unimplemented boundary."""

import pytest
from pydantic import ValidationError

from bowling_rg.drilling import calculate_hole_mass_properties
from bowling_rg.models import HoleSpec


def valid_hole() -> HoleSpec:
    """Return a valid cylindrical hole specification."""
    return HoleSpec(
        name="thumb",
        diameter_in=1.0,
        depth_in=2.5,
        axis_direction=(0.0, 0.0, 1.0),
        surface_entry_position=(1.0, 2.0, 3.0),
        removed_material_density_g_cm3=1.2,
    )


def test_hole_accepts_normalized_axis() -> None:
    assert valid_hole().axis_direction == (0.0, 0.0, 1.0)


def test_hole_rejects_non_unit_axis() -> None:
    with pytest.raises(ValidationError, match="unit vector"):
        HoleSpec(
            name="finger",
            diameter_in=0.75,
            depth_in=2.0,
            axis_direction=(0.0, 0.0, 2.0),
            surface_entry_position=(1.0, 2.0, 3.0),
            removed_material_density_g_cm3=1.2,
        )


def test_hole_rejects_wrong_vector_length() -> None:
    with pytest.raises(ValidationError):
        HoleSpec(
            name="finger",
            diameter_in=0.75,
            depth_in=2.0,
            axis_direction=(0.0, 1.0),
            surface_entry_position=(1.0, 2.0, 3.0),
            removed_material_density_g_cm3=1.2,
        )


def test_drilling_physics_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="has not been specified"):
        calculate_hole_mass_properties(valid_hole())
