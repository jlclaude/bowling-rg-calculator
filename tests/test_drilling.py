"""Tests for finite-cylinder drilling mass properties."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from bowling_rg.drilling import calculate_hole_mass_properties
from bowling_rg.models import HoleSpec


def centered_hole() -> HoleSpec:
    return HoleSpec(
        name="centered test hole",
        diameter_in=1.0,
        depth_in=2.0,
        axis_direction=(0.0, 0.0, 1.0),
        surface_entry_position=(0.0, 0.0, -1.0),
        removed_material_density_g_cm3=1.2,
    )


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


def test_removing_centered_cylindrical_hole_properties() -> None:
    hole = centered_hole()
    properties = calculate_hole_mass_properties(hole)
    radius_m = 0.5 * 0.0254
    length_m = 2.0 * 0.0254
    expected_mass = 1200.0 * math.pi * radius_m**2 * length_m
    i_axis = 0.5 * expected_mass * radius_m**2
    i_perpendicular = expected_mass * (3 * radius_m**2 + length_m**2) / 12

    assert properties.mass_kg == pytest.approx(expected_mass)
    np.testing.assert_allclose(properties.first_moment_kg_m, np.zeros(3), atol=1e-18)
    np.testing.assert_allclose(
        properties.inertia_kg_m2,
        np.diag([i_perpendicular, i_perpendicular, i_axis]),
    )
