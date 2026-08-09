"""Tests for generic rigid-body mass-property composition."""

import numpy as np
import pytest

from bowling_rg.inertia import solid_cylinder_centroid_tensor
from bowling_rg.mass_properties import (
    center_of_mass,
    combine_mass_properties,
    cylinder_mass_from_density,
    cylinder_mass_properties,
    inertia_at_center_of_mass,
    mass_properties_from_centroid,
    point_mass_properties,
    subtract_mass_properties,
)


def test_single_centered_object_preserves_com_and_tensor() -> None:
    tensor = np.diag([1.0, 2.0, 3.0])
    properties = mass_properties_from_centroid(4.0, np.zeros(3), tensor)

    np.testing.assert_allclose(center_of_mass(properties), np.zeros(3))
    np.testing.assert_allclose(inertia_at_center_of_mass(properties), tensor)


def test_equal_opposed_point_masses_have_zero_com() -> None:
    properties = combine_mass_properties(
        [
            point_mass_properties(2.0, [1.5, 0, 0]),
            point_mass_properties(2.0, [-1.5, 0, 0]),
        ]
    )

    np.testing.assert_allclose(center_of_mass(properties), np.zeros(3))


def test_unequal_point_masses_have_hand_calculated_com() -> None:
    properties = combine_mass_properties(
        [point_mass_properties(2.0, [1, 0, 0]), point_mass_properties(6.0, [3, 0, 0])]
    )

    np.testing.assert_allclose(center_of_mass(properties), [2.5, 0, 0])


def test_combine_then_subtract_component_recovers_base() -> None:
    base = cylinder_mass_properties(0.2, 0.7, 800.0, [-0.1, 0.2, 0.3], [0, 0, 1])
    component = point_mass_properties(0.4, [0.6, -0.2, 0.1])

    recovered = subtract_mass_properties(
        combine_mass_properties([base, component]), component
    )

    assert recovered.mass_kg == pytest.approx(base.mass_kg)
    np.testing.assert_allclose(recovered.first_moment_kg_m, base.first_moment_kg_m)
    np.testing.assert_allclose(
        recovered.inertia_about_origin_kg_m2,
        base.inertia_about_origin_kg_m2,
    )


def test_removing_off_center_material_moves_com_opposite_direction() -> None:
    centered = point_mass_properties(10.0, [0, 0, 0])
    removed = point_mass_properties(1.0, [2.0, 0, 0])

    remaining = subtract_mass_properties(centered, removed)

    assert center_of_mass(remaining)[0] == pytest.approx(-2.0 / 9.0)


def test_cylinder_mass_matches_density_times_volume() -> None:
    radius, length, density = 0.13, 0.42, 780.0
    expected = density * np.pi * radius**2 * length

    assert cylinder_mass_from_density(radius, length, density) == pytest.approx(
        expected
    )


def test_centered_cylinder_matches_centroid_tensor() -> None:
    radius, length, density = 0.1, 0.4, 900.0
    mass = cylinder_mass_from_density(radius, length, density)
    expected = solid_cylinder_centroid_tensor(mass, radius, length, [0, 1, 0])
    properties = cylinder_mass_properties(radius, length, density, [0, 0, 0], [0, 1, 0])

    np.testing.assert_allclose(properties.inertia_about_origin_kg_m2, expected)


def test_translated_cylinder_recovers_centroid_tensor_at_own_com() -> None:
    radius, length, density = 0.08, 0.3, 1100.0
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    mass = cylinder_mass_from_density(radius, length, density)
    expected = solid_cylinder_centroid_tensor(mass, radius, length, axis)
    properties = cylinder_mass_properties(
        radius, length, density, [0.4, -0.3, 0.2], axis
    )

    np.testing.assert_allclose(
        inertia_at_center_of_mass(properties), expected, atol=1e-14
    )


def test_combination_is_independent_of_order() -> None:
    parts = [
        point_mass_properties(1.0, [1, 2, 3]),
        cylinder_mass_properties(0.1, 0.2, 700.0, [-1, 0, 0.5], [1, 0, 0]),
        point_mass_properties(4.0, [0, -2, 1]),
    ]

    forward = combine_mass_properties(parts)
    reverse = combine_mass_properties(reversed(parts))

    assert forward.mass_kg == pytest.approx(reverse.mass_kg)
    np.testing.assert_allclose(forward.first_moment_kg_m, reverse.first_moment_kg_m)
    np.testing.assert_allclose(
        forward.inertia_about_origin_kg_m2,
        reverse.inertia_about_origin_kg_m2,
    )


def test_composed_and_recovered_tensors_remain_symmetric() -> None:
    properties = combine_mass_properties(
        [
            cylinder_mass_properties(0.1, 0.5, 600.0, [0.3, -0.4, 0.2], [0, 0, 1]),
            point_mass_properties(2.0, [-0.2, 0.7, 0.1]),
        ]
    )

    np.testing.assert_allclose(
        properties.inertia_about_origin_kg_m2,
        properties.inertia_about_origin_kg_m2.T,
    )
    com_tensor = inertia_at_center_of_mass(properties)
    np.testing.assert_allclose(com_tensor, com_tensor.T)


@pytest.mark.parametrize("removed_mass", [5.0, 6.0])
def test_removing_too_much_mass_raises_value_error(removed_mass: float) -> None:
    base = point_mass_properties(5.0, [0, 0, 0])
    removed = point_mass_properties(removed_mass, [1, 0, 0])

    with pytest.raises(ValueError, match="resulting mass"):
        subtract_mass_properties(base, removed)
