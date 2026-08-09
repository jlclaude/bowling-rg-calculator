"""Tests for generic drilled-sphere geometry."""

import numpy as np
import pytest

from bowling_rg.sphere_geometry import (
    angular_separation,
    drilling_axis_from_surface_normal,
    great_circle_distance,
    hole_centroid,
    hole_end_point,
    inward_radial_direction,
    normalize,
    surface_point,
    tangent_basis_at_surface,
    validate_hole_inside_sphere,
)


def test_surface_point_has_radius_magnitude() -> None:
    point = surface_point(0.11, normalize([1, 2, 3]))

    assert np.linalg.norm(point) == pytest.approx(0.11)


def test_inward_radial_direction_opposes_outward_normal() -> None:
    outward = normalize([2, -1, 4])

    np.testing.assert_allclose(inward_radial_direction(0.11 * outward), -outward)


def test_zero_pitch_drilling_goes_through_center() -> None:
    radius = 0.11
    outward = normalize([1, 2, 3])
    tangent, _ = tangent_basis_at_surface(outward, [0, 0, 1])
    axis = drilling_axis_from_surface_normal(outward, tangent, 0)

    np.testing.assert_array_equal(axis, -outward)
    np.testing.assert_allclose(hole_end_point(radius * outward, axis, radius), 0)


def test_radial_hole_with_radius_depth_ends_at_center() -> None:
    radius = 0.108
    entry = surface_point(radius, [0, 1, 0])

    endpoint = hole_end_point(entry, [0, -1, 0], radius)

    np.testing.assert_allclose(endpoint, np.zeros(3))


def test_hole_centroid_is_halfway_between_entry_and_endpoint() -> None:
    entry = np.array([0.1, 0, 0])
    axis = np.array([-1, 0, 0])
    depth = 0.06

    centroid = hole_centroid(entry, axis, depth)
    endpoint = hole_end_point(entry, axis, depth)

    np.testing.assert_allclose(centroid, (entry + endpoint) / 2)


def test_orthogonal_great_circle_distance_is_quarter_circumference() -> None:
    radius = 0.109

    assert great_circle_distance(radius, [1, 0, 0], [0, 1, 0]) == pytest.approx(
        np.pi * radius / 2
    )


def test_orthogonal_vectors_have_90_degree_angular_separation() -> None:
    assert angular_separation([1, 0, 0], [0, 0, 1]) == pytest.approx(90)


def test_tangent_basis_is_orthonormal_and_perpendicular_to_normal() -> None:
    normal = normalize([1, 1, 1])
    tangent_a, tangent_b = tangent_basis_at_surface(normal, [1, -1, 0.5])

    assert np.linalg.norm(tangent_a) == pytest.approx(1)
    assert np.linalg.norm(tangent_b) == pytest.approx(1)
    assert np.dot(tangent_a, tangent_b) == pytest.approx(0, abs=1e-12)
    assert np.dot(normal, tangent_a) == pytest.approx(0, abs=1e-12)
    assert np.dot(normal, tangent_b) == pytest.approx(0, abs=1e-12)


def test_positive_pitch_tilts_axis_by_requested_angle() -> None:
    outward = np.array([0.0, 0.0, 1.0])
    tangent = np.array([1.0, 0.0, 0.0])
    axis = drilling_axis_from_surface_normal(outward, tangent, 17.5)

    assert angular_separation(-outward, axis) == pytest.approx(17.5)


def test_outward_drilling_direction_is_impossible() -> None:
    radius = 0.11

    with pytest.raises(ValueError, match="point inward"):
        validate_hole_inside_sphere(radius, [radius, 0, 0], [1, 0, 0], 0.02)


def test_excessive_depth_crossing_sphere_raises_value_error() -> None:
    radius = 0.11

    with pytest.raises(ValueError, match="opposite side"):
        validate_hole_inside_sphere(
            radius, [radius, 0, 0], [-1, 0, 0], 2 * radius + 0.001
        )


def test_returned_geometry_is_finite() -> None:
    radius = 0.11
    outward = normalize([1, 3, -2])
    tangent_a, tangent_b = tangent_basis_at_surface(outward, [0, 0, 1])
    axis = drilling_axis_from_surface_normal(outward, tangent_a, 12)
    entry = surface_point(radius, outward)
    centroid = hole_centroid(entry, axis, 0.04)
    endpoint = hole_end_point(entry, axis, 0.04)

    validate_hole_inside_sphere(radius, entry, axis, 0.04, 0.005)
    for value in (outward, tangent_a, tangent_b, axis, entry, centroid, endpoint):
        assert np.all(np.isfinite(value))
    assert np.isfinite(great_circle_distance(radius, outward, normalize([0, 1, 0])))
    assert np.isfinite(angular_separation(outward, normalize([0, 1, 0])))
