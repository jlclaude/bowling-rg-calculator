"""Tests for provisional version-1 spherical layout geometry."""

import numpy as np
import pytest

from bowling_rg.layout_geometry import (
    LayoutMarkers,
    build_pap_frame,
    dual_angle_pin_direction,
    marker_angle_at,
    marker_distance_in,
    pap_distance_in,
    psa_direction_from_dual_angle,
    rotate_about_axis,
    solve_asymmetric_markers_from_constraints,
    spherical_triangle_angle_at_vertex,
    surface_direction_from_pap,
    tangent_bearing_deg,
    validate_dual_angle_geometry,
    validate_marker_constraints,
)
from bowling_rg.models import BowlerSpec
from bowling_rg.sphere_geometry import ball_radius_m


def bowler(handedness: str) -> BowlerSpec:
    return BowlerSpec(
        pap_horizontal_in=5.0,
        pap_vertical_in=0.5,
        handedness=handedness,
    )


def test_pap_frame_is_right_handed_and_orthonormal() -> None:
    frame = build_pap_frame(bowler("right"))
    basis = np.column_stack((frame.x_tangent, frame.y_tangent, frame.pap_unit))

    np.testing.assert_allclose(basis.T @ basis, np.identity(3), atol=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1)


def test_handed_frames_mirror_horizontal_tangent() -> None:
    right = build_pap_frame(bowler("right"))
    left = build_pap_frame(bowler("left"))

    np.testing.assert_allclose(left.x_tangent, -right.x_tangent)
    np.testing.assert_allclose(left.y_tangent, -right.y_tangent)
    np.testing.assert_allclose(left.pap_unit, right.pap_unit)


@pytest.mark.parametrize(
    ("bearing", "tangent_name"),
    [(0.0, "y_tangent"), (90.0, "x_tangent")],
)
def test_cardinal_bearing_moves_along_expected_tangent(
    bearing: float, tangent_name: str
) -> None:
    frame = build_pap_frame(bowler("right"))
    radius = ball_radius_m(8.585)
    direction = surface_direction_from_pap(frame, 0.1, bearing, radius)
    tangent_component = direction - np.dot(direction, frame.pap_unit) * frame.pap_unit

    np.testing.assert_allclose(
        tangent_component / np.linalg.norm(tangent_component),
        getattr(frame, tangent_name),
        atol=1e-14,
    )


def test_surface_distance_round_trip() -> None:
    frame = build_pap_frame(bowler("right"))
    radius = ball_radius_m(8.585)
    direction = surface_direction_from_pap(frame, 5.0, 43.0, radius)

    assert pap_distance_in(direction, frame, radius) == pytest.approx(5.0)


@pytest.mark.parametrize("bearing", [0.0, 35.0, 90.0, 179.0, 270.0, 359.0])
def test_bearing_round_trip_away_from_pap(bearing: float) -> None:
    frame = build_pap_frame(bowler("right"))
    direction = surface_direction_from_pap(frame, 3.0, bearing, ball_radius_m(8.585))

    assert tangent_bearing_deg(direction, frame) == pytest.approx(bearing)


def test_rodrigues_rotation_preserves_magnitude() -> None:
    vector = np.array([2.0, -3.0, 4.0])

    rotated = rotate_about_axis(vector, [0, 0, 1], 127.0)

    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(vector))


def test_spherical_triangle_angle_for_orthogonal_axes() -> None:
    angle = spherical_triangle_angle_at_vertex([1, 0, 0], [0, 0, 1], [0, 1, 0])

    assert angle == pytest.approx(90.0)


def test_generated_pin_is_five_inches_from_pap() -> None:
    frame = build_pap_frame(bowler("right"))
    radius = ball_radius_m(8.585)
    pin = dual_angle_pin_direction(frame, 5.0, 35.0, radius)

    assert pap_distance_in(pin, frame, radius) == pytest.approx(5.0)


def test_left_and_right_layout_directions_are_mirrored() -> None:
    radius = ball_radius_m(8.585)
    right = build_pap_frame(bowler("right"))
    left = build_pap_frame(bowler("left"))
    right_pin = dual_angle_pin_direction(right, 5.0, 40.0, radius)
    left_pin = dual_angle_pin_direction(left, 5.0, 40.0, radius)
    right_psa = psa_direction_from_dual_angle(right, right_pin, 55.0)
    left_psa = psa_direction_from_dual_angle(left, left_pin, 55.0)

    np.testing.assert_allclose(left_pin, [-right_pin[0], -right_pin[1], right_pin[2]])
    np.testing.assert_allclose(left_psa, [-right_psa[0], -right_psa[1], right_psa[2]])


def test_dual_angle_validation_accepts_generated_geometry() -> None:
    radius = ball_radius_m(8.585)
    frame = build_pap_frame(bowler("right"))
    pin = dual_angle_pin_direction(frame, 5.0, 35.0, radius)
    psa = psa_direction_from_dual_angle(frame, pin, 60.0)

    assert (
        validate_dual_angle_geometry(frame.pap_unit, pin, psa, 5.0, 60.0, 35.0, radius)
        == ()
    )


def test_dual_angle_validation_catches_mismatched_geometry() -> None:
    radius = ball_radius_m(8.585)
    frame = build_pap_frame(bowler("right"))
    pin = dual_angle_pin_direction(frame, 5.0, 35.0, radius)
    psa = psa_direction_from_dual_angle(frame, pin, 60.0)

    with pytest.raises(ValueError, match="drilling angle"):
        validate_dual_angle_geometry(frame.pap_unit, pin, psa, 5.0, 45.0, 35.0, radius)


def test_layout_geometry_outputs_are_finite() -> None:
    radius = ball_radius_m(8.585)
    frame = build_pap_frame(bowler("left"))
    pin = dual_angle_pin_direction(frame, 4.5, 70.0, radius)
    psa = psa_direction_from_dual_angle(frame, pin, 30.0)
    values = [
        frame.pap_unit,
        frame.x_tangent,
        frame.y_tangent,
        pin,
        psa,
        rotate_about_axis([1, 2, 3], [0, 1, 0], 22),
        pap_distance_in(pin, frame, radius),
        tangent_bearing_deg(pin, frame),
        spherical_triangle_angle_at_vertex(frame.pap_unit, pin, psa),
    ]

    assert all(np.all(np.isfinite(value)) for value in values)


def test_spherical_triangle_reconstructed_from_side_lengths() -> None:
    radius = ball_radius_m(8.585)
    frame = build_pap_frame(bowler("right"))
    original_pin = surface_direction_from_pap(frame, 3.25, 0.0, radius)
    original_psa = surface_direction_from_pap(frame, 4.1, 67.0, radius)
    pin_psa_distance = marker_distance_in(original_pin, original_psa, radius)

    markers = solve_asymmetric_markers_from_constraints(
        frame, 3.25, 4.1, pin_psa_distance, radius
    )

    assert marker_distance_in(
        markers.pin_unit, markers.pap_unit, radius
    ) == pytest.approx(3.25)
    assert marker_distance_in(
        markers.psa_unit, markers.pap_unit, radius
    ) == pytest.approx(4.1)
    assert marker_distance_in(
        markers.pin_unit, markers.psa_unit, radius
    ) == pytest.approx(pin_psa_distance)
    assert (
        validate_marker_constraints(markers, 3.25, 4.1, pin_psa_distance, radius) == ()
    )


def test_constraint_solutions_mirror_between_handed_frames() -> None:
    radius = ball_radius_m(8.585)
    right = solve_asymmetric_markers_from_constraints(
        build_pap_frame(bowler("right")), 3.0, 4.0, 4.5, radius
    )
    left = solve_asymmetric_markers_from_constraints(
        build_pap_frame(bowler("left")), 3.0, 4.0, 4.5, radius
    )

    np.testing.assert_allclose(
        left.pin_unit, [-right.pin_unit[0], -right.pin_unit[1], right.pin_unit[2]]
    )
    np.testing.assert_allclose(
        left.psa_unit, [-right.psa_unit[0], -right.psa_unit[1], right.psa_unit[2]]
    )


@pytest.mark.parametrize(
    "side_lengths",
    [
        (1.0, 1.0, 3.0),
        (10.0, 10.0, 10.0),
    ],
)
def test_impossible_spherical_triangle_is_rejected(
    side_lengths: tuple[float, float, float],
) -> None:
    radius = ball_radius_m(8.585)
    frame = build_pap_frame(bowler("right"))

    with pytest.raises(ValueError, match="triangle|circumference|perimeter"):
        solve_asymmetric_markers_from_constraints(frame, *side_lengths, radius)


def test_marker_angle_uses_spherical_tangents() -> None:
    markers = LayoutMarkers(
        pap_unit=np.array([0.0, 0.0, 1.0]),
        pin_unit=np.array([1.0, 0.0, 0.0]),
        psa_unit=np.array([0.0, 1.0, 0.0]),
    )

    assert marker_angle_at(
        markers.pap_unit, markers.pin_unit, markers.psa_unit
    ) == pytest.approx(90.0)
