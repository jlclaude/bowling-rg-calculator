"""Tests for exact spherical dual-angle and VAL geometry."""

import numpy as np
import pytest

from bowling_rg.dual_angle_conversion import (
    ConversionStatus,
    DualAngleLayout,
    asymmetric_core_markers_from_dual_angle,
    construct_val_from_layout,
    derive_constraint_set,
    dual_angle_measurements_from_markers,
    great_circle_through_point_and_tangent,
    pin_buffer_from_geometry,
    shortest_surface_distance_to_great_circle,
    tangent_toward_marker_at_point,
    val_from_pap_and_tangent,
    validate_dual_angle_geometry_exact,
)
from bowling_rg.layout_geometry import build_pap_frame, marker_angle_at
from bowling_rg.models import BowlerSpec
from bowling_rg.sphere_geometry import ball_radius_m


def frame(handedness: str = "right"):
    return build_pap_frame(
        BowlerSpec(
            pap_horizontal_in=5.0,
            pap_vertical_in=0.5,
            handedness=handedness,
        )
    )


def test_val_constructed_through_pap_contains_pap() -> None:
    pap_frame = frame()
    val = val_from_pap_and_tangent(pap_frame.pap_unit, pap_frame.y_tangent)

    assert np.dot(pap_frame.pap_unit, val.pole_unit) == pytest.approx(0, abs=1e-14)


def test_great_circle_pole_is_perpendicular_to_point_and_tangent() -> None:
    circle = great_circle_through_point_and_tangent([0, 0, 1], [1, 0, 0])

    assert np.dot(circle.pole_unit, [0, 0, 1]) == pytest.approx(0)
    assert np.dot(circle.pole_unit, [1, 0, 0]) == pytest.approx(0)


def test_pin_buffer_is_zero_when_pin_lies_on_val() -> None:
    radius = ball_radius_m(8.585)
    val = val_from_pap_and_tangent([0, 0, 1], [0, 1, 0])
    pin = np.array([0.0, 1.0, 0.0])

    assert pin_buffer_from_geometry(pin, val, radius) == pytest.approx(0)


def test_rotating_val_away_from_pin_increases_pin_buffer() -> None:
    radius = ball_radius_m(8.585)
    pap = np.array([0.0, 0.0, 1.0])
    pin = np.array([0.0, 1.0, 0.0])
    aligned = val_from_pap_and_tangent(pap, pin)
    rotated = construct_val_from_layout(pap, pin, 38.0)

    assert pin_buffer_from_geometry(pin, rotated, radius) > pin_buffer_from_geometry(
        pin, aligned, radius
    )


def test_dual_angle_geometry_has_exact_orthogonal_pin_psa() -> None:
    geometry = asymmetric_core_markers_from_dual_angle(
        frame(), DualAngleLayout(48, 5, 38), ball_radius_m(8.585)
    )

    assert np.dot(geometry.pin_unit, geometry.psa_unit) == pytest.approx(0, abs=1e-14)


def test_drilling_angle_round_trip_matches_input() -> None:
    layout = DualAngleLayout(48, 5, 38)
    geometry = asymmetric_core_markers_from_dual_angle(
        frame(), layout, ball_radius_m(8.585)
    )

    assert marker_angle_at(
        geometry.pin_unit, geometry.pap_unit, geometry.psa_unit
    ) == pytest.approx(layout.drilling_angle_deg)


def test_val_angle_round_trip_matches_input() -> None:
    layout = DualAngleLayout(48, 5, 38)
    geometry = asymmetric_core_markers_from_dual_angle(
        frame(), layout, ball_radius_m(8.585)
    )
    pin_tangent = tangent_toward_marker_at_point(geometry.pap_unit, geometry.pin_unit)
    val_tangent = np.cross(geometry.val_great_circle.pole_unit, geometry.pap_unit)

    assert np.rad2deg(
        np.arccos(np.clip(np.dot(pin_tangent, val_tangent), -1, 1))
    ) == pytest.approx(layout.val_angle_deg)


def test_48_by_5_by_38_geometry_is_finite_and_normalized() -> None:
    layout = DualAngleLayout(48, 5, 38)
    radius = ball_radius_m(8.585)
    geometry = asymmetric_core_markers_from_dual_angle(frame(), layout, radius)

    assert validate_dual_angle_geometry_exact(geometry, layout, radius) == ()
    for vector in (
        geometry.pap_unit,
        geometry.pin_unit,
        geometry.psa_unit,
        geometry.val_great_circle.pole_unit,
    ):
        assert np.all(np.isfinite(vector))
        assert np.linalg.norm(vector) == pytest.approx(1)
    assert np.isfinite(geometry.pin_buffer_in)


def test_right_and_left_handed_geometry_mirror() -> None:
    layout = DualAngleLayout(48, 5, 38)
    radius = ball_radius_m(8.585)
    right = asymmetric_core_markers_from_dual_angle(frame("right"), layout, radius)
    left = asymmetric_core_markers_from_dual_angle(frame("left"), layout, radius)

    for left_vector, right_vector in (
        (left.pin_unit, right.pin_unit),
        (left.psa_unit, right.psa_unit),
        (left.val_great_circle.pole_unit, right.val_great_circle.pole_unit),
    ):
        np.testing.assert_allclose(
            left_vector, [-right_vector[0], -right_vector[1], right_vector[2]]
        )


def test_pap_is_never_treated_as_val_pole() -> None:
    geometry = asymmetric_core_markers_from_dual_angle(
        frame(), DualAngleLayout(48, 5, 38), ball_radius_m(8.585)
    )

    assert abs(np.dot(geometry.pap_unit, geometry.val_great_circle.pole_unit)) < 1e-12
    assert not np.allclose(geometry.pap_unit, geometry.val_great_circle.pole_unit)


def test_explicit_val_is_required_for_great_circle_distance_and_pin_buffer() -> None:
    radius = ball_radius_m(8.585)
    with pytest.raises(TypeError, match="GreatCircle"):
        shortest_surface_distance_to_great_circle([1, 0, 0], [0, 0, 1], radius)
    with pytest.raises(TypeError, match="GreatCircle"):
        pin_buffer_from_geometry([1, 0, 0], [0, 0, 1], radius)


def test_marker_measurements_use_explicit_val() -> None:
    layout = DualAngleLayout(48, 5, 38)
    radius = ball_radius_m(8.585)
    geometry = asymmetric_core_markers_from_dual_angle(frame(), layout, radius)

    measurements = dual_angle_measurements_from_markers(
        geometry.pap_unit,
        geometry.pin_unit,
        geometry.psa_unit,
        geometry.val_great_circle,
        radius,
    )

    assert measurements.pin_to_pap_in == pytest.approx(5)
    assert measurements.pin_to_psa_arc_in == pytest.approx(np.pi * radius / 2 / 0.0254)
    assert measurements.pin_buffer_in == pytest.approx(geometry.pin_buffer_in)


def test_derive_constraint_set_still_does_not_invent_sss_constraints() -> None:
    result = derive_constraint_set(DualAngleLayout(48, 5, 38))

    assert result.status == ConversionStatus.INSUFFICIENT_DATA
    assert result.constraints.psa_to_pap_in is None
    assert result.constraints.pin_to_psa_arc_in is None
