"""Tests for measured dual-angle marker geometry and conversion status."""

import numpy as np
import pytest

from bowling_rg.dual_angle_conversion import (
    ConversionStatus,
    DualAngleLayout,
    angle_between_great_circle_paths_at_marker,
    derive_constraint_set,
    dual_angle_measurements_from_markers,
    shortest_surface_distance_to_great_circle,
    val_great_circle,
)
from bowling_rg.layout_geometry import build_pap_frame, surface_direction_from_pap
from bowling_rg.models import BowlerSpec
from bowling_rg.sphere_geometry import ball_radius_m


def frame():
    return build_pap_frame(
        BowlerSpec(
            pap_horizontal_in=5.0,
            pap_vertical_in=0.5,
            handedness="right",
        )
    )


def marker_fixture():
    radius = ball_radius_m(8.585)
    pap_frame = frame()
    pin = surface_direction_from_pap(pap_frame, 5.0, 20.0, radius)
    psa = surface_direction_from_pap(pap_frame, 3.5, 100.0, radius)
    return radius, pap_frame.pap_unit, pin, psa


def test_marker_derived_pin_to_pap_matches_known_input() -> None:
    radius, pap, pin, psa = marker_fixture()

    measurements = dual_angle_measurements_from_markers(pap, pin, psa, radius)

    assert measurements.pin_to_pap_in == pytest.approx(5.0)


def test_marker_derived_psa_to_pap_matches_explicit_geometry() -> None:
    radius, pap, pin, psa = marker_fixture()

    measurements = dual_angle_measurements_from_markers(pap, pin, psa, radius)

    assert measurements.psa_to_pap_in == pytest.approx(3.5)


def test_marker_derived_pin_to_psa_matches_explicit_geometry() -> None:
    radius, pap, pin, psa = marker_fixture()
    expected_angle = np.arccos(np.clip(np.dot(pin, psa), -1.0, 1.0))

    measurements = dual_angle_measurements_from_markers(pap, pin, psa, radius)

    assert measurements.pin_to_psa_arc_in == pytest.approx(
        np.rad2deg(expected_angle) / 360.0 * 2.0 * np.pi * radius / 0.0254
    )


def test_shortest_distance_to_val_for_simple_orthogonal_cases() -> None:
    radius = 0.1
    pap_frame = frame()
    pole = val_great_circle(pap_frame, pap_frame.pap_unit, pap_frame.x_tangent)

    assert shortest_surface_distance_to_great_circle(
        [1, 0, 0], pole, radius
    ) == pytest.approx(0)
    assert shortest_surface_distance_to_great_circle(
        [0, 0, 1], pole, radius
    ) == pytest.approx(np.pi * radius / 2)


def test_marker_derived_spherical_angle_at_pin_is_correct() -> None:
    angle = angle_between_great_circle_paths_at_marker([0, 0, 1], [1, 0, 0], [0, 1, 0])

    assert angle == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("marker", "point_a", "point_b"),
    [
        ([0, 0, 0], [1, 0, 0], [0, 1, 0]),
        ([0, 0, 1], [0, 0, 1], [0, 1, 0]),
    ],
)
def test_impossible_or_degenerate_geometry_is_rejected(
    marker: list[int], point_a: list[int], point_b: list[int]
) -> None:
    with pytest.raises(ValueError):
        angle_between_great_circle_paths_at_marker(marker, point_a, point_b)


def test_derive_constraint_set_does_not_invent_missing_constraints() -> None:
    layout = DualAngleLayout(48.0, 5.0, 38.0)

    result = derive_constraint_set(layout)

    assert result.status == ConversionStatus.INSUFFICIENT_DATA
    assert result.constraints.pin_to_pap_in == pytest.approx(5.0)
    assert result.constraints.psa_to_pap_in is None
    assert result.constraints.pin_to_psa_arc_in is None
    assert result.constraints.pin_buffer_in is None
    assert result.constraints.diagnostics
