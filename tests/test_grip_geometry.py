"""Tests for pitch-free bowling grip surface geometry."""

import numpy as np
import pytest

from bowling_rg.dual_angle_conversion import val_from_pap_and_tangent
from bowling_rg.grip_geometry import (
    GripMeasurements,
    construct_grip_frame,
    move_along_great_circle,
    solve_hole_entry_points,
    tangent_of_great_circle_at_point,
    validate_grip_geometry,
)
from bowling_rg.layout_geometry import marker_distance_in
from bowling_rg.sphere_geometry import ball_radius_m

RADIUS_M = ball_radius_m(8.585)
PAP = np.array([0.0, 0.0, 1.0])
VAL = val_from_pap_and_tangent(PAP, [0, 1, 0])


def measurements(**overrides: float) -> GripMeasurements:
    values = {
        "pap_horizontal_in": 3.875,
        "pap_vertical_in": 0.375,
        "middle_span_in": 4.5625,
        "ring_span_in": 4.625,
        "bridge_in": 0.25,
    }
    values.update(overrides)
    return GripMeasurements(**values)


def test_zero_vertical_offset_keeps_midline_crossing_at_pap() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.0, RADIUS_M, "right")

    np.testing.assert_allclose(grip.midline_intersection_unit, PAP)


def test_positive_pap_up_moves_reconstruction_down_val() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")
    val_up = tangent_of_great_circle_at_point(VAL, PAP)
    expected = move_along_great_circle(PAP, val_up, -0.375, RADIUS_M)

    np.testing.assert_allclose(grip.midline_intersection_unit, expected)
    assert np.dot(grip.midline_intersection_unit, val_up) < 0


def test_right_and_left_pap_offsets_mirror() -> None:
    right = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")
    left = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "left")

    np.testing.assert_allclose(
        left.grip_center_unit,
        [
            -right.grip_center_unit[0],
            right.grip_center_unit[1],
            right.grip_center_unit[2],
        ],
    )


def test_pap_horizontal_offset_round_trips() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")

    assert marker_distance_in(
        grip.midline_intersection_unit, grip.grip_center_unit, RADIUS_M
    ) == pytest.approx(3.875)


def test_equal_spans_and_diameters_produce_symmetric_fingers() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")
    values = measurements(middle_span_in=4.6, ring_span_in=4.6)
    entries = solve_hole_entry_points(grip, values, 0.96875, 0.96875, RADIUS_M)
    middle_lateral = np.dot(entries.middle_unit, grip.midline_tangent_at_grip)
    ring_lateral = np.dot(entries.ring_unit, grip.midline_tangent_at_grip)

    assert middle_lateral == pytest.approx(-ring_lateral)
    assert np.dot(
        entries.middle_unit, grip.centerline_tangent_at_grip
    ) == pytest.approx(np.dot(entries.ring_unit, grip.centerline_tangent_at_grip))


def test_unequal_spans_shift_the_corresponding_finger() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")
    entries = solve_hole_entry_points(grip, measurements(), 0.96875, 0.96875, RADIUS_M)

    assert marker_distance_in(
        entries.thumb_unit, entries.ring_unit, RADIUS_M
    ) > marker_distance_in(entries.thumb_unit, entries.middle_unit, RADIUS_M)


def test_quarter_inch_bridge_and_31_32_holes_set_center_spacing() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")
    entries = solve_hole_entry_points(grip, measurements(), 31 / 32, 31 / 32, RADIUS_M)

    assert marker_distance_in(
        entries.middle_unit, entries.ring_unit, RADIUS_M
    ) == pytest.approx(31 / 64 + 0.25 + 31 / 64)


def test_requested_grip_case_produces_finite_valid_geometry() -> None:
    values = measurements()
    grip = construct_grip_frame(
        PAP,
        VAL,
        values.pap_horizontal_in,
        values.pap_vertical_in,
        RADIUS_M,
        "right",
    )
    entries = solve_hole_entry_points(grip, values, 0.96875, 0.96875, RADIUS_M)

    assert (
        validate_grip_geometry(grip, entries, values, 0.96875, 0.96875, RADIUS_M) == ()
    )
    for vector in (
        entries.thumb_unit,
        entries.middle_unit,
        entries.ring_unit,
        entries.grip_center_unit,
    ):
        assert np.all(np.isfinite(vector))
        assert np.linalg.norm(vector) == pytest.approx(1)


def test_impossible_spans_raise_value_error() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")

    with pytest.raises(ValueError, match="triangle|half the sphere"):
        solve_hole_entry_points(
            grip,
            measurements(middle_span_in=0.1, ring_span_in=4.6),
            0.96875,
            0.96875,
            RADIUS_M,
        )


def test_surface_entries_do_not_apply_drilling_pitch() -> None:
    grip = construct_grip_frame(PAP, VAL, 3.875, 0.375, RADIUS_M, "right")
    entries = solve_hole_entry_points(grip, measurements(), 0.96875, 0.96875, RADIUS_M)

    assert not hasattr(entries, "drilling_axis")
    assert not hasattr(entries, "pitch")
