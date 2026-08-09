"""Tests for pitch-to-drilling-axis geometry without mass calculations."""

import numpy as np
import pytest

from bowling_rg.dual_angle_conversion import val_from_pap_and_tangent
from bowling_rg.grip_geometry import (
    GripMeasurements,
    construct_grip_frame,
    solve_hole_entry_points,
)
from bowling_rg.hole_pitch import (
    HolePitch,
    bowling_pitch_to_angles,
    drilling_axis_from_two_component_pitch,
    finger_pitch_basis,
    make_drilled_hole_geometry,
    thumb_pitch_basis,
)
from bowling_rg.models import Handedness
from bowling_rg.sphere_geometry import ball_radius_m, hole_centroid, surface_point

ENTRY = np.array([0.0, 0.0, 1.0])
LONGITUDINAL = np.array([0.0, 1.0, 0.0])
LATERAL = np.array([1.0, 0.0, 0.0])
RADIUS_M = ball_radius_m(8.585)


def test_zero_pitch_is_exactly_radial() -> None:
    axis = drilling_axis_from_two_component_pitch(
        ENTRY, LONGITUDINAL, LATERAL, HolePitch(0, 0)
    )

    np.testing.assert_array_equal(axis, -ENTRY)


def test_three_eighths_pitch_has_expected_tilt() -> None:
    pitch = HolePitch(0.375, 0)
    axis = drilling_axis_from_two_component_pitch(ENTRY, LONGITUDINAL, LATERAL, pitch)

    assert bowling_pitch_to_angles(pitch)[0] == pytest.approx(np.arctan(0.375 / 12))
    assert np.arccos(np.dot(axis, -ENTRY)) == pytest.approx(np.arctan(0.375 / 12))


def test_reverse_pitch_has_equal_opposite_tilt() -> None:
    forward = drilling_axis_from_two_component_pitch(
        ENTRY, LONGITUDINAL, LATERAL, HolePitch(0.375, 0)
    )
    reverse = drilling_axis_from_two_component_pitch(
        ENTRY, LONGITUDINAL, LATERAL, HolePitch(-0.375, 0)
    )

    assert forward[1] == pytest.approx(-reverse[1])
    assert forward[2] == pytest.approx(reverse[2])


def test_lateral_left_and_right_pitch_mirror() -> None:
    right = drilling_axis_from_two_component_pitch(
        ENTRY, LONGITUDINAL, LATERAL, HolePitch(0, 0.25)
    )
    left = drilling_axis_from_two_component_pitch(
        ENTRY, LONGITUDINAL, LATERAL, HolePitch(0, -0.25)
    )

    assert right[0] == pytest.approx(-left[0])
    assert right[2] == pytest.approx(left[2])


def test_combined_pitch_axis_is_normalized() -> None:
    axis = drilling_axis_from_two_component_pitch(
        ENTRY, LONGITUDINAL, LATERAL, HolePitch(0.5, -0.375)
    )

    assert np.linalg.norm(axis) == pytest.approx(1)
    assert np.dot(axis, ENTRY) < 0


def test_combined_pitch_uses_simultaneous_vector_offsets() -> None:
    pitch = HolePitch(0.5, -0.375)
    forward_angle, lateral_angle = bowling_pitch_to_angles(pitch)
    expected = (
        -ENTRY - np.tan(forward_angle) * LONGITUDINAL - np.tan(lateral_angle) * LATERAL
    )
    expected /= np.linalg.norm(expected)

    axis = drilling_axis_from_two_component_pitch(ENTRY, LONGITUDINAL, LATERAL, pitch)

    np.testing.assert_allclose(axis, expected)


def test_zero_pitch_centroid_matches_sphere_geometry() -> None:
    geometry = make_drilled_hole_geometry(
        "thumb",
        ENTRY,
        RADIUS_M,
        1.0,
        3.0,
        HolePitch(0, 0),
        LONGITUDINAL,
        LATERAL,
    )
    expected = hole_centroid(
        surface_point(RADIUS_M, ENTRY),
        -ENTRY,
        geometry.depth_m,
    )

    np.testing.assert_allclose(geometry.centroid_m, expected)


def test_pitched_hole_endpoint_remains_inside_sphere() -> None:
    geometry = make_drilled_hole_geometry(
        "middle",
        ENTRY,
        RADIUS_M,
        31 / 32,
        3.0,
        HolePitch(0.375, -0.25),
        LONGITUDINAL,
        LATERAL,
    )

    assert np.linalg.norm(geometry.endpoint_m) < RADIUS_M


def test_impossible_pitched_hole_raises_value_error() -> None:
    with pytest.raises(ValueError, match="opposite side"):
        make_drilled_hole_geometry(
            "impossible",
            ENTRY,
            RADIUS_M,
            1.0,
            12.0,
            HolePitch(1.0, 1.0),
            LONGITUDINAL,
            LATERAL,
        )


def test_right_left_handed_lateral_bases_mirror() -> None:
    right_thumb = np.array([-0.4, 0.0, np.sqrt(0.84)])
    right_fingers = np.array([0.4, 0.0, np.sqrt(0.84)])
    left_thumb = np.array([0.4, 0.0, np.sqrt(0.84)])
    left_fingers = np.array([-0.4, 0.0, np.sqrt(0.84)])

    right_basis = thumb_pitch_basis(right_thumb, right_fingers)
    left_basis = thumb_pitch_basis(left_thumb, left_fingers)

    for left_vector, right_vector in zip(left_basis, right_basis):
        np.testing.assert_allclose(
            left_vector, [-right_vector[0], -right_vector[1], right_vector[2]]
        )


def test_test_ball_grip_entries_produce_valid_pitched_holes_without_rg() -> None:
    pap = np.array([0.0, 0.0, 1.0])
    val = val_from_pap_and_tangent(pap, [0, 1, 0])
    grip_frame = construct_grip_frame(
        pap, val, 3.875, 0.375, RADIUS_M, Handedness.RIGHT
    )
    measurements = GripMeasurements(3.875, 0.375, 4.5625, 4.625, 0.25)
    entries = solve_hole_entry_points(
        grip_frame, measurements, 0.96875, 0.96875, RADIUS_M
    )
    middle_basis = finger_pitch_basis(
        entries.middle_unit, entries.thumb_unit, entries.ring_unit
    )
    ring_basis = finger_pitch_basis(
        entries.ring_unit, entries.thumb_unit, entries.middle_unit
    )
    finger_midpoint = entries.middle_unit + entries.ring_unit
    finger_midpoint /= np.linalg.norm(finger_midpoint)
    thumb_basis = thumb_pitch_basis(entries.thumb_unit, finger_midpoint)

    holes = [
        make_drilled_hole_geometry(
            "thumb",
            entries.thumb_unit,
            RADIUS_M,
            1.0,
            3.0,
            HolePitch(0.125, 0),
            *thumb_basis,
        ),
        make_drilled_hole_geometry(
            "middle",
            entries.middle_unit,
            RADIUS_M,
            0.96875,
            3.0,
            HolePitch(0.25, -0.125),
            *middle_basis,
        ),
        make_drilled_hole_geometry(
            "ring",
            entries.ring_unit,
            RADIUS_M,
            0.96875,
            3.0,
            HolePitch(0.25, 0.125),
            *ring_basis,
        ),
    ]

    assert all(np.all(np.isfinite(hole.drilling_axis_unit)) for hole in holes)
    assert all(not hasattr(hole, "rg") for hole in holes)
