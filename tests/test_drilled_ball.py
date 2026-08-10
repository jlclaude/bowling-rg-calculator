"""Tests for finished drilled-ball mass-property composition."""

import numpy as np
import pytest

from bowling_rg.bowling_physics import factory_mass_properties, factory_principal_rgs
from bowling_rg.drilled_ball import (
    HardwareMass,
    MaterialDensityModel,
    add_hardware_masses,
    calculate_drilled_ball,
    removed_mass_properties_for_hole,
)
from bowling_rg.dual_angle_conversion import (
    DualAngleLayout,
    asymmetric_core_markers_from_dual_angle,
)
from bowling_rg.grip_geometry import (
    GripMeasurements,
    construct_grip_frame,
    solve_hole_entry_points,
)
from bowling_rg.hole_pitch import (
    HolePitch,
    finger_pitch_basis,
    make_drilled_hole_geometry,
    thumb_pitch_basis,
)
from bowling_rg.layout_geometry import build_pap_frame
from bowling_rg.mass_properties import combine_mass_properties
from bowling_rg.models import BallSpec, BowlerSpec
from bowling_rg.sphere_geometry import ball_radius_m

RADIUS_M = ball_radius_m(8.585)
DENSITY = MaterialDensityModel(1200.0)
ORIENTATION = {
    "pin_unit": np.array([1.0, 0.0, 0.0]),
    "psa_unit": np.array([0.0, 1.0, 0.0]),
}


def ball() -> BallSpec:
    return BallSpec(
        manufacturer="Radical",
        model="Outer Limits Black Hole",
        nominal_weight_lb=14,
        gross_mass_g=6350.3,
        low_rg_in=2.515,
        total_diff=0.051,
        intermediate_diff=0.014,
        core_type="asymmetric",
    )


def radial_hole(
    name: str, entry: np.ndarray, diameter: float = 1.0, depth: float = 3.0
):
    tangent_a = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(entry, tangent_a)) > 0.9:
        tangent_a = np.array([1.0, 0.0, 0.0])
    tangent_a -= np.dot(tangent_a, entry) * entry
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(entry, tangent_a)
    return make_drilled_hole_geometry(
        name,
        entry,
        RADIUS_M,
        diameter,
        depth,
        HolePitch(0, 0),
        tangent_a,
        tangent_b,
    )


def test_no_holes_or_hardware_recovers_factory_rgs() -> None:
    result = calculate_drilled_ball(ball(), [], DENSITY)

    assert (
        result.low_rg_in,
        result.intermediate_rg_in,
        result.high_rg_in,
    ) == pytest.approx(factory_principal_rgs(ball()))


def test_centered_radial_cylinder_mass_loss_matches_geometry() -> None:
    hole = radial_hole("centered", np.array([0.0, 0.0, 1.0]))
    removed = removed_mass_properties_for_hole(hole, 1200.0)
    result = calculate_drilled_ball(ball(), [hole], DENSITY, **ORIENTATION)

    assert result.removed_mass_kg == pytest.approx(removed.mass_kg)
    assert result.hole_removed_masses["centered"] == pytest.approx(removed.mass_kg)


def test_off_center_hole_moves_com_opposite_removed_material() -> None:
    hole = radial_hole("off-center", np.array([1.0, 0.0, 0.0]), depth=2.0)
    result = calculate_drilled_ball(ball(), [hole], DENSITY, **ORIENTATION)

    assert result.center_of_mass_m[0] < 0


def test_adding_back_removed_cylinder_recovers_factory_properties() -> None:
    factory = factory_mass_properties(ball())
    hole = radial_hole("restore", np.array([1.0, 0.0, 0.0]), depth=2.0)
    removed = removed_mass_properties_for_hole(hole, 1200.0)
    drilled_result = calculate_drilled_ball(ball(), [hole], DENSITY, **ORIENTATION)
    # Recreate the remaining properties directly, then add the exact cylinder.
    from bowling_rg.drilled_ball import remove_holes_from_factory

    remaining, _ = remove_holes_from_factory(factory, [hole], DENSITY)
    restored = combine_mass_properties([remaining, removed])

    assert restored.mass_kg == pytest.approx(factory.mass_kg)
    np.testing.assert_allclose(restored.first_moment_kg_m, factory.first_moment_kg_m)
    np.testing.assert_allclose(
        restored.inertia_about_origin_kg_m2, factory.inertia_about_origin_kg_m2
    )
    assert drilled_result.removed_mass_kg > 0


def test_point_mass_hardware_increases_mass_exactly() -> None:
    factory = factory_mass_properties(ball())
    hardware = [HardwareMass("insert", 23.0, np.array([0.01, 0.0, 0.0]))]

    result = add_hardware_masses(factory, hardware)

    assert result.mass_kg == pytest.approx(factory.mass_kg + 0.023)


def test_finished_rg_values_and_differentials_are_consistent() -> None:
    result = calculate_drilled_ball(
        ball(),
        [radial_hole("hole", np.array([0.0, 0.0, 1.0]))],
        DENSITY,
        **ORIENTATION,
    )

    assert 0 < result.low_rg_in <= result.intermediate_rg_in <= result.high_rg_in
    assert result.total_diff == pytest.approx(result.high_rg_in - result.low_rg_in)
    assert result.intermediate_diff == pytest.approx(
        result.intermediate_rg_in - result.low_rg_in
    )


def test_hole_order_does_not_affect_result() -> None:
    holes = [
        radial_hole("x", np.array([1.0, 0.0, 0.0]), depth=2.0),
        radial_hole("z", np.array([0.0, 0.0, 1.0]), depth=2.5),
    ]

    forward = calculate_drilled_ball(ball(), holes, DENSITY, **ORIENTATION)
    reverse = calculate_drilled_ball(ball(), reversed(holes), DENSITY, **ORIENTATION)

    assert forward.finished_mass_kg == pytest.approx(reverse.finished_mass_kg)
    np.testing.assert_allclose(forward.center_of_mass_m, reverse.center_of_mass_m)
    assert forward.low_rg_in == pytest.approx(reverse.low_rg_in)


def test_hardware_order_does_not_affect_result() -> None:
    hardware = [
        HardwareMass("a", 10, np.array([0.01, 0, 0])),
        HardwareMass("b", 20, np.array([0, 0.02, 0])),
    ]

    forward = calculate_drilled_ball(ball(), [], DENSITY, hardware)
    reverse = calculate_drilled_ball(ball(), [], DENSITY, reversed(hardware))

    assert forward.finished_mass_kg == pytest.approx(reverse.finished_mass_kg)
    np.testing.assert_allclose(forward.center_of_mass_m, reverse.center_of_mass_m)
    assert forward.low_rg_in == pytest.approx(reverse.low_rg_in)


def test_predicted_mass_accounting_is_exact() -> None:
    hole = radial_hole("hole", np.array([0.0, 0.0, 1.0]))
    hardware = [HardwareMass("insert", 15, hole.centroid_m)]
    result = calculate_drilled_ball(
        ball(),
        [hole],
        DENSITY,
        hardware,
        pin_unit=ORIENTATION["pin_unit"],
        psa_unit=ORIENTATION["psa_unit"],
    )

    assert result.finished_mass_kg == pytest.approx(
        result.factory_mass_kg - result.removed_mass_kg + result.added_mass_kg
    )
    assert result.predicted_finished_mass_g == pytest.approx(
        result.finished_mass_kg * 1000
    )


def test_actual_mass_residual_is_reported_without_calibration() -> None:
    result = calculate_drilled_ball(ball(), [], DENSITY, actual_finished_mass_g=6300.0)

    assert result.actual_finished_mass_g == pytest.approx(6300.0)
    assert result.mass_residual_g == pytest.approx(
        6300.0 - result.predicted_finished_mass_g
    )


def test_current_ball_integration_has_consistent_finite_accounting() -> None:
    bowler = BowlerSpec(
        pap_horizontal_in=3.875,
        pap_vertical_in=0.375,
        handedness="right",
    )
    frame = build_pap_frame(bowler)
    layout_geometry = asymmetric_core_markers_from_dual_angle(
        frame, DualAngleLayout(48, 5, 38), RADIUS_M
    )
    grip_measurements = GripMeasurements(3.875, 0.375, 4.5625, 4.625, 0.25)
    grip_frame = construct_grip_frame(
        frame.pap_unit,
        layout_geometry.val_great_circle,
        bowler.pap_horizontal_in,
        bowler.pap_vertical_in,
        RADIUS_M,
        bowler.handedness,
    )
    entries = solve_hole_entry_points(
        grip_frame, grip_measurements, 31 / 32, 31 / 32, RADIUS_M
    )
    finger_midpoint = entries.middle_unit + entries.ring_unit
    finger_midpoint /= np.linalg.norm(finger_midpoint)
    thumb_basis = thumb_pitch_basis(entries.thumb_unit, finger_midpoint)
    middle_basis = finger_pitch_basis(
        entries.middle_unit, entries.thumb_unit, entries.ring_unit
    )
    ring_basis = finger_pitch_basis(
        entries.ring_unit, entries.thumb_unit, entries.middle_unit
    )
    holes = [
        make_drilled_hole_geometry(
            "thumb",
            entries.thumb_unit,
            RADIUS_M,
            1.5,
            4.0,
            HolePitch(0, 0),
            *thumb_basis,
        ),
        make_drilled_hole_geometry(
            "middle",
            entries.middle_unit,
            RADIUS_M,
            31 / 32,
            2.25,
            HolePitch(0, 0),
            *middle_basis,
        ),
        make_drilled_hole_geometry(
            "ring",
            entries.ring_unit,
            RADIUS_M,
            31 / 32,
            2.25,
            HolePitch(0, 0),
            *ring_basis,
        ),
    ]
    hardware = [
        HardwareMass("JoPo outer", 16, holes[0].centroid_m),
        HardwareMass("JoPo inner", 46, holes[0].centroid_m),
        HardwareMass("middle insert", 11, holes[1].centroid_m),
        HardwareMass("ring insert", 13, holes[2].centroid_m),
    ]

    result = calculate_drilled_ball(
        ball(),
        holes,
        DENSITY,
        hardware,
        pin_unit=layout_geometry.pin_unit,
        psa_unit=layout_geometry.psa_unit,
    )

    assert result.finished_mass_kg > 0
    assert result.added_mass_kg == pytest.approx(0.086)
    assert 0 < result.low_rg_in <= result.intermediate_rg_in <= result.high_rg_in
    assert result.total_diff >= 0
    assert result.intermediate_diff >= 0
    assert result.finished_mass_kg == pytest.approx(
        result.factory_mass_kg - result.removed_mass_kg + result.added_mass_kg
    )
    assert np.all(np.isfinite(result.center_of_mass_m))


def test_asymmetric_holes_require_explicit_factory_orientation() -> None:
    modeled_hole = radial_hole("hole", np.array([0.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="PIN and PSA orientation"):
        calculate_drilled_ball(ball(), [modeled_hole], DENSITY)
