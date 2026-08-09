"""Tests for effective uniform-density mass calibration."""

import numpy as np
import pytest

from bowling_rg.bowling_physics import build_factory_inertia_tensor
from bowling_rg.calibration import (
    MassCalibrationInput,
    actual_removed_material_mass_g,
    build_calibrated_density_model,
    calibrate_uniform_density,
    hole_volume_m3,
    run_mass_calibrated_drilled_ball,
    total_hole_volume_m3,
)
from bowling_rg.drilled_ball import HardwareMass
from bowling_rg.hole_pitch import HolePitch, make_drilled_hole_geometry
from bowling_rg.models import BallSpec
from bowling_rg.sphere_geometry import ball_radius_m

RADIUS_M = ball_radius_m(8.585)


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


def hole(name: str, entry: np.ndarray, diameter: float = 1.0, depth: float = 3.0):
    reference = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(reference, entry)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    longitudinal = reference - np.dot(reference, entry) * entry
    longitudinal /= np.linalg.norm(longitudinal)
    lateral = np.cross(entry, longitudinal)
    return make_drilled_hole_geometry(
        name,
        entry,
        RADIUS_M,
        diameter,
        depth,
        HolePitch(0, 0),
        longitudinal,
        lateral,
    )


def synthetic_calibration(holes, density: float, hardware_g: float = 0.0):
    removed_g = density * total_hole_volume_m3(holes) * 1000.0
    finished_g = ball().gross_mass_g - removed_g + hardware_g
    return MassCalibrationInput(ball().gross_mass_g, finished_g, hardware_g)


def test_synthetic_known_density_is_recovered_exactly() -> None:
    holes = [hole("thumb", np.array([0.0, 0.0, 1.0]))]
    calibration = synthetic_calibration(holes, 1350.0)

    assert calibrate_uniform_density(holes, calibration) == pytest.approx(1350.0)


def test_hardware_mass_is_included_in_removed_material_calculation() -> None:
    calibration = MassCalibrationInput(6500.0, 6300.0, 50.0)

    assert actual_removed_material_mass_g(calibration) == pytest.approx(250.0)


def test_zero_hardware_calibration() -> None:
    calibration = MassCalibrationInput(6500.0, 6300.0, 0.0)

    assert actual_removed_material_mass_g(calibration) == pytest.approx(200.0)


def test_multiple_holes_recover_one_uniform_density() -> None:
    holes = [
        hole("thumb", np.array([0.0, 0.0, 1.0]), 1.5, 4.0),
        hole("middle", np.array([1.0, 0.0, 0.0]), 31 / 32, 2.25),
    ]
    calibration = synthetic_calibration(holes, 1425.0)

    assert calibrate_uniform_density(holes, calibration) == pytest.approx(1425.0)


def test_calibration_does_not_change_hole_geometry() -> None:
    holes = [hole("thumb", np.array([0.0, 0.0, 1.0]))]
    before = (
        holes[0].entry_position_m.copy(),
        holes[0].drilling_axis_unit.copy(),
        holes[0].centroid_m.copy(),
    )

    build_calibrated_density_model(holes, synthetic_calibration(holes, 1200.0))

    for actual, expected in zip(
        (holes[0].entry_position_m, holes[0].drilling_axis_unit, holes[0].centroid_m),
        before,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_calibration_does_not_change_factory_tensor() -> None:
    holes = [hole("thumb", np.array([0.0, 0.0, 1.0]))]
    before = build_factory_inertia_tensor(ball()).copy()

    run_mass_calibrated_drilled_ball(
        ball(), holes, [], synthetic_calibration(holes, 1200.0)
    )

    np.testing.assert_array_equal(build_factory_inertia_tensor(ball()), before)


def test_finished_mass_after_calibration_matches_actual_mass() -> None:
    holes = [hole("thumb", np.array([0.0, 0.0, 1.0]))]
    hardware = [HardwareMass("hardware", 20.0, holes[0].centroid_m)]
    calibration = synthetic_calibration(holes, 1300.0, 20.0)

    drilled, result = run_mass_calibrated_drilled_ball(
        ball(), holes, hardware, calibration
    )

    assert result.predicted_finished_mass_after_g == pytest.approx(
        calibration.finished_mass_g
    )
    assert result.residual_after_g == pytest.approx(0.0, abs=1e-9)
    assert drilled.finished_mass_kg * 1000 == pytest.approx(calibration.finished_mass_g)


def test_one_measurement_does_not_create_per_hole_densities() -> None:
    holes = [
        hole("a", np.array([0.0, 0.0, 1.0])),
        hole("b", np.array([1.0, 0.0, 0.0])),
    ]
    model = build_calibrated_density_model(holes, synthetic_calibration(holes, 1200.0))

    assert model.per_hole_density_kg_m3 == {}
    assert any("cannot identify per-hole" in item for item in model.diagnostics)


def test_implausible_density_produces_warning_without_rejection() -> None:
    holes = [hole("thumb", np.array([0.0, 0.0, 1.0]))]
    model = build_calibrated_density_model(holes, synthetic_calibration(holes, 3000.0))

    assert model.default_density_kg_m3 == pytest.approx(3000.0)
    assert any("outside the plausible range" in item for item in model.diagnostics)


def test_hardware_total_of_86_grams_is_calibrated_correctly() -> None:
    holes = [hole("thumb", np.array([0.0, 0.0, 1.0]))]
    hardware = [
        HardwareMass("JoPo outer", 16, holes[0].centroid_m),
        HardwareMass("JoPo inner", 46, holes[0].centroid_m),
        HardwareMass("middle insert", 11, holes[0].centroid_m),
        HardwareMass("ring insert", 13, holes[0].centroid_m),
    ]
    calibration = synthetic_calibration(holes, 1250.0, 86.0)

    _, result = run_mass_calibrated_drilled_ball(ball(), holes, hardware, calibration)

    assert sum(item.mass_g for item in hardware) == pytest.approx(86.0)
    assert result.calibrated_uniform_density_kg_m3 == pytest.approx(1250.0)
    assert result.residual_after_g == pytest.approx(0.0, abs=1e-9)


def test_outer_limits_setup_is_declared_without_fake_finished_weight() -> None:
    test_ball = ball()
    nominal_hardware_masses_g = (16.0, 46.0, 11.0, 13.0)

    assert test_ball.model == "Outer Limits Black Hole"
    assert test_ball.low_rg_in == pytest.approx(2.515)
    assert test_ball.total_diff == pytest.approx(0.051)
    assert test_ball.intermediate_diff == pytest.approx(0.014)
    assert sum(nominal_hardware_masses_g) == pytest.approx(86.0)
    # No MassCalibrationInput is created until an actual finished weight exists.


def test_hole_volume_matches_cylinder_formula() -> None:
    modeled = hole("volume", np.array([0.0, 0.0, 1.0]), 1.0, 3.0)

    assert hole_volume_m3(modeled) == pytest.approx(
        np.pi * (modeled.diameter_m / 2) ** 2 * modeled.depth_m
    )
