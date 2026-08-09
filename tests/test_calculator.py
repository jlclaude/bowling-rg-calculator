"""End-to-end tests for finished-ball mass properties."""

import numpy as np
import pytest
from pydantic import ValidationError

from bowling_rg.calculator import calculate_finished_ball
from bowling_rg.drilling import calculate_hole_mass_properties
from bowling_rg.models import (
    BallSpec,
    BowlerSpec,
    CalculationResult,
    HoleSpec,
    InsertGeometry,
    InsertSpec,
    LayoutSpec,
    StaticWeights,
)


def ball_spec() -> BallSpec:
    return BallSpec(
        manufacturer="Example",
        model="Control",
        nominal_weight_lb=15,
        gross_mass_g=6803.9,
        low_rg_in=2.49,
        total_diff=0.045,
        intermediate_diff=0.012,
        core_type="asymmetric",
    )


def test_supporting_models_validate_and_preserve_signed_measurements() -> None:
    bowler = BowlerSpec(
        pap_horizontal_in=5.0, pap_vertical_in=-0.5, handedness="right"
    )
    layout = LayoutSpec(drilling_angle_deg=60, pin_to_pap_in=4.5, val_angle_deg=30)
    weights = StaticWeights(
        top_bottom_oz=-0.25,
        positive_negative_side_oz=0.5,
        finger_thumb_oz=-0.125,
    )
    assert bowler.pap_vertical_in == -0.5
    assert layout.pin_to_pap_in == 4.5
    assert weights.top_bottom_oz == -0.25


def test_insert_requires_unit_orientation() -> None:
    with pytest.raises(ValidationError, match="unit vector"):
        InsertSpec(
            name="thumb slug",
            mass_g=12,
            geometry=InsertGeometry(
                shape="cylinder", dimensions_in={"diameter": 1.0, "depth": 2.0}
            ),
            location=(0, 0, 1),
            orientation=(0, 0, 0),
        )


def test_result_requires_three_unit_principal_axes() -> None:
    result = CalculationResult(
        finished_mass_g=6750,
        low_rg=2.48,
        intermediate_rg=2.50,
        high_rg=2.53,
        total_diff=0.05,
        intermediate_diff=0.02,
        principal_axis_vectors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        calculated_center_of_mass=(0.01, -0.02, 0.0),
        warnings=[],
    )
    assert len(result.principal_axis_vectors) == 3


def test_no_drilling_returns_original_values() -> None:
    ball = ball_spec()
    result = calculate_finished_ball(ball, [])
    assert result.finished_mass_g == pytest.approx(ball.gross_mass_g)
    assert result.low_rg == pytest.approx(ball.low_rg_in)
    assert result.intermediate_rg == pytest.approx(
        ball.low_rg_in + ball.intermediate_diff
    )
    assert result.high_rg == pytest.approx(ball.low_rg_in + ball.total_diff)
    assert result.total_diff == pytest.approx(ball.total_diff)
    assert result.intermediate_diff == pytest.approx(ball.intermediate_diff)
    np.testing.assert_allclose(result.calculated_center_of_mass, (0, 0, 0))


def test_removing_centered_cylindrical_hole_reduces_mass() -> None:
    hole = HoleSpec(
        name="center hole",
        diameter_in=1.0,
        depth_in=2.0,
        axis_direction=(0, 0, 1),
        surface_entry_position=(0, 0, -1),
        removed_material_density_g_cm3=1.2,
    )
    removed = calculate_hole_mass_properties(hole)
    result = calculate_finished_ball(ball_spec(), [hole])
    assert result.finished_mass_g == pytest.approx(
        ball_spec().gross_mass_g - removed.mass_kg * 1000
    )
    np.testing.assert_allclose(result.calculated_center_of_mass, (0, 0, 0), atol=1e-15)


def test_adding_back_same_removed_cylinder_recovers_original_ball() -> None:
    hole = HoleSpec(
        name="offset hole",
        diameter_in=1.0,
        depth_in=2.25,
        axis_direction=(0, 1, 0),
        surface_entry_position=(0.4, -1.125, -0.2),
        removed_material_density_g_cm3=1.15,
    )
    removed = calculate_hole_mass_properties(hole)
    centroid_in = np.asarray(hole.surface_entry_position) + (
        0.5 * hole.depth_in * np.asarray(hole.axis_direction)
    )
    insert = InsertSpec(
        name="replacement cylinder",
        mass_g=removed.mass_kg * 1000,
        geometry=InsertGeometry(
            shape="cylinder",
            dimensions_in={"diameter": hole.diameter_in, "depth": hole.depth_in},
        ),
        location=tuple(centroid_in),
        orientation=hole.axis_direction,
    )
    result = calculate_finished_ball(ball_spec(), [hole], [insert])
    original = calculate_finished_ball(ball_spec(), [])

    assert result.finished_mass_g == pytest.approx(original.finished_mass_g)
    assert result.low_rg == pytest.approx(original.low_rg, abs=1e-12)
    assert result.intermediate_rg == pytest.approx(original.intermediate_rg, abs=1e-12)
    assert result.high_rg == pytest.approx(original.high_rg, abs=1e-12)
    np.testing.assert_allclose(result.calculated_center_of_mass, (0, 0, 0), atol=1e-14)


def test_finished_eigenvalues_are_real_and_positive() -> None:
    hole = HoleSpec(
        name="angled hole",
        diameter_in=0.75,
        depth_in=2.5,
        axis_direction=(1 / np.sqrt(2), 1 / np.sqrt(2), 0),
        surface_entry_position=(-2.0, 1.0, 0.5),
        removed_material_density_g_cm3=1.1,
    )
    result = calculate_finished_ball(ball_spec(), [hole])
    radii = np.array([result.low_rg, result.intermediate_rg, result.high_rg])
    assert np.isrealobj(radii)
    assert np.all(radii > 0)
    assert np.all(np.diff(radii) >= 0)
