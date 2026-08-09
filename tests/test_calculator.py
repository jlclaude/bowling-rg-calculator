"""Tests for supporting models and the high-level calculation boundary."""

import pytest
from pydantic import ValidationError

from bowling_rg.calculator import calculate_finished_ball
from bowling_rg.models import (
    BallSpec,
    BowlerSpec,
    CalculationResult,
    InsertGeometry,
    InsertSpec,
    LayoutSpec,
    StaticWeights,
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


def test_layout_rejects_out_of_range_angle() -> None:
    with pytest.raises(ValidationError):
        LayoutSpec(drilling_angle_deg=0, pin_to_pap_in=4.5, val_angle_deg=30)


def test_insert_requires_unit_orientation() -> None:
    with pytest.raises(ValidationError, match="unit vector"):
        InsertSpec(
            name="thumb slug",
            mass_g=12,
            geometry=InsertGeometry(shape="cylinder", dimensions_in={"diameter": 1.0}),
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
        warnings=["Physics model pending verification"],
    )
    assert len(result.principal_axis_vectors) == 3


def test_result_rejects_blank_warning() -> None:
    with pytest.raises(ValidationError, match="blank"):
        CalculationResult(
            finished_mass_g=6750,
            low_rg=2.48,
            intermediate_rg=2.50,
            high_rg=2.53,
            total_diff=0.05,
            intermediate_diff=0.02,
            principal_axis_vectors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            calculated_center_of_mass=(0.01, -0.02, 0.0),
            warnings=[""],
        )


def test_calculator_physics_is_explicitly_unimplemented() -> None:
    ball = BallSpec(
        manufacturer="Example",
        model="Control",
        nominal_weight_lb=15,
        gross_mass_g=6803.9,
        low_rg_in=2.49,
        total_diff=0.045,
        intermediate_diff=0.0,
        core_type="symmetric",
    )
    with pytest.raises(NotImplementedError, match="has not been specified"):
        calculate_finished_ball(ball, [])
