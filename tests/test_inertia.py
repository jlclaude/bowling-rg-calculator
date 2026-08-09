"""Tests for inertia boundaries and core ball validation."""

import pytest
from pydantic import ValidationError

from bowling_rg.inertia import build_undrilled_inertia_tensor
from bowling_rg.models import BallSpec


def ball_spec() -> BallSpec:
    """Return a representative valid asymmetric ball specification."""
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


def test_ball_spec_accepts_valid_data() -> None:
    ball = ball_spec()
    assert ball.manufacturer == "Example"
    assert ball.core_type.value == "asymmetric"


def test_ball_spec_rejects_intermediate_diff_above_total() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        BallSpec(
            manufacturer="Example",
            model="Invalid",
            nominal_weight_lb=15,
            gross_mass_g=6803.9,
            low_rg_in=2.49,
            total_diff=0.045,
            intermediate_diff=0.05,
            core_type="asymmetric",
        )


def test_ball_spec_rejects_nonpositive_mass() -> None:
    with pytest.raises(ValidationError):
        BallSpec(
            manufacturer="Example",
            model="Invalid",
            nominal_weight_lb=15,
            gross_mass_g=0,
            low_rg_in=2.49,
            total_diff=0.045,
            intermediate_diff=0.01,
            core_type="asymmetric",
        )


def test_inertia_physics_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="has not been specified"):
        build_undrilled_inertia_tensor(ball_spec())
