"""Tests for version-1 bowling factory-spec conversion."""

import numpy as np
import pytest
from pydantic import ValidationError

from bowling_rg.bowling_physics import (
    build_factory_inertia_tensor,
    factory_mass_properties,
    factory_principal_moments,
    factory_principal_properties,
    factory_principal_rgs,
    factory_spec_warnings,
    published_differentials_from_rgs,
)
from bowling_rg.inertia import (
    grams_to_kilograms,
    meters_to_inches,
    moment_to_rg,
    principal_properties,
)
from bowling_rg.mass_properties import center_of_mass
from bowling_rg.models import BallSpec


def outer_limits_black_hole_14lb(**overrides: object) -> BallSpec:
    """Return the published RG values with an explicit gross-mass input."""
    values: dict[str, object] = {
        "manufacturer": "Radical",
        "model": "Outer Limits Black Hole",
        "nominal_weight_lb": 14,
        "gross_mass_g": 6350.3,
        "low_rg_in": 2.515,
        "total_diff": 0.051,
        "intermediate_diff": 0.014,
        "core_type": "asymmetric",
    }
    values.update(overrides)
    return BallSpec(**values)


def test_factory_principal_rgs_apply_version_one_convention() -> None:
    assert factory_principal_rgs(outer_limits_black_hole_14lb()) == pytest.approx(
        (2.515, 2.529, 2.566)
    )


def test_rg_to_moment_round_trip_recovers_factory_rgs() -> None:
    ball = outer_limits_black_hole_14lb()
    mass_kg = grams_to_kilograms(ball.gross_mass_g)
    recovered = tuple(
        meters_to_inches(moment_to_rg(moment, mass_kg))
        for moment in factory_principal_moments(ball)
    )

    assert recovered == pytest.approx(factory_principal_rgs(ball))


def test_factory_mass_properties_have_com_at_origin() -> None:
    properties = factory_mass_properties(outer_limits_black_hole_14lb())

    np.testing.assert_allclose(center_of_mass(properties), np.zeros(3))


def test_factory_tensor_principal_properties_recover_original_rgs() -> None:
    ball = outer_limits_black_hole_14lb()
    mass_kg = grams_to_kilograms(ball.gross_mass_g)
    principal = principal_properties(build_factory_inertia_tensor(ball), mass_kg)
    recovered_rg_in = [meters_to_inches(value) for value in principal.rg_values_m]

    np.testing.assert_allclose(recovered_rg_in, factory_principal_rgs(ball))


def test_published_differentials_recover_original_values() -> None:
    ball = outer_limits_black_hole_14lb()
    differentials = published_differentials_from_rgs(*factory_principal_rgs(ball))

    assert differentials == pytest.approx((ball.total_diff, ball.intermediate_diff))


def test_intermediate_diff_above_total_diff_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        outer_limits_black_hole_14lb(intermediate_diff=0.052)

    invalid = outer_limits_black_hole_14lb().model_copy(
        update={"intermediate_diff": 0.052}
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        factory_principal_rgs(invalid)


def test_nonpositive_rg_is_rejected() -> None:
    with pytest.raises(ValidationError):
        outer_limits_black_hole_14lb(low_rg_in=0)

    invalid = outer_limits_black_hole_14lb().model_copy(update={"low_rg_in": 0})
    with pytest.raises(ValueError, match="positive"):
        factory_principal_rgs(invalid)


def test_radical_outer_limits_black_hole_14lb_factory_properties() -> None:
    ball = outer_limits_black_hole_14lb()
    properties = factory_principal_properties(ball)

    assert properties.low_rg_in == pytest.approx(2.515)
    assert properties.intermediate_rg_in == pytest.approx(2.529)
    assert properties.high_rg_in == pytest.approx(2.566)
    assert properties.total_diff == pytest.approx(0.051)
    assert properties.intermediate_diff == pytest.approx(0.014)
    assert properties.moments_kg_m2 == pytest.approx(factory_principal_moments(ball))


def test_symmetric_core_nonzero_intermediate_diff_has_advisory() -> None:
    ball = outer_limits_black_hole_14lb(core_type="symmetric")

    assert factory_spec_warnings(ball)
