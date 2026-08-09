"""Bowling-specific conversion of published factory specifications.

Version 1 assumes manufacturers publish ``total_diff`` as ``high_rg - low_rg``
and ``intermediate_diff`` as ``intermediate_rg - low_rg``.  Manufacturer
conventions may differ, so this interpretation is intentionally isolated here
and may become configurable in a later version.

This module only converts factory data.  It does not interpret PAP, layouts,
pin/PSA locations, drilling coordinates, or other manufacturer geometry.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .inertia import (
    grams_to_kilograms,
    inches_to_meters,
    principal_tensor,
    rg_to_moment,
)
from .mass_properties import MassProperties, mass_properties_from_centroid
from .models import BallSpec, CoreType

FloatArray = NDArray[np.float64]


class FactoryPrincipalProperties(NamedTuple):
    """Factory principal RGs, differentials, and SI principal moments."""

    low_rg_in: float
    intermediate_rg_in: float
    high_rg_in: float
    total_diff: float
    intermediate_diff: float
    moments_kg_m2: tuple[float, float, float]


def factory_principal_rgs(ball: BallSpec) -> tuple[float, float, float]:
    """Return factory principal RGs in low/intermediate/high order, in inches.

    Version 1 applies the manufacturer-convention assumption documented at the
    module level: each published differential is an RG difference from low RG.
    """
    _require_ball_spec(ball)
    low_rg = float(ball.low_rg_in)
    intermediate_diff = float(ball.intermediate_diff)
    total_diff = float(ball.total_diff)

    if low_rg <= 0:
        raise ValueError("low_rg_in must be positive")
    if intermediate_diff < 0 or total_diff < 0:
        raise ValueError("factory differentials must be nonnegative")
    if intermediate_diff > total_diff:
        raise ValueError("intermediate_diff cannot exceed total_diff")

    intermediate_rg = low_rg + intermediate_diff
    high_rg = low_rg + total_diff
    if not low_rg <= intermediate_rg <= high_rg:
        raise ValueError(
            "factory principal RGs must satisfy low <= intermediate <= high"
        )
    return low_rg, intermediate_rg, high_rg


def factory_principal_moments(ball: BallSpec) -> tuple[float, float, float]:
    """Return low/intermediate/high factory principal moments in kg·m²."""
    mass_kg = grams_to_kilograms(ball.gross_mass_g)
    return tuple(
        rg_to_moment(mass_kg, inches_to_meters(rg_in))
        for rg_in in factory_principal_rgs(ball)
    )


def factory_principal_properties(ball: BallSpec) -> FactoryPrincipalProperties:
    """Return the complete version-1 interpretation of published factory data."""
    low_rg, intermediate_rg, high_rg = factory_principal_rgs(ball)
    total_diff, intermediate_diff = published_differentials_from_rgs(
        low_rg, intermediate_rg, high_rg
    )
    return FactoryPrincipalProperties(
        low_rg,
        intermediate_rg,
        high_rg,
        total_diff,
        intermediate_diff,
        factory_principal_moments(ball),
    )


def build_factory_inertia_tensor(ball: BallSpec) -> FloatArray:
    """Return the diagonal low/intermediate/high factory tensor in kg·m²."""
    return principal_tensor(*factory_principal_moments(ball))


def factory_mass_properties(ball: BallSpec) -> MassProperties:
    """Return undrilled factory mass properties with COM at the origin."""
    mass_kg = grams_to_kilograms(ball.gross_mass_g)
    return mass_properties_from_centroid(
        mass_kg,
        np.zeros(3, dtype=np.float64),
        build_factory_inertia_tensor(ball),
    )


def published_differentials_from_rgs(
    low_rg: float, intermediate_rg: float, high_rg: float
) -> tuple[float, float]:
    """Return version-1 total and intermediate RG differences."""
    low = _finite_float(low_rg, "low_rg")
    intermediate = _finite_float(intermediate_rg, "intermediate_rg")
    high = _finite_float(high_rg, "high_rg")
    if low <= 0:
        raise ValueError("low_rg must be positive")
    if not low <= intermediate <= high:
        raise ValueError("principal RGs must satisfy low <= intermediate <= high")
    return high - low, intermediate - low


def factory_spec_warnings(ball: BallSpec) -> tuple[str, ...]:
    """Return advisory messages for unusual, but currently accepted, factory data."""
    _require_ball_spec(ball)
    if ball.core_type == CoreType.SYMMETRIC and ball.intermediate_diff != 0:
        return (
            (
                "symmetric core has a nonzero intermediate differential; "
                "the value is accepted under the version-1 factory convention"
            ),
        )
    return ()


def _require_ball_spec(ball: BallSpec) -> None:
    if not isinstance(ball, BallSpec):
        raise TypeError("ball must be a BallSpec")


def _finite_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
