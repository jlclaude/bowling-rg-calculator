"""Measured geometry and conversion status for dual-angle layouts.

A production dual-angle-to-marker conversion requires a verified industry or
manufacturer convention.  This module deliberately does not infer PSA-to-PAP
or PIN-to-PSA constraints from drilling angle, PIN-to-PAP distance, and VAL
angle.  It provides generic spherical measurements and reports missing data
instead of silently inventing geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inertia import meters_to_inches
from .layout_geometry import (
    BowlingFrame,
    marker_angle_at,
    marker_distance_in,
)

FloatArray = NDArray[np.float64]
_VECTOR_ATOL = 1e-12


@dataclass(frozen=True)
class DualAngleLayout:
    """Published drilling angle × PIN-to-PAP × VAL angle layout input."""

    drilling_angle_deg: float
    pin_to_pap_in: float
    val_angle_deg: float

    def __post_init__(self) -> None:
        drilling = _angle_0_to_180(self.drilling_angle_deg, "drilling_angle_deg")
        pin_to_pap = _positive_scalar(self.pin_to_pap_in, "pin_to_pap_in")
        val = _angle_0_to_180(self.val_angle_deg, "val_angle_deg")
        object.__setattr__(self, "drilling_angle_deg", drilling)
        object.__setattr__(self, "pin_to_pap_in", pin_to_pap)
        object.__setattr__(self, "val_angle_deg", val)


@dataclass(frozen=True)
class LayoutConstraintSet:
    """Explicit marker constraints available for a layout calculation."""

    pin_to_pap_in: float
    psa_to_pap_in: Optional[float] = None  # noqa: UP045 - Python 3.9
    pin_to_psa_arc_in: Optional[float] = None  # noqa: UP045 - Python 3.9
    pin_buffer_in: Optional[float] = None  # noqa: UP045 - Python 3.9
    diagnostics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pin_to_pap_in",
            _positive_scalar(self.pin_to_pap_in, "pin_to_pap_in"),
        )
        for name in ("psa_to_pap_in", "pin_to_psa_arc_in", "pin_buffer_in"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative_scalar(value, name))
        if any(not diagnostic.strip() for diagnostic in self.diagnostics):
            raise ValueError("diagnostics cannot contain blank messages")
        object.__setattr__(self, "diagnostics", list(self.diagnostics))


class ConversionStatus(str, Enum):
    """Confidence/status of a dual-angle constraint derivation."""

    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class ConstraintDerivationResult:
    """Status and explicit constraints produced without hidden assumptions."""

    status: ConversionStatus
    constraints: LayoutConstraintSet


@dataclass(frozen=True)
class MarkerDerivedMeasurements:
    """Directly measurable spherical marker quantities."""

    pin_to_pap_in: float
    psa_to_pap_in: float
    pin_to_psa_arc_in: float
    pin_buffer_in: float
    angle_at_pin_deg: float


def val_great_circle(
    frame: BowlingFrame,
    pap_unit: ArrayLike,
    orientation_reference: ArrayLike,
) -> FloatArray:
    """Return the unit pole representing the VAL great circle.

    In this geometric definition the VAL plane passes through the ball center
    and has PAP as its pole. ``orientation_reference`` establishes that a valid
    tangent direction on the circle was supplied, although an unoriented great
    circle itself is fully represented by its pole.
    """
    if not isinstance(frame, BowlingFrame):
        raise TypeError("frame must be a BowlingFrame")
    pap = _unit_vector3(pap_unit, "pap_unit")
    reference = _unit_vector3(orientation_reference, "orientation_reference")
    if not np.allclose(pap, frame.pap_unit, atol=_VECTOR_ATOL, rtol=1e-12):
        raise ValueError("pap_unit must match the BowlingFrame PAP")
    if not np.isclose(np.dot(pap, reference), 0.0, atol=_VECTOR_ATOL, rtol=0.0):
        raise ValueError("orientation_reference must lie in the VAL great circle")
    return pap.copy()


def shortest_surface_distance_to_great_circle(
    marker_unit: ArrayLike,
    great_circle_pole: ArrayLike,
    ball_radius_m: float,
) -> float:
    """Return shortest marker-to-great-circle surface distance in meters."""
    marker = _unit_vector3(marker_unit, "marker_unit")
    pole = _unit_vector3(great_circle_pole, "great_circle_pole")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    angular_distance = np.arcsin(np.clip(abs(np.dot(marker, pole)), 0.0, 1.0))
    return float(radius * angular_distance)


def pin_buffer_from_geometry(
    pin_unit: ArrayLike,
    pap_unit: ArrayLike,
    ball_radius_m: float,
) -> float:
    """Return shortest PIN-to-VAL surface distance, in inches."""
    distance_m = shortest_surface_distance_to_great_circle(
        pin_unit, pap_unit, ball_radius_m
    )
    return meters_to_inches(distance_m)


def angle_between_great_circle_paths_at_marker(
    marker: ArrayLike,
    path_point_a: ArrayLike,
    path_point_b: ArrayLike,
) -> float:
    """Return the spherical tangent angle between two paths, in degrees."""
    return marker_angle_at(marker, path_point_a, path_point_b)


def dual_angle_measurements_from_markers(
    pap: ArrayLike,
    pin: ArrayLike,
    psa: ArrayLike,
    ball_radius_m: float,
) -> MarkerDerivedMeasurements:
    """Measure marker geometry without claiming a dual-angle conversion."""
    pap_unit = _unit_vector3(pap, "pap")
    pin_unit = _unit_vector3(pin, "pin")
    psa_unit = _unit_vector3(psa, "psa")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    return MarkerDerivedMeasurements(
        pin_to_pap_in=marker_distance_in(pin_unit, pap_unit, radius),
        psa_to_pap_in=marker_distance_in(psa_unit, pap_unit, radius),
        pin_to_psa_arc_in=marker_distance_in(pin_unit, psa_unit, radius),
        pin_buffer_in=pin_buffer_from_geometry(pin_unit, pap_unit, radius),
        angle_at_pin_deg=angle_between_great_circle_paths_at_marker(
            pin_unit, pap_unit, psa_unit
        ),
    )


def derive_constraint_set(
    layout: DualAngleLayout,
    *,
    psa_to_pap_in: Optional[float] = None,  # noqa: UP045 - Python 3.9
    pin_to_psa_arc_in: Optional[float] = None,  # noqa: UP045 - Python 3.9
    pin_buffer_in: Optional[float] = None,  # noqa: UP045 - Python 3.9
    manufacturer_geometry_verified: bool = False,
) -> ConstraintDerivationResult:
    """Return only explicitly supplied constraints and their conversion status.

    The three dual-angle fields alone are insufficient.  Explicit PSA-to-PAP
    and PIN-to-PSA distances make the marker constraint set complete; it is only
    marked verified when the caller also confirms a verified manufacturer
    geometry source.  No value is derived from drilling or VAL angle here.
    """
    if not isinstance(layout, DualAngleLayout):
        raise TypeError("layout must be a DualAngleLayout")

    diagnostics: list[str] = []
    complete = psa_to_pap_in is not None and pin_to_psa_arc_in is not None
    if not complete:
        status = ConversionStatus.INSUFFICIENT_DATA
        diagnostics.append(
            "dual-angle inputs do not define PSA-to-PAP and PIN-to-PSA distances"
        )
    elif manufacturer_geometry_verified:
        status = ConversionStatus.VERIFIED
        diagnostics.append(
            "explicit marker constraints have a verified geometry source"
        )
    else:
        status = ConversionStatus.PROVISIONAL
        diagnostics.append(
            "explicit marker constraints were supplied without a verified manufacturer convention"
        )

    constraints = LayoutConstraintSet(
        pin_to_pap_in=layout.pin_to_pap_in,
        psa_to_pap_in=psa_to_pap_in,
        pin_to_psa_arc_in=pin_to_psa_arc_in,
        pin_buffer_in=pin_buffer_in,
        diagnostics=diagnostics,
    )
    return ConstraintDerivationResult(status, constraints)


def _vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3D vector")
    return vector


def _unit_vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = _vector3(value, name)
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=_VECTOR_ATOL, rtol=1e-12):
        raise ValueError(f"{name} must be normalized")
    return vector


def _finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_scalar(value: float, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar <= 0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _nonnegative_scalar(value: float, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar < 0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _angle_0_to_180(value: float, name: str) -> float:
    angle = _finite_scalar(value, name)
    if not 0 < angle <= 180:
        raise ValueError(f"{name} must be greater than 0 and at most 180 degrees")
    return angle
