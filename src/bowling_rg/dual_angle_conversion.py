"""Exact spherical geometry for the documented dual-angle convention.

The Vertical Axis Line (VAL) is a great circle passing through PAP; PAP is not
its pole.  This module interprets VAL angle as the tangent angle at PAP between
the PAP->PIN great-circle path and the VAL.  PIN and PSA are treated as
orthogonal principal-axis markers.  No finger/thumb geometry is modeled.
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
    rotate_about_axis,
    surface_direction_from_pap,
)

FloatArray = NDArray[np.float64]
_VECTOR_ATOL = 1e-12
_GEOMETRY_ATOL = 1e-9


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
class GreatCircle:
    """A centered plane on the unit sphere, represented by its unit pole."""

    pole_unit: FloatArray

    def __post_init__(self) -> None:
        pole = _unit_vector3(self.pole_unit, "pole_unit").copy()
        pole.setflags(write=False)
        object.__setattr__(self, "pole_unit", pole)


@dataclass(frozen=True)
class DualAngleGeometry:
    """Explicit marker and VAL geometry produced by a dual-angle layout."""

    pap_unit: FloatArray
    pin_unit: FloatArray
    psa_unit: FloatArray
    val_great_circle: GreatCircle
    drilling_angle_deg: float
    pin_to_pap_in: float
    val_angle_deg: float
    pin_buffer_in: float

    def __post_init__(self) -> None:
        for name in ("pap_unit", "pin_unit", "psa_unit"):
            vector = _unit_vector3(getattr(self, name), name).copy()
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)
        if not isinstance(self.val_great_circle, GreatCircle):
            raise TypeError("val_great_circle must be a GreatCircle")
        object.__setattr__(
            self,
            "drilling_angle_deg",
            _angle_0_to_180(self.drilling_angle_deg, "drilling_angle_deg"),
        )
        object.__setattr__(
            self,
            "pin_to_pap_in",
            _positive_scalar(self.pin_to_pap_in, "pin_to_pap_in"),
        )
        object.__setattr__(
            self,
            "val_angle_deg",
            _angle_0_to_180(self.val_angle_deg, "val_angle_deg"),
        )
        object.__setattr__(
            self,
            "pin_buffer_in",
            _nonnegative_scalar(self.pin_buffer_in, "pin_buffer_in"),
        )


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


def great_circle_through_point_and_tangent(
    point_unit: ArrayLike, tangent_unit: ArrayLike
) -> GreatCircle:
    """Construct the great circle spanned by perpendicular point and tangent."""
    point = _unit_vector3(point_unit, "point_unit")
    tangent = _unit_vector3(tangent_unit, "tangent_unit")
    if not np.isclose(np.dot(point, tangent), 0.0, atol=_VECTOR_ATOL, rtol=0.0):
        raise ValueError("point_unit and tangent_unit must be perpendicular")
    pole = _normalize(np.cross(point, tangent))
    if not np.isclose(np.dot(pole, point), 0.0, atol=_VECTOR_ATOL, rtol=0.0):
        raise ValueError("great-circle pole must be perpendicular to point")
    if not np.isclose(np.dot(pole, tangent), 0.0, atol=_VECTOR_ATOL, rtol=0.0):
        raise ValueError("great-circle pole must be perpendicular to tangent")
    return GreatCircle(pole)


def val_from_pap_and_tangent(
    pap_unit: ArrayLike, val_tangent_at_pap: ArrayLike
) -> GreatCircle:
    """Construct a VAL great circle that passes through PAP."""
    val = great_circle_through_point_and_tangent(pap_unit, val_tangent_at_pap)
    pap = _unit_vector3(pap_unit, "pap_unit")
    if not np.isclose(np.dot(pap, val.pole_unit), 0.0, atol=_VECTOR_ATOL, rtol=0.0):
        raise ValueError("PAP must lie on the VAL great circle")
    return val


def tangent_toward_marker_at_point(
    point_unit: ArrayLike, marker_unit: ArrayLike
) -> FloatArray:
    """Return the initial great-circle tangent from point toward marker."""
    point = _unit_vector3(point_unit, "point_unit")
    marker = _unit_vector3(marker_unit, "marker_unit")
    projection = marker - np.dot(marker, point) * point
    if np.linalg.norm(projection) <= _VECTOR_ATOL:
        raise ValueError("point and marker cannot be coincident or antipodal")
    return _normalize(projection)


def construct_val_from_layout(
    pap_unit: ArrayLike,
    pin_unit: ArrayLike,
    val_angle_deg: float,
    side_sign: int = 1,
) -> GreatCircle:
    """Construct VAL from the tangent angle at PAP between PAP->PIN and VAL."""
    pap = _unit_vector3(pap_unit, "pap_unit")
    tangent_to_pin = tangent_toward_marker_at_point(pap, pin_unit)
    val_angle = _angle_0_to_180(val_angle_deg, "val_angle_deg")
    side = _side_sign(side_sign)
    val_tangent = _normalize(rotate_about_axis(tangent_to_pin, pap, side * val_angle))
    return val_from_pap_and_tangent(pap, val_tangent)


def shortest_surface_distance_to_great_circle(
    marker_unit: ArrayLike,
    great_circle: GreatCircle,
    ball_radius_m: float,
) -> float:
    """Return shortest marker-to-great-circle surface distance in meters."""
    marker = _unit_vector3(marker_unit, "marker_unit")
    if not isinstance(great_circle, GreatCircle):
        raise TypeError("great_circle must be a GreatCircle")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    angular_distance = np.arcsin(
        np.clip(abs(np.dot(marker, great_circle.pole_unit)), 0.0, 1.0)
    )
    return float(radius * angular_distance)


def pin_buffer_from_geometry(
    pin_unit: ArrayLike,
    val_great_circle: GreatCircle,
    ball_radius_m: float,
) -> float:
    """Return shortest PIN-to-explicit-VAL surface distance, in inches."""
    distance_m = shortest_surface_distance_to_great_circle(
        pin_unit, val_great_circle, ball_radius_m
    )
    return meters_to_inches(distance_m)


def angle_between_great_circle_paths_at_marker(
    marker: ArrayLike,
    path_point_a: ArrayLike,
    path_point_b: ArrayLike,
) -> float:
    """Return the spherical tangent angle between two paths, in degrees."""
    return marker_angle_at(marker, path_point_a, path_point_b)


def asymmetric_core_markers_from_dual_angle(
    frame: BowlingFrame,
    layout: DualAngleLayout,
    ball_radius_m: float,
    side_sign: int = 1,
) -> DualAngleGeometry:
    """Construct exact spherical marker geometry for the documented convention."""
    if not isinstance(frame, BowlingFrame):
        raise TypeError("frame must be a BowlingFrame")
    if not isinstance(layout, DualAngleLayout):
        raise TypeError("layout must be a DualAngleLayout")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    side = _side_sign(side_sign)

    # Bearing zero (+Y) is a coordinate gauge; only relative geometry matters.
    pin = surface_direction_from_pap(frame, layout.pin_to_pap_in, 0.0, radius)
    pin_toward_pap = tangent_toward_marker_at_point(pin, frame.pap_unit)
    pin_toward_psa = _normalize(
        rotate_about_axis(
            pin_toward_pap,
            pin,
            side * layout.drilling_angle_deg,
        )
    )
    # A 90-degree exponential-map step from PIN has endpoint equal to its unit
    # initial tangent, giving an exactly orthogonal PIN/PSA marker pair.
    psa = pin_toward_psa
    val = construct_val_from_layout(
        frame.pap_unit,
        pin,
        layout.val_angle_deg,
        side,
    )
    pin_buffer = pin_buffer_from_geometry(pin, val, radius)
    geometry = DualAngleGeometry(
        pap_unit=frame.pap_unit,
        pin_unit=pin,
        psa_unit=psa,
        val_great_circle=val,
        drilling_angle_deg=layout.drilling_angle_deg,
        pin_to_pap_in=layout.pin_to_pap_in,
        val_angle_deg=layout.val_angle_deg,
        pin_buffer_in=pin_buffer,
    )
    validate_dual_angle_geometry_exact(geometry, layout, radius)
    return geometry


def validate_dual_angle_geometry_exact(
    geometry: DualAngleGeometry,
    layout: DualAngleLayout,
    ball_radius_m: float,
) -> tuple[str, ...]:
    """Verify every defining marker, VAL, angle, and pin-buffer constraint."""
    if not isinstance(geometry, DualAngleGeometry):
        raise TypeError("geometry must be DualAngleGeometry")
    if not isinstance(layout, DualAngleLayout):
        raise TypeError("layout must be a DualAngleLayout")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")

    pin_pap = marker_distance_in(geometry.pin_unit, geometry.pap_unit, radius)
    _require_close(pin_pap, layout.pin_to_pap_in, "PIN-to-PAP distance")

    pin_psa_angle = np.rad2deg(
        np.arccos(np.clip(np.dot(geometry.pin_unit, geometry.psa_unit), -1.0, 1.0))
    )
    _require_close(float(pin_psa_angle), 90.0, "PIN-to-PSA angular separation")

    drilling = angle_between_great_circle_paths_at_marker(
        geometry.pin_unit, geometry.pap_unit, geometry.psa_unit
    )
    _require_close(drilling, layout.drilling_angle_deg, "drilling angle")

    pap_pole_dot = float(np.dot(geometry.pap_unit, geometry.val_great_circle.pole_unit))
    _require_close(pap_pole_dot, 0.0, "PAP-to-VAL-pole dot product")

    tangent_to_pin = tangent_toward_marker_at_point(
        geometry.pap_unit, geometry.pin_unit
    )
    val_tangent = _normalize(
        np.cross(geometry.val_great_circle.pole_unit, geometry.pap_unit)
    )
    val_angle = np.rad2deg(
        np.arccos(np.clip(np.dot(tangent_to_pin, val_tangent), -1.0, 1.0))
    )
    _require_close(float(val_angle), layout.val_angle_deg, "VAL angle")

    measured_buffer = pin_buffer_from_geometry(
        geometry.pin_unit, geometry.val_great_circle, radius
    )
    _require_close(measured_buffer, geometry.pin_buffer_in, "pin buffer")
    return ()


def dual_angle_measurements_from_markers(
    pap: ArrayLike,
    pin: ArrayLike,
    psa: ArrayLike,
    val_great_circle: GreatCircle,
    ball_radius_m: float,
) -> MarkerDerivedMeasurements:
    """Measure marker geometry using an explicit, correctly defined VAL."""
    pap_unit = _unit_vector3(pap, "pap")
    pin_unit = _unit_vector3(pin, "pin")
    psa_unit = _unit_vector3(psa, "psa")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    return MarkerDerivedMeasurements(
        pin_to_pap_in=marker_distance_in(pin_unit, pap_unit, radius),
        psa_to_pap_in=marker_distance_in(psa_unit, pap_unit, radius),
        pin_to_psa_arc_in=marker_distance_in(pin_unit, psa_unit, radius),
        pin_buffer_in=pin_buffer_from_geometry(pin_unit, val_great_circle, radius),
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
    """Return only explicitly supplied marker constraints and status.

    This compatibility API still refuses to infer marker side lengths from the
    published three-number layout.  Use ``asymmetric_core_markers_from_dual_angle``
    only for the exact spherical convention documented by this module.
    """
    if not isinstance(layout, DualAngleLayout):
        raise TypeError("layout must be a DualAngleLayout")
    diagnostics: list[str] = []
    complete = psa_to_pap_in is not None and pin_to_psa_arc_in is not None
    if not complete:
        status = ConversionStatus.INSUFFICIENT_DATA
        diagnostics.append(
            "dual-angle inputs do not independently define all SSS marker constraints"
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
    return ConstraintDerivationResult(
        status,
        LayoutConstraintSet(
            pin_to_pap_in=layout.pin_to_pap_in,
            psa_to_pap_in=psa_to_pap_in,
            pin_to_psa_arc_in=pin_to_psa_arc_in,
            pin_buffer_in=pin_buffer_in,
            diagnostics=diagnostics,
        ),
    )


def _require_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, atol=_GEOMETRY_ATOL, rtol=1e-9):
        raise ValueError(f"{name} does not match requested geometry")


def _side_sign(value: int) -> int:
    if value not in (-1, 1):
        raise ValueError("side_sign must be +1 or -1")
    return value


def _normalize(vector: ArrayLike) -> FloatArray:
    value = _vector3(vector, "vector")
    magnitude = float(np.linalg.norm(value))
    if magnitude <= _VECTOR_ATOL:
        raise ValueError("vector must be nonzero")
    return value / magnitude


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
    if not 0 < angle < 180:
        raise ValueError(f"{name} must be between 0 and 180 degrees")
    return angle
