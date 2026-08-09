"""Provisional version-1 PAP, PIN, PSA, VAL, and dual-angle geometry.

The ball center is the origin and PAP is the local north pole.  Layout distances
are great-circle surface distances.  These conventions are deliberately
isolated: they are a consistent spherical model, not yet an industry-verified
interpretation of every manufacturer's dual-angle layout system.

To close the otherwise underdetermined PAP-PIN-PSA triangle, version 1 assumes
PIN and PSA are orthogonal principal-axis directions (90 degrees apart on the
unit sphere).  ``drilling_angle_deg`` is the spherical angle at PIN between the
PIN->PAP and PIN->PSA arcs.  The handedness-aware local frame selects the
corresponding mirrored solution.  No finger/thumb placement, spans, or drilling
pitches are modeled here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inertia import inches_to_meters, meters_to_inches
from .models import BowlerSpec, Handedness

FloatArray = NDArray[np.float64]
_VECTOR_ATOL = 1e-12
_GEOMETRY_ATOL = 1e-9


@dataclass(frozen=True)
class BowlingFrame:
    """Right-handed orthonormal frame anchored at the PAP direction."""

    pap_unit: FloatArray
    x_tangent: FloatArray
    y_tangent: FloatArray
    handedness: Handedness

    def __post_init__(self) -> None:
        pap = _unit_vector3(self.pap_unit, "pap_unit").copy()
        x_tangent = _unit_vector3(self.x_tangent, "x_tangent").copy()
        y_tangent = _unit_vector3(self.y_tangent, "y_tangent").copy()
        handedness = Handedness(self.handedness)

        basis = np.column_stack((x_tangent, y_tangent, pap))
        if not np.allclose(
            basis.T @ basis, np.identity(3), atol=_VECTOR_ATOL, rtol=1e-12
        ):
            raise ValueError("BowlingFrame axes must be orthonormal")
        if not np.isclose(np.linalg.det(basis), 1.0, atol=_VECTOR_ATOL, rtol=0.0):
            raise ValueError("BowlingFrame axes must form a right-handed basis")

        pap.setflags(write=False)
        x_tangent.setflags(write=False)
        y_tangent.setflags(write=False)
        object.__setattr__(self, "pap_unit", pap)
        object.__setattr__(self, "x_tangent", x_tangent)
        object.__setattr__(self, "y_tangent", y_tangent)
        object.__setattr__(self, "handedness", handedness)


def build_pap_frame(bowler: BowlerSpec) -> BowlingFrame:
    """Build the provisional local PAP frame for a bowler's handedness.

    PAP offset measurements are intentionally not interpreted yet.  Mirroring
    horizontal +X for a left-hander also reverses local +Y to retain a proper,
    right-handed frame with fixed PAP +Z.
    """
    if not isinstance(bowler, BowlerSpec):
        raise TypeError("bowler must be a BowlerSpec")
    pap = np.array([0.0, 0.0, 1.0])
    horizontal_sign = 1.0 if bowler.handedness == Handedness.RIGHT else -1.0
    x_tangent = np.array([horizontal_sign, 0.0, 0.0])
    y_tangent = np.cross(pap, x_tangent)
    return BowlingFrame(pap, x_tangent, y_tangent, bowler.handedness)


def rotate_about_axis(
    vector: ArrayLike, axis: ArrayLike, angle_deg: float
) -> FloatArray:
    """Rotate a finite vector about a unit axis using Rodrigues' formula."""
    value = _vector3(vector, "vector")
    unit_axis = _unit_vector3(axis, "axis")
    angle = np.deg2rad(_finite_scalar(angle_deg, "angle_deg"))
    rotated = (
        value * np.cos(angle)
        + np.cross(unit_axis, value) * np.sin(angle)
        + unit_axis * np.dot(unit_axis, value) * (1.0 - np.cos(angle))
    )
    if not np.all(np.isfinite(rotated)):
        raise ValueError("rotation produced nonfinite geometry")
    return rotated


def surface_direction_from_pap(
    frame: BowlingFrame,
    arc_distance_in: float,
    bearing_deg: float,
    ball_radius_m: float,
) -> FloatArray:
    """Move from PAP along a great circle at a local tangent bearing.

    Bearing is clockwise in the local coordinate naming: 0° is +Y, 90° is
    +X, 180° is -Y, and 270° is -X.
    """
    _require_frame(frame)
    distance_m = inches_to_meters(
        _nonnegative_scalar(arc_distance_in, "arc_distance_in")
    )
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    theta = distance_m / radius
    if theta > np.pi + _GEOMETRY_ATOL:
        raise ValueError("arc_distance_in cannot exceed half the sphere circumference")
    bearing = np.deg2rad(_finite_scalar(bearing_deg, "bearing_deg"))
    tangent = np.cos(bearing) * frame.y_tangent + np.sin(bearing) * frame.x_tangent
    return _normalize(frame.pap_unit * np.cos(theta) + tangent * np.sin(theta))


def pap_distance_in(
    unit_direction: ArrayLike, frame: BowlingFrame, ball_radius_m: float
) -> float:
    """Return great-circle surface distance from PAP, in inches."""
    _require_frame(frame)
    direction = _unit_vector3(unit_direction, "unit_direction")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    angle = np.arccos(np.clip(np.dot(frame.pap_unit, direction), -1.0, 1.0))
    return meters_to_inches(float(radius * angle))


def tangent_bearing_deg(unit_direction: ArrayLike, frame: BowlingFrame) -> float:
    """Return the local PAP bearing in degrees in the range [0, 360)."""
    _require_frame(frame)
    direction = _unit_vector3(unit_direction, "unit_direction")
    tangent = direction - np.dot(direction, frame.pap_unit) * frame.pap_unit
    if np.linalg.norm(tangent) <= _VECTOR_ATOL:
        raise ValueError("bearing is undefined at PAP and its antipode")
    tangent = _normalize(tangent)
    bearing = np.rad2deg(
        np.arctan2(
            np.dot(tangent, frame.x_tangent),
            np.dot(tangent, frame.y_tangent),
        )
    )
    return float(bearing % 360.0)


def dual_angle_pin_direction(
    frame: BowlingFrame,
    pin_to_pap_in: float,
    val_angle_deg: float,
    ball_radius_m: float,
) -> FloatArray:
    """Place PIN using VAL angle as a provisional PAP tangent bearing."""
    return surface_direction_from_pap(
        frame, pin_to_pap_in, val_angle_deg, ball_radius_m
    )


def psa_direction_from_dual_angle(
    frame: BowlingFrame,
    pin_direction: ArrayLike,
    drilling_angle_deg: float,
) -> FloatArray:
    """Construct the version-1 PSA direction from spherical dual-angle data.

    PIN->PSA is fixed at 90 degrees.  At PIN, its initial tangent is obtained by
    rotating PIN->PAP's great-circle tangent through the drilling angle.  The
    same local positive rotation is used in both handed frames; the mirrored
    frame and PIN direction produce the corresponding handedness-aware PSA.
    """
    _require_frame(frame)
    pin = _unit_vector3(pin_direction, "pin_direction")
    drilling_angle = _finite_scalar(drilling_angle_deg, "drilling_angle_deg")
    if not 0.0 < drilling_angle < 180.0:
        raise ValueError("drilling_angle_deg must be between 0 and 180 degrees")
    toward_pap = frame.pap_unit - np.dot(frame.pap_unit, pin) * pin
    if np.linalg.norm(toward_pap) <= _VECTOR_ATOL:
        raise ValueError("pin_direction cannot coincide with PAP or its antipode")
    toward_pap = _normalize(toward_pap)
    toward_psa = rotate_about_axis(toward_pap, pin, drilling_angle)
    # At 90° spherical travel from PIN, the exponential-map endpoint equals
    # the unit initial tangent itself.
    return _normalize(toward_psa)


def spherical_triangle_angle_at_vertex(
    a: ArrayLike, b: ArrayLike, c: ArrayLike
) -> float:
    """Return, in degrees, the spherical angle at b between b->a and b->c."""
    unit_a = _unit_vector3(a, "a")
    unit_b = _unit_vector3(b, "b")
    unit_c = _unit_vector3(c, "c")
    tangent_a = unit_a - np.dot(unit_a, unit_b) * unit_b
    tangent_c = unit_c - np.dot(unit_c, unit_b) * unit_b
    if (
        np.linalg.norm(tangent_a) <= _VECTOR_ATOL
        or np.linalg.norm(tangent_c) <= _VECTOR_ATOL
    ):
        raise ValueError("spherical triangle has a coincident or antipodal vertex")
    tangent_a = _normalize(tangent_a)
    tangent_c = _normalize(tangent_c)
    angle = np.arccos(np.clip(np.dot(tangent_a, tangent_c), -1.0, 1.0))
    return float(np.rad2deg(angle))


def validate_dual_angle_geometry(
    pap: ArrayLike,
    pin: ArrayLike,
    psa: ArrayLike,
    pin_to_pap_in: float,
    drilling_angle_deg: float,
    val_angle_deg: float,
    ball_radius_m: float,
) -> tuple[str, ...]:
    """Validate version-1 spherical dual-angle geometry or raise ValueError."""
    pap_unit = _unit_vector3(pap, "pap")
    pin_unit = _unit_vector3(pin, "pin")
    psa_unit = _unit_vector3(psa, "psa")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    expected_distance = _nonnegative_scalar(pin_to_pap_in, "pin_to_pap_in")
    expected_drilling = _finite_scalar(drilling_angle_deg, "drilling_angle_deg")
    expected_bearing = _finite_scalar(val_angle_deg, "val_angle_deg") % 360.0

    actual_distance = meters_to_inches(
        radius * float(np.arccos(np.clip(np.dot(pap_unit, pin_unit), -1.0, 1.0)))
    )
    if not np.isclose(actual_distance, expected_distance, atol=1e-8, rtol=1e-9):
        raise ValueError("pin-to-PAP surface distance does not match layout input")

    actual_drilling = spherical_triangle_angle_at_vertex(pap_unit, pin_unit, psa_unit)
    if not np.isclose(actual_drilling, expected_drilling, atol=1e-8, rtol=1e-9):
        raise ValueError("spherical angle at PIN does not match drilling angle")

    # The supplied PAP establishes +Z, but bearing needs the version-1 global
    # tangent orientation.  Try each handed frame; validation is geometric and
    # succeeds when the pin matches the requested bearing in either valid frame.
    bearing_matches = False
    for handedness in (Handedness.RIGHT, Handedness.LEFT):
        frame = _frame_for_pap(pap_unit, handedness)
        actual_bearing = tangent_bearing_deg(pin_unit, frame)
        if _circular_difference_deg(actual_bearing, expected_bearing) <= 1e-8:
            bearing_matches = True
            break
    if not bearing_matches:
        raise ValueError("PIN bearing does not match VAL-angle placement")
    return ()


def _frame_for_pap(pap: FloatArray, handedness: Handedness) -> BowlingFrame:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, pap)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x = _normalize(reference - np.dot(reference, pap) * pap)
    if handedness == Handedness.LEFT:
        x = -x
    y = np.cross(pap, x)
    return BowlingFrame(pap, x, y, handedness)


def _circular_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _require_frame(frame: BowlingFrame) -> None:
    if not isinstance(frame, BowlingFrame):
        raise TypeError("frame must be a BowlingFrame")


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
