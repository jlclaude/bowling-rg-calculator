"""Surface-entry geometry for a bowling grip, without drilling pitch.

Sign conventions
----------------
The oriented VAL tangent ``cross(VAL pole, PAP)`` is local PAP-up.  A positive
reported vertical PAP offset is undone by moving down (the negative tangent).
At that crossing, the midline tangent is perpendicular to VAL.  A positive
right-handed horizontal offset moves along that tangent back toward the grip;
left-handed geometry uses its mirror.

Version 1 uses the reconstructed grip center as the thumb-center reference.
The requested inputs provide no independent longitudinal thumb location, so no
additional offset is guessed.  Finger entry points are an exact spherical SSS
solution from their spans and center-to-center spacing.  Hole pitch is not
modeled or applied anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dual_angle_conversion import (
    GreatCircle,
    great_circle_through_point_and_tangent,
)
from .inertia import inches_to_meters
from .layout_geometry import marker_distance_in
from .models import Handedness

FloatArray = NDArray[np.float64]
_VECTOR_ATOL = 1e-12
_GEOMETRY_ATOL = 1e-9


@dataclass(frozen=True)
class GripFrame:
    """Orthonormal local geometry connecting PAP, VAL, midline, and grip."""

    pap_unit: FloatArray
    grip_center_unit: FloatArray
    midline_intersection_unit: FloatArray
    val_tangent_at_pap: FloatArray
    midline_tangent_at_grip: FloatArray
    centerline_tangent_at_grip: FloatArray

    def __post_init__(self) -> None:
        unit_names = (
            "pap_unit",
            "grip_center_unit",
            "midline_intersection_unit",
            "val_tangent_at_pap",
            "midline_tangent_at_grip",
            "centerline_tangent_at_grip",
        )
        for name in unit_names:
            vector = _unit_vector3(getattr(self, name), name).copy()
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)

        grip = self.grip_center_unit
        midline = self.midline_tangent_at_grip
        centerline = self.centerline_tangent_at_grip
        if not np.allclose(
            np.column_stack((midline, centerline, grip)).T
            @ np.column_stack((midline, centerline, grip)),
            np.identity(3),
            atol=_VECTOR_ATOL,
            rtol=1e-12,
        ):
            raise ValueError("grip-center tangents must be orthonormal")


@dataclass(frozen=True)
class GripMeasurements:
    """Inch-based PAP offsets and grip measurements."""

    pap_horizontal_in: float
    pap_vertical_in: float
    middle_span_in: float
    ring_span_in: float
    bridge_in: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pap_horizontal_in",
            _finite_scalar(self.pap_horizontal_in, "pap_horizontal_in"),
        )
        object.__setattr__(
            self,
            "pap_vertical_in",
            _finite_scalar(self.pap_vertical_in, "pap_vertical_in"),
        )
        object.__setattr__(
            self,
            "middle_span_in",
            _positive_scalar(self.middle_span_in, "middle_span_in"),
        )
        object.__setattr__(
            self,
            "ring_span_in",
            _positive_scalar(self.ring_span_in, "ring_span_in"),
        )
        object.__setattr__(
            self,
            "bridge_in",
            _nonnegative_scalar(self.bridge_in, "bridge_in"),
        )


@dataclass(frozen=True)
class HoleEntryPoints:
    """Normalized thumb and finger surface-entry directions."""

    thumb_unit: FloatArray
    middle_unit: FloatArray
    ring_unit: FloatArray
    grip_center_unit: FloatArray

    def __post_init__(self) -> None:
        for name in ("thumb_unit", "middle_unit", "ring_unit", "grip_center_unit"):
            vector = _unit_vector3(getattr(self, name), name).copy()
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)


def tangent_of_great_circle_at_point(
    great_circle: GreatCircle,
    point_unit: ArrayLike,
    sign: int = 1,
) -> FloatArray:
    """Return an oriented unit tangent to a great circle at a point on it."""
    if not isinstance(great_circle, GreatCircle):
        raise TypeError("great_circle must be a GreatCircle")
    point = _unit_vector3(point_unit, "point_unit")
    direction_sign = _sign(sign)
    if not np.isclose(
        np.dot(great_circle.pole_unit, point),
        0.0,
        atol=_VECTOR_ATOL,
        rtol=0.0,
    ):
        raise ValueError("point_unit must lie on the great circle")
    return _normalize(direction_sign * np.cross(great_circle.pole_unit, point))


def move_along_great_circle(
    start_unit: ArrayLike,
    tangent_unit: ArrayLike,
    arc_distance_in: float,
    ball_radius_m: float,
) -> FloatArray:
    """Move a signed surface distance along a great circle."""
    start = _unit_vector3(start_unit, "start_unit")
    tangent = _unit_vector3(tangent_unit, "tangent_unit")
    if not np.isclose(np.dot(start, tangent), 0.0, atol=_VECTOR_ATOL, rtol=0.0):
        raise ValueError("tangent_unit must be perpendicular to start_unit")
    distance = _finite_scalar(arc_distance_in, "arc_distance_in")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    theta = inches_to_meters(distance) / radius
    if abs(theta) > np.pi + _GEOMETRY_ATOL:
        raise ValueError("arc distance cannot exceed half the sphere circumference")
    return _normalize(start * np.cos(theta) + tangent * np.sin(theta))


def grip_center_from_pap(
    pap_unit: ArrayLike,
    val_great_circle: GreatCircle,
    pap_horizontal_in: float,
    pap_vertical_in: float,
    ball_radius_m: float,
    handedness: Handedness,
) -> FloatArray:
    """Undo vertical then handed horizontal PAP offsets to find grip center."""
    _, grip_center, _, _, _ = _grip_construction(
        pap_unit,
        val_great_circle,
        pap_horizontal_in,
        pap_vertical_in,
        ball_radius_m,
        handedness,
    )
    return grip_center


def construct_grip_frame(
    pap_unit: ArrayLike,
    val_great_circle: GreatCircle,
    pap_horizontal_in: float,
    pap_vertical_in: float,
    ball_radius_m: float,
    handedness: Handedness,
) -> GripFrame:
    """Construct the local grip frame from PAP offsets and an explicit VAL."""
    pap, grip, crossing, val_tangent, midline_tangent = _grip_construction(
        pap_unit,
        val_great_circle,
        pap_horizontal_in,
        pap_vertical_in,
        ball_radius_m,
        handedness,
    )
    centerline_tangent = _normalize(np.cross(midline_tangent, grip))
    return GripFrame(
        pap,
        grip,
        crossing,
        val_tangent,
        midline_tangent,
        centerline_tangent,
    )


def finger_pair_from_bridge(
    grip_frame: GripFrame,
    middle_span_in: float,
    ring_span_in: float,
    bridge_in: float,
    middle_hole_diameter_in: float,
    ring_hole_diameter_in: float,
    ball_radius_m: float,
) -> tuple[FloatArray, FloatArray]:
    """Solve middle/ring centers from two spans and explicit center spacing."""
    if not isinstance(grip_frame, GripFrame):
        raise TypeError("grip_frame must be a GripFrame")
    middle_span = _positive_scalar(middle_span_in, "middle_span_in")
    ring_span = _positive_scalar(ring_span_in, "ring_span_in")
    bridge = _nonnegative_scalar(bridge_in, "bridge_in")
    middle_diameter = _positive_scalar(
        middle_hole_diameter_in, "middle_hole_diameter_in"
    )
    ring_diameter = _positive_scalar(ring_hole_diameter_in, "ring_hole_diameter_in")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    center_separation = middle_diameter / 2.0 + bridge + ring_diameter / 2.0

    middle_angle = _arc_angle(middle_span, radius, "middle_span_in")
    ring_angle = _arc_angle(ring_span, radius, "ring_span_in")
    separation_angle = _arc_angle(center_separation, radius, "finger center separation")
    _validate_spherical_sides(middle_angle, ring_angle, separation_angle)
    denominator = np.sin(middle_angle) * np.sin(ring_angle)
    cosine_included = (
        np.cos(separation_angle) - np.cos(middle_angle) * np.cos(ring_angle)
    ) / denominator
    if (
        cosine_included < -1.0 - _GEOMETRY_ATOL
        or cosine_included > 1.0 + _GEOMETRY_ATOL
    ):
        raise ValueError("spans and finger spacing do not form a spherical triangle")
    included = float(np.arccos(np.clip(cosine_included, -1.0, 1.0)))

    half_included = included / 2.0
    middle_tangent = _normalize(
        np.cos(half_included) * grip_frame.centerline_tangent_at_grip
        + np.sin(half_included) * grip_frame.midline_tangent_at_grip
    )
    ring_tangent = _normalize(
        np.cos(half_included) * grip_frame.centerline_tangent_at_grip
        - np.sin(half_included) * grip_frame.midline_tangent_at_grip
    )
    thumb = grip_frame.grip_center_unit
    middle = _normalize(
        thumb * np.cos(middle_angle) + middle_tangent * np.sin(middle_angle)
    )
    ring = _normalize(thumb * np.cos(ring_angle) + ring_tangent * np.sin(ring_angle))
    return middle, ring


def solve_hole_entry_points(
    grip_frame: GripFrame,
    measurements: GripMeasurements,
    middle_hole_diameter_in: float,
    ring_hole_diameter_in: float,
    ball_radius_m: float,
) -> HoleEntryPoints:
    """Return thumb, middle, and ring entry directions without drilling pitch."""
    if not isinstance(measurements, GripMeasurements):
        raise TypeError("measurements must be GripMeasurements")
    middle, ring = finger_pair_from_bridge(
        grip_frame,
        measurements.middle_span_in,
        measurements.ring_span_in,
        measurements.bridge_in,
        middle_hole_diameter_in,
        ring_hole_diameter_in,
        ball_radius_m,
    )
    entries = HoleEntryPoints(
        thumb_unit=grip_frame.grip_center_unit,
        middle_unit=middle,
        ring_unit=ring,
        grip_center_unit=grip_frame.grip_center_unit,
    )
    validate_grip_geometry(
        grip_frame,
        entries,
        measurements,
        middle_hole_diameter_in,
        ring_hole_diameter_in,
        ball_radius_m,
    )
    return entries


def validate_grip_geometry(
    grip_frame: GripFrame,
    entries: HoleEntryPoints,
    measurements: GripMeasurements,
    middle_hole_diameter_in: float,
    ring_hole_diameter_in: float,
    ball_radius_m: float,
) -> tuple[str, ...]:
    """Validate normalization, spans, bridge spacing, and PAP offset distances."""
    if not isinstance(grip_frame, GripFrame):
        raise TypeError("grip_frame must be a GripFrame")
    if not isinstance(entries, HoleEntryPoints):
        raise TypeError("entries must be HoleEntryPoints")
    if not isinstance(measurements, GripMeasurements):
        raise TypeError("measurements must be GripMeasurements")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    for name in ("thumb_unit", "middle_unit", "ring_unit", "grip_center_unit"):
        _unit_vector3(getattr(entries, name), name)

    _require_distance(
        entries.thumb_unit,
        entries.middle_unit,
        measurements.middle_span_in,
        radius,
        "middle span",
    )
    _require_distance(
        entries.thumb_unit,
        entries.ring_unit,
        measurements.ring_span_in,
        radius,
        "ring span",
    )
    expected_separation = (
        _positive_scalar(middle_hole_diameter_in, "middle_hole_diameter_in") / 2
        + measurements.bridge_in
        + _positive_scalar(ring_hole_diameter_in, "ring_hole_diameter_in") / 2
    )
    _require_distance(
        entries.middle_unit,
        entries.ring_unit,
        expected_separation,
        radius,
        "finger center separation",
    )
    _require_distance(
        grip_frame.pap_unit,
        grip_frame.midline_intersection_unit,
        abs(measurements.pap_vertical_in),
        radius,
        "vertical PAP offset",
    )
    _require_distance(
        grip_frame.midline_intersection_unit,
        grip_frame.grip_center_unit,
        abs(measurements.pap_horizontal_in),
        radius,
        "horizontal PAP offset",
    )
    return ()


def _grip_construction(
    pap_unit: ArrayLike,
    val_great_circle: GreatCircle,
    pap_horizontal_in: float,
    pap_vertical_in: float,
    ball_radius_m: float,
    handedness: Handedness,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    pap = _unit_vector3(pap_unit, "pap_unit")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    horizontal = _finite_scalar(pap_horizontal_in, "pap_horizontal_in")
    vertical = _finite_scalar(pap_vertical_in, "pap_vertical_in")
    hand = Handedness(handedness)
    val_tangent = tangent_of_great_circle_at_point(val_great_circle, pap)
    crossing = move_along_great_circle(pap, val_tangent, -vertical, radius)

    val_tangent_at_crossing = tangent_of_great_circle_at_point(
        val_great_circle, crossing
    )
    base_midline_tangent = _normalize(np.cross(crossing, val_tangent_at_crossing))
    hand_sign = 1.0 if hand == Handedness.RIGHT else -1.0
    toward_grip = hand_sign * base_midline_tangent
    midline_circle = great_circle_through_point_and_tangent(crossing, toward_grip)
    grip = move_along_great_circle(crossing, toward_grip, horizontal, radius)
    midline_at_grip = tangent_of_great_circle_at_point(midline_circle, grip)
    return pap, grip, crossing, val_tangent, midline_at_grip


def _require_distance(
    a: FloatArray,
    b: FloatArray,
    expected_in: float,
    radius_m: float,
    name: str,
) -> None:
    actual = marker_distance_in(a, b, radius_m)
    if not np.isclose(actual, expected_in, atol=1e-8, rtol=1e-9):
        raise ValueError(f"{name} does not match requested geometry")


def _arc_angle(distance_in: float, radius_m: float, name: str) -> float:
    angle = inches_to_meters(distance_in) / radius_m
    if not 0 < angle < np.pi - _GEOMETRY_ATOL:
        raise ValueError(f"{name} must be shorter than half the sphere circumference")
    return angle


def _validate_spherical_sides(a: float, b: float, c: float) -> None:
    if not (
        a + b > c + _GEOMETRY_ATOL
        and a + c > b + _GEOMETRY_ATOL
        and b + c > a + _GEOMETRY_ATOL
    ):
        raise ValueError(
            "spans and finger spacing violate spherical triangle inequality"
        )
    if a + b + c >= 2 * np.pi - _GEOMETRY_ATOL:
        raise ValueError("spans and finger spacing have impossible spherical perimeter")


def _sign(value: int) -> int:
    if value not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
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
