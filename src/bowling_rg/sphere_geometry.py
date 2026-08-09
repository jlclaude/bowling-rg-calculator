"""Generic vector geometry for drilling a sphere, using SI units internally."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inertia import inches_to_meters

DEFAULT_BALL_DIAMETER_IN = 8.585

FloatArray = NDArray[np.float64]
_VECTOR_ATOL = 1e-12
_SURFACE_RTOL = 1e-9


def ball_radius_m(diameter_in: float) -> float:
    """Convert a positive sphere diameter in inches to its radius in meters."""
    diameter = _positive_scalar(diameter_in, "diameter_in")
    return inches_to_meters(diameter) / 2.0


def normalize(vector: ArrayLike) -> FloatArray:
    """Return a finite, normalized three-dimensional vector."""
    value = _vector3(vector, "vector")
    magnitude = float(np.linalg.norm(value))
    if magnitude <= _VECTOR_ATOL:
        raise ValueError("vector must be nonzero")
    return value / magnitude


def surface_point(radius_m: float, outward_unit_vector: ArrayLike) -> FloatArray:
    """Return a point on the sphere in the supplied outward direction."""
    radius = _positive_scalar(radius_m, "radius_m")
    outward = _unit_vector3(outward_unit_vector, "outward_unit_vector")
    return radius * outward


def inward_radial_direction(surface_position: ArrayLike) -> FloatArray:
    """Return the unit direction from a surface point toward sphere center."""
    return -normalize(surface_position)


def drilling_axis_from_surface_normal(
    outward_normal: ArrayLike,
    pitch_vector_tangent: ArrayLike,
    pitch_angle_deg: float,
) -> FloatArray:
    """Tilt an inward radial drilling axis toward a tangent direction.

    The outward normal and tangent direction must be unit vectors, and the
    tangent must be perpendicular to the normal.  Pitch magnitude must remain
    below 90 degrees so that the resulting axis points into the sphere.
    """
    outward = _unit_vector3(outward_normal, "outward_normal")
    tangent = _unit_vector3(pitch_vector_tangent, "pitch_vector_tangent")
    if not np.isclose(np.dot(outward, tangent), 0.0, atol=_VECTOR_ATOL, rtol=0.0):
        raise ValueError("pitch_vector_tangent must be perpendicular to outward_normal")

    pitch = _finite_scalar(pitch_angle_deg, "pitch_angle_deg")
    if abs(pitch) >= 90.0:
        raise ValueError("pitch_angle_deg must have magnitude less than 90 degrees")
    if pitch == 0.0:
        return -outward

    angle = np.deg2rad(pitch)
    return normalize(-outward * np.cos(angle) + tangent * np.sin(angle))


def hole_centroid(
    surface_entry_position: ArrayLike,
    drilling_axis: ArrayLike,
    depth_m: float,
) -> FloatArray:
    """Return the midpoint of a straight cylindrical hole centerline."""
    entry = _vector3(surface_entry_position, "surface_entry_position")
    axis = _unit_vector3(drilling_axis, "drilling_axis")
    depth = _positive_scalar(depth_m, "depth_m")
    return entry + axis * (depth / 2.0)


def hole_end_point(
    surface_entry_position: ArrayLike,
    drilling_axis: ArrayLike,
    depth_m: float,
) -> FloatArray:
    """Return the endpoint of a straight hole centerline."""
    entry = _vector3(surface_entry_position, "surface_entry_position")
    axis = _unit_vector3(drilling_axis, "drilling_axis")
    depth = _positive_scalar(depth_m, "depth_m")
    return entry + axis * depth


def validate_hole_inside_sphere(
    radius_m: float,
    entry_position: ArrayLike,
    drilling_axis: ArrayLike,
    depth_m: float,
    hole_radius_m: float = 0,
) -> tuple[str, ...]:
    """Validate that a straight cylindrical hole is feasible in a sphere.

    The entry center must lie on the sphere, the axis must point inward, and
    depth cannot pass the second sphere intersection.  For a nonzero bore
    radius, the axis line must also leave enough radial clearance for that bore.
    Valid geometry currently produces no advisory warnings.
    """
    radius = _positive_scalar(radius_m, "radius_m")
    entry = _vector3(entry_position, "entry_position")
    axis = _unit_vector3(drilling_axis, "drilling_axis")
    depth = _positive_scalar(depth_m, "depth_m")
    hole_radius = _nonnegative_scalar(hole_radius_m, "hole_radius_m")

    surface_tolerance = max(_VECTOR_ATOL, radius * _SURFACE_RTOL)
    if not np.isclose(
        np.linalg.norm(entry), radius, atol=surface_tolerance, rtol=_SURFACE_RTOL
    ):
        raise ValueError("entry_position must lie on the sphere surface")

    radial_component = float(np.dot(entry, axis))
    if radial_component >= -_VECTOR_ATOL:
        raise ValueError("drilling_axis must point inward")

    opposite_intersection_depth = -2.0 * radial_component
    if depth > opposite_intersection_depth + surface_tolerance:
        raise ValueError("hole depth extends outside the opposite side of the sphere")

    if hole_radius >= radius:
        raise ValueError("hole_radius_m must be smaller than the sphere radius")
    line_offset = float(np.linalg.norm(np.cross(entry, axis)))
    if line_offset + hole_radius > radius + surface_tolerance:
        raise ValueError("hole cylinder radius is incompatible with the sphere")

    endpoint = entry + axis * depth
    if np.linalg.norm(endpoint) > radius + surface_tolerance:
        raise ValueError("hole endpoint lies outside the sphere")
    return ()


def great_circle_distance(
    radius_m: float, unit_a: ArrayLike, unit_b: ArrayLike
) -> float:
    """Return the shortest surface arc length between two sphere directions."""
    radius = _positive_scalar(radius_m, "radius_m")
    return float(radius * _angle_radians(unit_a, unit_b))


def angular_separation(unit_a: ArrayLike, unit_b: ArrayLike) -> float:
    """Return the smaller angle between two unit vectors, in degrees."""
    return float(np.rad2deg(_angle_radians(unit_a, unit_b)))


def tangent_basis_at_surface(
    outward_normal: ArrayLike, reference_direction: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Construct a right-handed orthonormal tangent basis at a sphere point."""
    normal = _unit_vector3(outward_normal, "outward_normal")
    reference = _vector3(reference_direction, "reference_direction")
    tangent_projection = reference - np.dot(reference, normal) * normal
    if np.linalg.norm(tangent_projection) <= _VECTOR_ATOL:
        raise ValueError("reference_direction must not be parallel to outward_normal")
    tangent_a = normalize(tangent_projection)
    tangent_b = normalize(np.cross(normal, tangent_a))
    return tangent_a, tangent_b


def _angle_radians(unit_a: ArrayLike, unit_b: ArrayLike) -> float:
    a = _unit_vector3(unit_a, "unit_a")
    b = _unit_vector3(unit_b, "unit_b")
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


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
