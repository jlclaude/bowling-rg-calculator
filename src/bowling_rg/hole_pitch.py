"""Convert bowling pitch measurements into three-dimensional drilling axes.

Pitch convention
----------------
Positive ``forward_reverse_in`` is forward; negative is reverse.  Positive
``lateral_in`` is right; negative is left.  Each value is interpreted as a
linear offset over a configurable 12-inch reference length.  This convention is
isolated here so a verified pro-shop/manufacturer convention can replace it.

The convenience grip bases orient their longitudinal vector *away* from the
opposing grip reference.  Because drilling-axis construction subtracts the
positive pitch component, positive forward pitch tilts the axis toward the grip
interior.  Finger lateral reference points toward the opposite finger; positive
lateral pitch therefore tilts away from it.  Thumb lateral positive is the
right-handed ``cross(entry, longitudinal)`` direction.  These vector rules
mirror with mirrored grip geometry and do not assume global X/Y axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inertia import inches_to_meters
from .sphere_geometry import (
    hole_centroid,
    hole_end_point,
    surface_point,
    validate_hole_inside_sphere,
)

FloatArray = NDArray[np.float64]
_VECTOR_ATOL = 1e-12
_GEOMETRY_ATOL = 1e-10


@dataclass(frozen=True)
class HolePitch:
    """Signed forward/reverse and right/left bowling pitch, in inches."""

    forward_reverse_in: float
    lateral_in: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forward_reverse_in",
            _finite_scalar(self.forward_reverse_in, "forward_reverse_in"),
        )
        object.__setattr__(
            self,
            "lateral_in",
            _finite_scalar(self.lateral_in, "lateral_in"),
        )


@dataclass(frozen=True)
class DrilledHoleGeometry:
    """Complete SI geometry for one pitched cylindrical hole."""

    name: str
    entry_unit: FloatArray
    entry_position_m: FloatArray
    drilling_axis_unit: FloatArray
    depth_m: float
    diameter_m: float
    centroid_m: FloatArray
    endpoint_m: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must not be blank")
        object.__setattr__(self, "name", self.name.strip())
        entry_unit = _unit_vector3(self.entry_unit, "entry_unit").copy()
        drilling_axis = _unit_vector3(
            self.drilling_axis_unit, "drilling_axis_unit"
        ).copy()
        entry_position = _vector3(self.entry_position_m, "entry_position_m").copy()
        centroid = _vector3(self.centroid_m, "centroid_m").copy()
        endpoint = _vector3(self.endpoint_m, "endpoint_m").copy()
        for vector in (
            entry_unit,
            drilling_axis,
            entry_position,
            centroid,
            endpoint,
        ):
            vector.setflags(write=False)
        object.__setattr__(self, "entry_unit", entry_unit)
        object.__setattr__(self, "drilling_axis_unit", drilling_axis)
        object.__setattr__(self, "entry_position_m", entry_position)
        object.__setattr__(self, "centroid_m", centroid)
        object.__setattr__(self, "endpoint_m", endpoint)
        object.__setattr__(self, "depth_m", _positive_scalar(self.depth_m, "depth_m"))
        object.__setattr__(
            self, "diameter_m", _positive_scalar(self.diameter_m, "diameter_m")
        )


def local_tangent_basis_for_hole(
    entry_unit: ArrayLike,
    toward_fingers_reference: ArrayLike,
    lateral_reference: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Build longitudinal and lateral tangents from explicit surface references."""
    entry = _unit_vector3(entry_unit, "entry_unit")
    longitudinal = _tangent_toward(
        entry, toward_fingers_reference, "toward_fingers_reference"
    )
    lateral_candidate = _tangent_toward(entry, lateral_reference, "lateral_reference")
    lateral_projection = (
        lateral_candidate - np.dot(lateral_candidate, longitudinal) * longitudinal
    )
    if np.linalg.norm(lateral_projection) <= _VECTOR_ATOL:
        raise ValueError("lateral reference must define an independent tangent")
    lateral = _normalize(lateral_projection)
    return longitudinal, lateral


def bowling_pitch_to_angles(
    pitch: HolePitch,
    reference_length_in: float = 12.0,
) -> tuple[float, float]:
    """Convert linear pitch offsets to radians using ``atan(offset/reference)``."""
    if not isinstance(pitch, HolePitch):
        raise TypeError("pitch must be HolePitch")
    reference = _positive_scalar(reference_length_in, "reference_length_in")
    return (
        float(np.arctan(pitch.forward_reverse_in / reference)),
        float(np.arctan(pitch.lateral_in / reference)),
    )


def drilling_axis_from_two_component_pitch(
    entry_unit: ArrayLike,
    longitudinal_tangent: ArrayLike,
    lateral_tangent: ArrayLike,
    pitch: HolePitch,
) -> FloatArray:
    """Construct an inward axis by simultaneous tangent-vector offsets."""
    entry = _unit_vector3(entry_unit, "entry_unit")
    longitudinal = _unit_vector3(longitudinal_tangent, "longitudinal_tangent")
    lateral = _unit_vector3(lateral_tangent, "lateral_tangent")
    _validate_tangent_basis(entry, longitudinal, lateral)
    forward_angle, lateral_angle = bowling_pitch_to_angles(pitch)
    raw_axis = (
        -entry - np.tan(forward_angle) * longitudinal - np.tan(lateral_angle) * lateral
    )
    axis = _normalize(raw_axis)
    if np.dot(axis, entry) >= 0:
        raise ValueError("pitched drilling axis must point inward")
    return axis


def make_drilled_hole_geometry(
    name: str,
    entry_unit: ArrayLike,
    ball_radius_m: float,
    diameter_in: float,
    depth_in: float,
    pitch: HolePitch,
    longitudinal_reference: ArrayLike,
    lateral_reference: ArrayLike,
) -> DrilledHoleGeometry:
    """Build and validate SI hole geometry from surface and bowling inputs."""
    entry = _unit_vector3(entry_unit, "entry_unit")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    diameter_m = inches_to_meters(_positive_scalar(diameter_in, "diameter_in"))
    depth_m = inches_to_meters(_positive_scalar(depth_in, "depth_in"))
    longitudinal, lateral = local_tangent_basis_for_hole(
        entry, longitudinal_reference, lateral_reference
    )
    axis = drilling_axis_from_two_component_pitch(entry, longitudinal, lateral, pitch)
    entry_position = surface_point(radius, entry)
    centroid = hole_centroid(entry_position, axis, depth_m)
    endpoint = hole_end_point(entry_position, axis, depth_m)
    validate_hole_inside_sphere(
        radius,
        entry_position,
        axis,
        depth_m,
        diameter_m / 2.0,
    )
    geometry = DrilledHoleGeometry(
        name,
        entry,
        entry_position,
        axis,
        depth_m,
        diameter_m,
        centroid,
        endpoint,
    )
    validate_drilled_hole_geometry(
        geometry,
        radius,
        pitch,
        longitudinal,
        lateral,
    )
    return geometry


def validate_drilled_hole_geometry(
    geometry: DrilledHoleGeometry,
    ball_radius_m: float,
    pitch: HolePitch,
    longitudinal_tangent: ArrayLike,
    lateral_tangent: ArrayLike,
    reference_length_in: float = 12.0,
) -> tuple[str, ...]:
    """Validate physical consistency and recover both pitch component angles."""
    if not isinstance(geometry, DrilledHoleGeometry):
        raise TypeError("geometry must be DrilledHoleGeometry")
    radius = _positive_scalar(ball_radius_m, "ball_radius_m")
    longitudinal = _unit_vector3(longitudinal_tangent, "longitudinal_tangent")
    lateral = _unit_vector3(lateral_tangent, "lateral_tangent")
    _validate_tangent_basis(geometry.entry_unit, longitudinal, lateral)
    if np.dot(geometry.drilling_axis_unit, geometry.entry_unit) >= 0:
        raise ValueError("drilling axis must point inward")
    if not np.allclose(
        geometry.entry_position_m,
        radius * geometry.entry_unit,
        atol=_GEOMETRY_ATOL,
        rtol=1e-12,
    ):
        raise ValueError("entry position is inconsistent with sphere radius")
    expected_centroid = hole_centroid(
        geometry.entry_position_m,
        geometry.drilling_axis_unit,
        geometry.depth_m,
    )
    expected_endpoint = hole_end_point(
        geometry.entry_position_m,
        geometry.drilling_axis_unit,
        geometry.depth_m,
    )
    if not np.allclose(
        geometry.centroid_m, expected_centroid, atol=_GEOMETRY_ATOL, rtol=1e-12
    ):
        raise ValueError("hole centroid is inconsistent with entry, axis, and depth")
    if not np.allclose(
        geometry.endpoint_m, expected_endpoint, atol=_GEOMETRY_ATOL, rtol=1e-12
    ):
        raise ValueError("hole endpoint is inconsistent with entry, axis, and depth")
    validate_hole_inside_sphere(
        radius,
        geometry.entry_position_m,
        geometry.drilling_axis_unit,
        geometry.depth_m,
        geometry.diameter_m / 2.0,
    )

    radial_component = -float(np.dot(geometry.drilling_axis_unit, geometry.entry_unit))
    recovered_forward = float(
        np.arctan(-np.dot(geometry.drilling_axis_unit, longitudinal) / radial_component)
    )
    recovered_lateral = float(
        np.arctan(-np.dot(geometry.drilling_axis_unit, lateral) / radial_component)
    )
    expected_forward, expected_lateral = bowling_pitch_to_angles(
        pitch, reference_length_in
    )
    if not np.allclose(
        [recovered_forward, recovered_lateral],
        [expected_forward, expected_lateral],
        atol=_GEOMETRY_ATOL,
        rtol=1e-10,
    ):
        raise ValueError("drilling axis does not round-trip to requested pitch angles")
    return ()


def finger_pitch_basis(
    hole_entry: ArrayLike,
    thumb_entry: ArrayLike,
    opposite_finger_entry: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return finger pitch basis with positive forward tilting toward thumb.

    Longitudinal basis points away from thumb, so the axis formula's minus sign
    makes positive pitch tilt toward thumb.  Lateral basis points toward the
    opposite finger, making positive pitch tilt away from that finger.
    """
    entry = _unit_vector3(hole_entry, "hole_entry")
    toward_thumb = _tangent_toward(entry, thumb_entry, "thumb_entry")
    longitudinal_reference = _normalize(entry - toward_thumb)
    # ``entry - tangent`` is a finite surface-direction reference whose tangent
    # projection is exactly away from thumb.
    return local_tangent_basis_for_hole(
        entry,
        longitudinal_reference,
        opposite_finger_entry,
    )


def thumb_pitch_basis(
    thumb_entry: ArrayLike,
    finger_pair_midpoint: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return thumb basis with positive forward tilting toward the finger pair.

    Longitudinal basis points away from the normalized finger-pair midpoint.
    Positive lateral uses ``cross(entry, longitudinal)``; this explicit local
    orientation mirrors when the entire grip geometry is mirrored.
    """
    thumb = _unit_vector3(thumb_entry, "thumb_entry")
    toward_fingers = _tangent_toward(
        thumb, finger_pair_midpoint, "finger_pair_midpoint"
    )
    longitudinal = -toward_fingers
    lateral = _normalize(np.cross(thumb, longitudinal))
    return longitudinal, lateral


def _validate_tangent_basis(
    entry: FloatArray, longitudinal: FloatArray, lateral: FloatArray
) -> None:
    if not np.allclose(
        [
            np.dot(entry, longitudinal),
            np.dot(entry, lateral),
            np.dot(longitudinal, lateral),
        ],
        0.0,
        atol=_VECTOR_ATOL,
        rtol=0.0,
    ):
        raise ValueError("entry, longitudinal, and lateral vectors must be orthogonal")


def _tangent_toward(entry: FloatArray, reference: ArrayLike, name: str) -> FloatArray:
    target = _unit_vector3(reference, name)
    projection = target - np.dot(target, entry) * entry
    if np.linalg.norm(projection) <= _VECTOR_ATOL:
        raise ValueError(f"{name} cannot be coincident or antipodal with entry")
    return _normalize(projection)


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
