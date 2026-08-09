"""SI-unit inertia tensor primitives for the bowling-ball engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import as_vector3
from .models import BallSpec

INCH_TO_M = 0.0254
GRAM_TO_KG = 0.001
G_PER_CM3_TO_KG_PER_M3 = 1000.0

Tensor3 = NDArray[np.float64]
Vector3Array = NDArray[np.float64]


@dataclass(frozen=True)
class MassProperties:
    """Mass, first moment, and inertia about a common SI-frame origin."""

    mass_kg: float
    first_moment_kg_m: Vector3Array
    inertia_kg_m2: Tensor3


def build_undrilled_inertia_tensor(ball: BallSpec) -> Tensor3:
    """Return the ball's principal-axis inertia tensor in kg·m².

    The X, Y, and Z axes correspond to low, intermediate, and high RG,
    respectively. Manufacturer differentials are treated as RG differences in
    inches, as represented by :class:`BallSpec`.
    """
    mass_kg = ball.gross_mass_g * GRAM_TO_KG
    radii_m = np.array(
        [
            ball.low_rg_in,
            ball.low_rg_in + ball.intermediate_diff,
            ball.low_rg_in + ball.total_diff,
        ],
        dtype=np.float64,
    ) * INCH_TO_M
    return np.diag(mass_kg * np.square(radii_m))


def rotate_inertia_tensor(inertia_body: ArrayLike, rotation: ArrayLike) -> Tensor3:
    """Rotate a body-frame tensor into a world frame with ``R I Rᵀ``."""
    tensor = _as_symmetric_tensor(inertia_body)
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix @ matrix.T, np.identity(3), atol=1e-10):
        raise ValueError("rotation must be orthogonal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10):
        raise ValueError("rotation must have determinant +1")
    rotated = matrix @ tensor @ matrix.T
    return _symmetrize(rotated)


def cylinder_centroid_inertia(
    mass_kg: float,
    radius_m: float,
    length_m: float,
    axis_direction: ArrayLike,
) -> Tensor3:
    """Return a finite solid cylinder tensor about its centroid in kg·m²."""
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be finite and positive")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("length_m must be finite and positive")
    axis = _as_unit_vector(axis_direction)
    i_axis = 0.5 * mass_kg * radius_m**2
    i_perpendicular = mass_kg * (3.0 * radius_m**2 + length_m**2) / 12.0
    tensor = (
        i_perpendicular * np.identity(3)
        + (i_axis - i_perpendicular) * np.outer(axis, axis)
    )
    return _symmetrize(tensor)


def translate_inertia_tensor(
    inertia_at_centroid: ArrayLike,
    mass_kg: float,
    displacement_m: ArrayLike,
) -> Tensor3:
    """Translate a centroidal tensor by ``d`` using the parallel-axis theorem."""
    tensor = _as_symmetric_tensor(inertia_at_centroid)
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be finite and positive")
    displacement = as_vector3(displacement_m)
    offset = mass_kg * (
        np.dot(displacement, displacement) * np.identity(3)
        - np.outer(displacement, displacement)
    )
    return _symmetrize(tensor + offset)


def cylinder_mass_properties(
    mass_kg: float,
    radius_m: float,
    length_m: float,
    axis_direction: ArrayLike,
    centroid_m: ArrayLike,
) -> MassProperties:
    """Return cylinder mass properties about the ball-frame origin."""
    centroid = as_vector3(centroid_m)
    inertia_cm = cylinder_centroid_inertia(
        mass_kg, radius_m, length_m, axis_direction
    )
    return MassProperties(
        mass_kg=mass_kg,
        first_moment_kg_m=mass_kg * centroid,
        inertia_kg_m2=translate_inertia_tensor(inertia_cm, mass_kg, centroid),
    )


def _as_unit_vector(value: ArrayLike) -> Vector3Array:
    vector = as_vector3(value)
    magnitude = np.linalg.norm(vector)
    if not np.isclose(magnitude, 1.0, rtol=1e-9, atol=1e-9):
        raise ValueError("axis_direction must be a unit vector")
    return vector


def _as_symmetric_tensor(value: ArrayLike) -> Tensor3:
    tensor = np.asarray(value, dtype=np.float64)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("inertia tensor must be a finite 3x3 matrix")
    if not np.allclose(tensor, tensor.T, atol=1e-12):
        raise ValueError("inertia tensor must be symmetric")
    return tensor


def _symmetrize(tensor: Tensor3) -> Tensor3:
    """Remove harmless floating-point asymmetry before eigendecomposition."""
    return (tensor + tensor.T) * 0.5
