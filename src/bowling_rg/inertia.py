"""Verified, bowling-agnostic rigid-body inertia mathematics in SI units."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import BallSpec

GRAMS_PER_KILOGRAM = 1000.0
METERS_PER_INCH = 0.0254
_SYMMETRY_ATOL = 1e-12
_ORTHONORMAL_ATOL = 1e-12

FloatArray = NDArray[np.float64]


class PrincipalProperties(NamedTuple):
    """Sorted principal moments, axis vectors, and radii of gyration.

    Eigenvectors are stored as columns, matching :func:`numpy.linalg.eigh`.
    Moments and RG values use kg·m² and meters, respectively.
    """

    eigenvalues: FloatArray
    eigenvectors: FloatArray
    rg_values_m: FloatArray


def grams_to_kilograms(mass_g: float) -> float:
    """Convert mass from grams to kilograms."""
    return _finite_scalar(mass_g, "mass_g") / GRAMS_PER_KILOGRAM


def inches_to_meters(length_in: float) -> float:
    """Convert length from inches to meters."""
    return _finite_scalar(length_in, "length_in") * METERS_PER_INCH


def meters_to_inches(length_m: float) -> float:
    """Convert length from meters to inches."""
    return _finite_scalar(length_m, "length_m") / METERS_PER_INCH


def rg_to_moment(mass_kg: float, rg_m: float) -> float:
    """Convert a radius of gyration to a principal moment in kg·m²."""
    mass = _positive_scalar(mass_kg, "mass_kg")
    rg = _positive_scalar(rg_m, "rg_m")
    return mass * rg**2


def moment_to_rg(moment: float, mass_kg: float) -> float:
    """Convert a positive principal moment in kg·m² to RG in meters."""
    inertia = _positive_scalar(moment, "moment")
    mass = _positive_scalar(mass_kg, "mass_kg")
    return float(np.sqrt(inertia / mass))


def principal_tensor(ix: float, iy: float, iz: float) -> FloatArray:
    """Construct a diagonal principal inertia tensor in kg·m²."""
    moments = [
        _positive_scalar(ix, "ix"),
        _positive_scalar(iy, "iy"),
        _positive_scalar(iz, "iz"),
    ]
    return np.diag(np.asarray(moments, dtype=np.float64))


def rotate_tensor(tensor: ArrayLike, rotation_matrix: ArrayLike) -> FloatArray:
    """Rotate an inertia tensor using ``R @ tensor @ R.T``.

    The rotation must be a finite, proper orthonormal matrix.
    """
    inertia = _symmetric_tensor(tensor)
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    if not np.allclose(
        rotation.T @ rotation, np.identity(3), atol=_ORTHONORMAL_ATOL, rtol=0.0
    ):
        raise ValueError("rotation_matrix must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ORTHONORMAL_ATOL, rtol=0.0):
        raise ValueError("rotation_matrix must be a proper rotation with determinant +1")
    return _symmetrize(rotation @ inertia @ rotation.T)


def parallel_axis_shift(mass_kg: float, displacement_m: ArrayLike) -> FloatArray:
    """Return the parallel-axis tensor for displacement from a centroid."""
    mass = _positive_scalar(mass_kg, "mass_kg")
    displacement = _vector3(displacement_m, "displacement_m")
    return mass * (
        np.dot(displacement, displacement) * np.identity(3)
        - np.outer(displacement, displacement)
    )


def solid_cylinder_centroid_tensor(
    mass_kg: float,
    radius_m: float,
    length_m: float,
    axis_direction: ArrayLike,
) -> FloatArray:
    """Construct a finite solid cylinder tensor about its centroid.

    The vector expression works for any normalized cylinder axis and does not
    branch on alignment with a coordinate axis.
    """
    mass = _positive_scalar(mass_kg, "mass_kg")
    radius = _positive_scalar(radius_m, "radius_m")
    length = _positive_scalar(length_m, "length_m")
    axis = _unit_vector3(axis_direction, "axis_direction")

    i_axis = 0.5 * mass * radius**2
    i_perpendicular = mass * (3.0 * radius**2 + length**2) / 12.0
    tensor = (
        i_perpendicular * np.identity(3)
        + (i_axis - i_perpendicular) * np.outer(axis, axis)
    )
    return _symmetrize(tensor)


def translate_tensor(
    tensor_at_centroid: ArrayLike,
    mass_kg: float,
    displacement_m: ArrayLike,
) -> FloatArray:
    """Translate a centroidal tensor to a parallel reference point."""
    inertia = _symmetric_tensor(tensor_at_centroid)
    return _symmetrize(inertia + parallel_axis_shift(mass_kg, displacement_m))


def principal_properties(tensor: ArrayLike, mass_kg: float) -> PrincipalProperties:
    """Return sorted principal moments, axes, and RG values for a tensor.

    ``numpy.linalg.eigh`` is used because inertia tensors are real symmetric.
    Eigenvectors are reordered by column to stay paired with sorted moments.
    """
    inertia = _symmetric_tensor(tensor)
    mass = _positive_scalar(mass_kg, "mass_kg")
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
    tolerance = 1e-12 * scale
    if np.any(eigenvalues <= tolerance):
        raise ValueError("inertia tensor eigenvalues must be positive")

    rg_values = np.sqrt(eigenvalues / mass)
    return PrincipalProperties(eigenvalues, eigenvectors, rg_values)


def build_undrilled_inertia_tensor(ball: BallSpec) -> None:
    """Reserve BallSpec conversion until bowling-specific rules are verified."""
    raise NotImplementedError("undrilled inertia physics has not been specified")


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


def _vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3D vector")
    return vector


def _unit_vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = _vector3(value, name)
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be normalized")
    return vector


def _symmetric_tensor(value: ArrayLike) -> FloatArray:
    tensor = np.asarray(value, dtype=np.float64)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("tensor must be a finite 3x3 array")
    if not np.allclose(tensor, tensor.T, atol=_SYMMETRY_ATOL, rtol=1e-12):
        raise ValueError("tensor must be symmetric")
    return tensor


def _symmetrize(tensor: FloatArray) -> FloatArray:
    return 0.5 * (tensor + tensor.T)
