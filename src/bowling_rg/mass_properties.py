"""Generic rigid-body mass-property composition in SI units."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inertia import (
    PrincipalProperties,
    parallel_axis_shift,
    principal_properties,
    solid_cylinder_centroid_tensor,
)

FloatArray = NDArray[np.float64]
_SYMMETRY_ATOL = 1e-12


@dataclass(frozen=True)
class MassProperties:
    """Mass, first moment, and inertia relative to one Cartesian origin."""

    mass_kg: float
    first_moment_kg_m: FloatArray
    inertia_about_origin_kg_m2: FloatArray

    def __post_init__(self) -> None:
        mass = _finite_scalar(self.mass_kg, "mass_kg")
        first_moment = _vector3(self.first_moment_kg_m, "first_moment_kg_m")
        inertia = _symmetric_tensor(
            self.inertia_about_origin_kg_m2,
            "inertia_about_origin_kg_m2",
        )

        # Own immutable copies so callers cannot change composed properties through
        # arrays retained outside this frozen dataclass.
        first_moment.setflags(write=False)
        inertia.setflags(write=False)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "first_moment_kg_m", first_moment)
        object.__setattr__(self, "inertia_about_origin_kg_m2", inertia)


def mass_properties_from_centroid(
    mass_kg: float,
    centroid_m: ArrayLike,
    inertia_at_centroid: ArrayLike,
) -> MassProperties:
    """Create mass properties about the fixed origin from centroidal data."""
    mass = _positive_scalar(mass_kg, "mass_kg")
    centroid = _vector3(centroid_m, "centroid_m")
    centroidal_inertia = _symmetric_tensor(inertia_at_centroid, "inertia_at_centroid")
    return MassProperties(
        mass_kg=mass,
        first_moment_kg_m=mass * centroid,
        inertia_about_origin_kg_m2=(
            centroidal_inertia + parallel_axis_shift(mass, centroid)
        ),
    )


def combine_mass_properties(parts: Iterable[MassProperties]) -> MassProperties:
    """Combine bodies expressed relative to the same Cartesian origin."""
    mass = 0.0
    first_moment = np.zeros(3, dtype=np.float64)
    inertia = np.zeros((3, 3), dtype=np.float64)
    for part in parts:
        _require_properties(part, "part")
        mass += part.mass_kg
        first_moment += part.first_moment_kg_m
        inertia += part.inertia_about_origin_kg_m2

    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("total mass must be positive")
    return MassProperties(mass, first_moment, inertia)


def subtract_mass_properties(
    base: MassProperties, removed: MassProperties
) -> MassProperties:
    """Subtract a removed body's properties using signed composition."""
    _require_properties(base, "base")
    _require_properties(removed, "removed")
    remaining_mass = base.mass_kg - removed.mass_kg
    if not np.isfinite(remaining_mass) or remaining_mass <= 0:
        raise ValueError("resulting mass must be positive")
    return MassProperties(
        remaining_mass,
        base.first_moment_kg_m - removed.first_moment_kg_m,
        (base.inertia_about_origin_kg_m2 - removed.inertia_about_origin_kg_m2),
    )


def center_of_mass(properties: MassProperties) -> FloatArray:
    """Return the center of mass relative to the fixed origin, in meters."""
    _require_positive_properties(properties)
    return properties.first_moment_kg_m / properties.mass_kg


def inertia_at_center_of_mass(properties: MassProperties) -> FloatArray:
    """Return the combined inertia tensor about its center of mass."""
    com = center_of_mass(properties)
    tensor = properties.inertia_about_origin_kg_m2 - parallel_axis_shift(
        properties.mass_kg, com
    )
    return 0.5 * (tensor + tensor.T)


def principal_properties_at_com(properties: MassProperties) -> PrincipalProperties:
    """Return principal moments, axes, and radii of gyration at the COM."""
    return principal_properties(
        inertia_at_center_of_mass(properties), properties.mass_kg
    )


def cylinder_mass_from_density(
    radius_m: float, length_m: float, density_kg_m3: float
) -> float:
    """Return a solid cylinder's mass from its SI dimensions and density."""
    radius = _positive_scalar(radius_m, "radius_m")
    length = _positive_scalar(length_m, "length_m")
    density = _positive_scalar(density_kg_m3, "density_kg_m3")
    return float(density * np.pi * radius**2 * length)


def cylinder_mass_properties(
    radius_m: float,
    length_m: float,
    density_kg_m3: float,
    centroid_m: ArrayLike,
    axis_direction: ArrayLike,
) -> MassProperties:
    """Return solid-cylinder properties relative to the fixed origin."""
    mass = cylinder_mass_from_density(radius_m, length_m, density_kg_m3)
    centroidal_inertia = solid_cylinder_centroid_tensor(
        mass, radius_m, length_m, axis_direction
    )
    return mass_properties_from_centroid(mass, centroid_m, centroidal_inertia)


def point_mass_properties(mass_kg: float, position_m: ArrayLike) -> MassProperties:
    """Return point-mass properties relative to the fixed origin."""
    return mass_properties_from_centroid(
        mass_kg, position_m, np.zeros((3, 3), dtype=np.float64)
    )


def _require_properties(value: MassProperties, name: str) -> None:
    if not isinstance(value, MassProperties):
        raise TypeError(f"{name} must be MassProperties")


def _require_positive_properties(value: MassProperties) -> None:
    _require_properties(value, "properties")
    if value.mass_kg <= 0:
        raise ValueError("mass must be positive")


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
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3D vector")
    return vector


def _symmetric_tensor(value: ArrayLike, name: str) -> FloatArray:
    tensor = np.array(value, dtype=np.float64, copy=True)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError(f"{name} must be a finite 3x3 tensor")
    if not np.allclose(tensor, tensor.T, atol=_SYMMETRY_ATOL, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (tensor + tensor.T)
