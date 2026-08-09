"""Compose factory bowling-ball properties with explicit holes and hardware.

All physics uses SI units.  Every removed hole mass remains visible in the
result.  Version 1 treats installed hardware as point masses at supplied
positions; it does not infer insert geometry or calibrate material density from
an actual finished mass.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bowling_physics import factory_mass_properties
from .hole_pitch import DrilledHoleGeometry
from .inertia import grams_to_kilograms, meters_to_inches
from .mass_properties import (
    MassProperties,
    center_of_mass,
    combine_mass_properties,
    cylinder_mass_properties,
    point_mass_properties,
    principal_properties_at_com,
    subtract_mass_properties,
)
from .models import BallSpec

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MaterialDensityModel:
    """Explicit default and per-hole removed-material densities."""

    default_density_kg_m3: float
    per_hole_density_kg_m3: dict[str, float] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        default = _positive_scalar(self.default_density_kg_m3, "default_density_kg_m3")
        densities: dict[str, float] = {}
        for name, density in self.per_hole_density_kg_m3.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("per-hole density names must not be blank")
            densities[name] = _positive_scalar(density, f"density for hole {name!r}")
        diagnostics = _diagnostics(self.diagnostics)
        object.__setattr__(self, "default_density_kg_m3", default)
        object.__setattr__(self, "per_hole_density_kg_m3", densities)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True)
class HardwareMass:
    """Installed hardware represented as a version-1 point mass."""

    name: str
    mass_g: float
    position_m: FloatArray
    geometry_mode: Literal["point_mass"] = "point_mass"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must not be blank")
        if self.geometry_mode != "point_mass":
            raise ValueError("geometry_mode must be 'point_mass' for version 1")
        position = _vector3(self.position_m, "position_m").copy()
        position.setflags(write=False)
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "mass_g", _positive_scalar(self.mass_g, "mass_g"))
        object.__setattr__(self, "position_m", position)


@dataclass(frozen=True)
class FinishedPrincipalProperties:
    """Sorted finished RGs, differentials, and principal axes."""

    low_rg_in: float
    intermediate_rg_in: float
    high_rg_in: float
    total_diff: float
    intermediate_diff: float
    principal_axes: FloatArray


@dataclass(frozen=True)
class DrilledBallResult:
    """Finished mass accounting and center-of-mass principal properties."""

    factory_mass_kg: float
    removed_mass_kg: float
    added_mass_kg: float
    finished_mass_kg: float
    center_of_mass_m: FloatArray
    low_rg_in: float
    intermediate_rg_in: float
    high_rg_in: float
    total_diff: float
    intermediate_diff: float
    principal_axes: FloatArray
    hole_removed_masses: dict[str, float]
    diagnostics: list[str]
    predicted_finished_mass_g: float
    actual_finished_mass_g: Optional[float] = None  # noqa: UP045 - Python 3.9
    mass_residual_g: Optional[float] = None  # noqa: UP045 - Python 3.9

    def __post_init__(self) -> None:
        com = _vector3(self.center_of_mass_m, "center_of_mass_m").copy()
        axes = np.asarray(self.principal_axes, dtype=np.float64).copy()
        if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
            raise ValueError("principal_axes must be a finite 3x3 matrix")
        com.setflags(write=False)
        axes.setflags(write=False)
        object.__setattr__(self, "center_of_mass_m", com)
        object.__setattr__(self, "principal_axes", axes)
        object.__setattr__(self, "hole_removed_masses", dict(self.hole_removed_masses))
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))


def removed_mass_properties_for_hole(
    hole: DrilledHoleGeometry,
    density_kg_m3: float,
) -> MassProperties:
    """Return cylindrical removed-material properties for one explicit hole."""
    if not isinstance(hole, DrilledHoleGeometry):
        raise TypeError("hole must be DrilledHoleGeometry")
    density = _positive_scalar(density_kg_m3, "density_kg_m3")
    return cylinder_mass_properties(
        radius_m=hole.diameter_m / 2.0,
        length_m=hole.depth_m,
        density_kg_m3=density,
        centroid_m=hole.centroid_m,
        axis_direction=hole.drilling_axis_unit,
    )


def remove_holes_from_factory(
    factory_properties: MassProperties,
    holes: Iterable[DrilledHoleGeometry],
    density_model: MaterialDensityModel,
) -> tuple[MassProperties, dict[str, float]]:
    """Subtract every explicitly modeled cylindrical hole."""
    if not isinstance(factory_properties, MassProperties):
        raise TypeError("factory_properties must be MassProperties")
    if not isinstance(density_model, MaterialDensityModel):
        raise TypeError("density_model must be MaterialDensityModel")
    remaining = factory_properties
    removed_masses: dict[str, float] = {}
    for hole in holes:
        if not isinstance(hole, DrilledHoleGeometry):
            raise TypeError("holes must contain DrilledHoleGeometry values")
        if hole.name in removed_masses:
            raise ValueError(f"duplicate hole name: {hole.name}")
        density = density_model.per_hole_density_kg_m3.get(
            hole.name, density_model.default_density_kg_m3
        )
        removed = removed_mass_properties_for_hole(hole, density)
        remaining = subtract_mass_properties(remaining, removed)
        removed_masses[hole.name] = removed.mass_kg
    return remaining, removed_masses


def add_hardware_masses(
    properties: MassProperties,
    hardware: Iterable[HardwareMass],
) -> MassProperties:
    """Add installed hardware using the documented point-mass approximation."""
    if not isinstance(properties, MassProperties):
        raise TypeError("properties must be MassProperties")
    point_masses: list[MassProperties] = [properties]
    for item in hardware:
        if not isinstance(item, HardwareMass):
            raise TypeError("hardware must contain HardwareMass values")
        point_masses.append(
            point_mass_properties(grams_to_kilograms(item.mass_g), item.position_m)
        )
    return combine_mass_properties(point_masses)


def finished_principal_properties(
    properties: MassProperties,
) -> FinishedPrincipalProperties:
    """Return sorted COM principal RGs in inches and their differentials."""
    principal = principal_properties_at_com(properties)
    rgs = tuple(meters_to_inches(value) for value in principal.rg_values_m)
    low, intermediate, high = rgs
    axes = principal.eigenvectors.copy()
    axes.setflags(write=False)
    return FinishedPrincipalProperties(
        low,
        intermediate,
        high,
        high - low,
        intermediate - low,
        axes,
    )


def calculate_drilled_ball(
    ball_spec: BallSpec,
    holes: Iterable[DrilledHoleGeometry],
    density_model: MaterialDensityModel,
    hardware: Iterable[HardwareMass] = (),
    actual_finished_mass_g: Optional[float] = None,  # noqa: UP045 - Python 3.9
) -> DrilledBallResult:
    """Run factory -> hole subtraction -> hardware -> finished properties."""
    if not isinstance(ball_spec, BallSpec):
        raise TypeError("ball_spec must be BallSpec")
    holes_list = list(holes)
    hardware_list = list(hardware)
    factory = factory_mass_properties(ball_spec)
    drilled, hole_masses = remove_holes_from_factory(factory, holes_list, density_model)
    finished = add_hardware_masses(drilled, hardware_list)
    principal = finished_principal_properties(finished)
    removed_mass = float(sum(hole_masses.values()))
    added_mass = float(sum(grams_to_kilograms(item.mass_g) for item in hardware_list))
    predicted_mass_g = finished.mass_kg * 1000.0
    actual_mass = None
    residual = None
    diagnostics = list(density_model.diagnostics)
    defaulted_holes = [
        hole.name
        for hole in holes_list
        if hole.name not in density_model.per_hole_density_kg_m3
    ]
    if defaulted_holes:
        diagnostics.append(
            "default density used for holes: " + ", ".join(defaulted_holes)
        )
    if hardware_list:
        diagnostics.append("hardware modeled as point masses for version 1")
    if actual_finished_mass_g is not None:
        actual_mass = _positive_scalar(actual_finished_mass_g, "actual_finished_mass_g")
        residual = actual_mass - predicted_mass_g
        diagnostics.append("actual mass reported without automatic density calibration")

    return DrilledBallResult(
        factory_mass_kg=factory.mass_kg,
        removed_mass_kg=removed_mass,
        added_mass_kg=added_mass,
        finished_mass_kg=finished.mass_kg,
        center_of_mass_m=center_of_mass(finished),
        low_rg_in=principal.low_rg_in,
        intermediate_rg_in=principal.intermediate_rg_in,
        high_rg_in=principal.high_rg_in,
        total_diff=principal.total_diff,
        intermediate_diff=principal.intermediate_diff,
        principal_axes=principal.principal_axes,
        hole_removed_masses=hole_masses,
        diagnostics=diagnostics,
        predicted_finished_mass_g=predicted_mass_g,
        actual_finished_mass_g=actual_mass,
        mass_residual_g=residual,
    )


def _diagnostics(values: Iterable[str]) -> list[str]:
    diagnostics = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in diagnostics):
        raise ValueError("diagnostics cannot contain blank messages")
    return diagnostics


def _vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3D vector")
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
