"""Calibrate one effective uniform hole-material density from measured mass.

Mass accounting follows the drilled-ball model::

    finished = undrilled - removed_material + installed_hardware

Therefore ``removed_material = undrilled + installed_hardware - finished``.
One total mass measurement identifies only this single uniform effective-density
parameter; it cannot identify independent per-hole densities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .drilled_ball import (
    DrilledBallResult,
    HardwareMass,
    MaterialDensityModel,
    calculate_drilled_ball,
)
from .hole_pitch import DrilledHoleGeometry
from .models import BallSpec

DEFAULT_PLAUSIBLE_DENSITY_RANGE_KG_M3 = (800.0, 2500.0)


@dataclass(frozen=True)
class MassCalibrationInput:
    """Measured masses used to identify uniform removed-material density."""

    undrilled_mass_g: float
    finished_mass_g: float
    installed_hardware_mass_g: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "undrilled_mass_g",
            _positive_scalar(self.undrilled_mass_g, "undrilled_mass_g"),
        )
        object.__setattr__(
            self,
            "finished_mass_g",
            _positive_scalar(self.finished_mass_g, "finished_mass_g"),
        )
        object.__setattr__(
            self,
            "installed_hardware_mass_g",
            _nonnegative_scalar(
                self.installed_hardware_mass_g, "installed_hardware_mass_g"
            ),
        )


@dataclass(frozen=True)
class MassCalibrationResult:
    """Uniform-density calibration accounting and diagnostics."""

    actual_net_mass_loss_g: float
    actual_removed_material_mass_g: float
    total_modeled_hole_volume_m3: float
    calibrated_uniform_density_kg_m3: float
    predicted_finished_mass_before_g: float
    predicted_finished_mass_after_g: float
    residual_before_g: float
    residual_after_g: float
    per_hole_volumes_m3: dict[str, float]
    diagnostics: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_hole_volumes_m3", dict(self.per_hole_volumes_m3))
        diagnostics = list(self.diagnostics)
        if any(not item.strip() for item in diagnostics):
            raise ValueError("diagnostics cannot contain blank messages")
        object.__setattr__(self, "diagnostics", diagnostics)


def hole_volume_m3(hole: DrilledHoleGeometry) -> float:
    """Return the modeled cylindrical volume of a drilled hole."""
    if not isinstance(hole, DrilledHoleGeometry):
        raise TypeError("hole must be DrilledHoleGeometry")
    radius = hole.diameter_m / 2.0
    return float(np.pi * radius**2 * hole.depth_m)


def total_hole_volume_m3(holes: Iterable[DrilledHoleGeometry]) -> float:
    """Return total modeled cylindrical hole volume."""
    return float(sum(hole_volume_m3(hole) for hole in holes))


def actual_removed_material_mass_g(calibration: MassCalibrationInput) -> float:
    """Return measured removed mass after accounting for added hardware.

    Hardware must be added to the observed net loss because it increased the
    finished mass: ``removed = undrilled + hardware - finished``.
    """
    if not isinstance(calibration, MassCalibrationInput):
        raise TypeError("calibration must be MassCalibrationInput")
    removed = (
        calibration.undrilled_mass_g
        + calibration.installed_hardware_mass_g
        - calibration.finished_mass_g
    )
    if removed <= 0:
        raise ValueError("calibration must imply positive removed-material mass")
    return removed


def calibrate_uniform_density(
    holes: Iterable[DrilledHoleGeometry],
    calibration: MassCalibrationInput,
) -> float:
    """Return the one identifiable effective uniform density in kg/m³."""
    holes_list = list(holes)
    volume = total_hole_volume_m3(holes_list)
    if volume <= 0:
        raise ValueError("total modeled hole volume must be positive")
    removed_mass_kg = actual_removed_material_mass_g(calibration) / 1000.0
    density = removed_mass_kg / volume
    if not np.isfinite(density) or density <= 0:
        raise ValueError("calibrated density must be finite and positive")
    return float(density)


def build_calibrated_density_model(
    holes: Iterable[DrilledHoleGeometry],
    calibration: MassCalibrationInput,
    plausible_density_range_kg_m3: tuple[float, float] = (
        DEFAULT_PLAUSIBLE_DENSITY_RANGE_KG_M3
    ),
) -> MaterialDensityModel:
    """Return a mass-calibrated effective uniform density model."""
    holes_list = list(holes)
    density = calibrate_uniform_density(holes_list, calibration)
    lower, upper = _plausible_range(plausible_density_range_kg_m3)
    diagnostics = ["mass-calibrated effective uniform density"]
    if not lower <= density <= upper:
        diagnostics.append(
            f"calibrated density {density:.6g} kg/m^3 is outside the "
            f"plausible range {lower:.6g} to {upper:.6g} kg/m^3"
        )
    diagnostics.append("one total mass measurement cannot identify per-hole densities")
    return MaterialDensityModel(
        default_density_kg_m3=density,
        per_hole_density_kg_m3={},
        diagnostics=diagnostics,
    )


def run_mass_calibrated_drilled_ball(
    ball_spec: BallSpec,
    holes: Iterable[DrilledHoleGeometry],
    hardware: Iterable[HardwareMass],
    calibration: MassCalibrationInput,
    plausible_density_range_kg_m3: tuple[float, float] = (
        DEFAULT_PLAUSIBLE_DENSITY_RANGE_KG_M3
    ),
) -> tuple[DrilledBallResult, MassCalibrationResult]:
    """Calibrate uniform density, run the ball model, and report residuals."""
    if not isinstance(ball_spec, BallSpec):
        raise TypeError("ball_spec must be BallSpec")
    holes_list = list(holes)
    hardware_list = list(hardware)
    hardware_total_g = float(sum(item.mass_g for item in hardware_list))
    if not np.isclose(
        hardware_total_g,
        calibration.installed_hardware_mass_g,
        atol=1e-9,
        rtol=1e-12,
    ):
        raise ValueError("calibration hardware mass must match supplied hardware")

    density_model = build_calibrated_density_model(
        holes_list, calibration, plausible_density_range_kg_m3
    )
    measured_mass_ball_spec = ball_spec.model_copy(
        update={"gross_mass_g": calibration.undrilled_mass_g}
    )
    drilled_result = calculate_drilled_ball(
        measured_mass_ball_spec,
        holes_list,
        density_model,
        hardware_list,
        actual_finished_mass_g=calibration.finished_mass_g,
    )
    volumes = _per_hole_volumes(holes_list)
    predicted_before = (
        calibration.undrilled_mass_g + calibration.installed_hardware_mass_g
    )
    predicted_after = drilled_result.predicted_finished_mass_g
    diagnostics = list(density_model.diagnostics)
    diagnostics.append("measured undrilled mass used for calibrated factory inertia")
    calibration_result = MassCalibrationResult(
        actual_net_mass_loss_g=(
            calibration.undrilled_mass_g - calibration.finished_mass_g
        ),
        actual_removed_material_mass_g=actual_removed_material_mass_g(calibration),
        total_modeled_hole_volume_m3=sum(volumes.values()),
        calibrated_uniform_density_kg_m3=density_model.default_density_kg_m3,
        predicted_finished_mass_before_g=predicted_before,
        predicted_finished_mass_after_g=predicted_after,
        residual_before_g=calibration.finished_mass_g - predicted_before,
        residual_after_g=calibration.finished_mass_g - predicted_after,
        per_hole_volumes_m3=volumes,
        diagnostics=diagnostics,
    )
    return drilled_result, calibration_result


def _per_hole_volumes(
    holes: Sequence[DrilledHoleGeometry],
) -> dict[str, float]:
    volumes: dict[str, float] = {}
    for hole in holes:
        if hole.name in volumes:
            raise ValueError(f"duplicate hole name: {hole.name}")
        volumes[hole.name] = hole_volume_m3(hole)
    return volumes


def _plausible_range(values: tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError("plausible density range must contain lower and upper bounds")
    lower = _positive_scalar(values[0], "plausible density lower bound")
    upper = _positive_scalar(values[1], "plausible density upper bound")
    if lower >= upper:
        raise ValueError("plausible density lower bound must be below upper bound")
    return lower, upper


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
