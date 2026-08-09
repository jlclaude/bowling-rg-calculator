"""High-level bowling-ball mass-properties calculation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .drilling import calculate_hole_mass_properties, calculate_insert_mass_properties
from .inertia import GRAM_TO_KG, INCH_TO_M, build_undrilled_inertia_tensor
from .models import BallSpec, CalculationResult, HoleSpec, InsertSpec


def calculate_finished_ball(
    ball: BallSpec,
    holes: Sequence[HoleSpec],
    inserts: Sequence[InsertSpec] = (),
) -> CalculationResult:
    """Calculate finished mass, center of mass, principal axes, and RG values.

    Hole and insert coordinates are consumed directly in the ball coordinate
    frame. No PAP or drilling-layout mapping is performed here.
    """
    mass_kg = ball.gross_mass_g * GRAM_TO_KG
    first_moment = np.zeros(3, dtype=np.float64)
    inertia_at_origin = build_undrilled_inertia_tensor(ball)

    for hole in holes:
        removed = calculate_hole_mass_properties(hole)
        mass_kg -= removed.mass_kg
        first_moment -= removed.first_moment_kg_m
        inertia_at_origin -= removed.inertia_kg_m2

    for insert in inserts:
        added = calculate_insert_mass_properties(insert)
        mass_kg += added.mass_kg
        first_moment += added.first_moment_kg_m
        inertia_at_origin += added.inertia_kg_m2

    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("holes remove all or more than the ball's gross mass")

    center_of_mass_m = first_moment / mass_kg
    shift = mass_kg * (
        np.dot(center_of_mass_m, center_of_mass_m) * np.identity(3)
        - np.outer(center_of_mass_m, center_of_mass_m)
    )
    inertia_at_cm = inertia_at_origin - shift
    inertia_at_cm = (inertia_at_cm + inertia_at_cm.T) * 0.5

    eigenvalues, eigenvectors = np.linalg.eigh(inertia_at_cm)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise ValueError("finished inertia tensor must have positive real eigenvalues")

    radii_in = np.sqrt(eigenvalues / mass_kg) / INCH_TO_M
    principal_axes = tuple(
        tuple(float(component) for component in eigenvectors[:, index])
        for index in range(3)
    )
    center_of_mass_in = tuple(float(value / INCH_TO_M) for value in center_of_mass_m)

    return CalculationResult(
        finished_mass_g=mass_kg / GRAM_TO_KG,
        low_rg=float(radii_in[0]),
        intermediate_rg=float(radii_in[1]),
        high_rg=float(radii_in[2]),
        total_diff=float(radii_in[2] - radii_in[0]),
        intermediate_diff=float(radii_in[1] - radii_in[0]),
        principal_axis_vectors=principal_axes,
        calculated_center_of_mass=center_of_mass_in,
        warnings=[],
    )
