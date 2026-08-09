"""Mass-property conversion for cylindrical holes and inserts."""

from __future__ import annotations

import math

import numpy as np

from .inertia import (
    GRAM_TO_KG,
    G_PER_CM3_TO_KG_PER_M3,
    INCH_TO_M,
    MassProperties,
    cylinder_mass_properties,
)
from .models import HoleSpec, InsertSpec


def calculate_hole_mass_properties(hole: HoleSpec) -> MassProperties:
    """Return removed-cylinder properties about the ball-frame origin in SI."""
    radius_m = 0.5 * hole.diameter_in * INCH_TO_M
    length_m = hole.depth_in * INCH_TO_M
    density_kg_m3 = hole.removed_material_density_g_cm3 * G_PER_CM3_TO_KG_PER_M3
    mass_kg = density_kg_m3 * math.pi * radius_m**2 * length_m
    axis = np.asarray(hole.axis_direction, dtype=np.float64)
    entry_m = np.asarray(hole.surface_entry_position, dtype=np.float64) * INCH_TO_M
    centroid_m = entry_m + 0.5 * length_m * axis
    return cylinder_mass_properties(mass_kg, radius_m, length_m, axis, centroid_m)


def calculate_insert_mass_properties(insert: InsertSpec) -> MassProperties:
    """Return added-cylinder properties about the ball-frame origin in SI."""
    if insert.geometry.shape.casefold() != "cylinder":
        raise ValueError("only cylinder insert geometry is supported")
    try:
        diameter_in = insert.geometry.dimensions_in["diameter"]
        depth_in = insert.geometry.dimensions_in["depth"]
    except KeyError as error:
        raise ValueError(
            "cylinder insert geometry requires diameter and depth dimensions"
        ) from error

    return cylinder_mass_properties(
        mass_kg=insert.mass_g * GRAM_TO_KG,
        radius_m=0.5 * diameter_in * INCH_TO_M,
        length_m=depth_in * INCH_TO_M,
        axis_direction=insert.orientation,
        centroid_m=np.asarray(insert.location, dtype=np.float64) * INCH_TO_M,
    )
