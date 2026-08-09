"""Validated, unit-explicit data models for bowling-ball calculations."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class EngineeringModel(BaseModel):
    """Base model configured for strict assignment and unknown-field checks."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class CoreType(str, Enum):
    """Supported bowling-ball core classifications."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class Handedness(str, Enum):
    """Bowler handedness used when interpreting layout coordinates."""

    RIGHT = "right"
    LEFT = "left"


class BallSpec(EngineeringModel):
    """Manufacturer-supplied properties of an undrilled bowling ball.

    RG values are in inches and differentials are dimensionless RG differences.
    """

    manufacturer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    nominal_weight_lb: PositiveFiniteFloat
    gross_mass_g: PositiveFiniteFloat
    low_rg_in: PositiveFiniteFloat
    total_diff: NonNegativeFiniteFloat
    intermediate_diff: NonNegativeFiniteFloat
    core_type: CoreType

    @field_validator("intermediate_diff")
    @classmethod
    def intermediate_not_above_total(cls, value: float, info: object) -> float:
        """Reject a stated intermediate differential above the total differential."""
        data = getattr(info, "data", {})
        total = data.get("total_diff")
        if total is not None and value > total:
            raise ValueError("intermediate_diff cannot exceed total_diff")
        return value


class BowlerSpec(EngineeringModel):
    """Bowler positive-axis-point (PAP) measurements and handedness."""

    pap_horizontal_in: FiniteFloat
    pap_vertical_in: FiniteFloat
    handedness: Handedness


class LayoutSpec(EngineeringModel):
    """Dual-angle drilling layout measurements in degrees and inches."""

    drilling_angle_deg: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)]
    pin_to_pap_in: PositiveFiniteFloat
    val_angle_deg: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)]


class HoleSpec(EngineeringModel):
    """Geometry and material information for a cylindrical drilled hole."""

    name: str = Field(min_length=1)
    diameter_in: PositiveFiniteFloat
    depth_in: PositiveFiniteFloat
    axis_direction: Vector3
    surface_entry_position: Vector3
    removed_material_density_g_cm3: PositiveFiniteFloat

    @field_validator("axis_direction")
    @classmethod
    def axis_must_be_unit_vector(cls, value: Vector3) -> Vector3:
        """Require a nonzero direction normalized to unit length."""
        magnitude = math.sqrt(sum(component * component for component in value))
        if not math.isclose(magnitude, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("axis_direction must be a 3D unit vector")
        return value


class InsertGeometry(EngineeringModel):
    """Shape label and positive dimensional parameters for an insert.

    ``dimensions_in`` intentionally remains shape-agnostic until supported insert
    shapes and their physical treatment are defined.
    """

    shape: str = Field(min_length=1)
    dimensions_in: dict[str, PositiveFiniteFloat] = Field(default_factory=dict)


class InsertSpec(EngineeringModel):
    """Mass, geometry, position, and orientation of added insert material."""

    name: str = Field(min_length=1)
    mass_g: PositiveFiniteFloat
    geometry: InsertGeometry
    location: Vector3
    orientation: Vector3

    @field_validator("orientation")
    @classmethod
    def orientation_must_be_unit_vector(cls, value: Vector3) -> Vector3:
        """Require insert orientation to be normalized."""
        magnitude = math.sqrt(sum(component * component for component in value))
        if not math.isclose(magnitude, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("orientation must be a 3D unit vector")
        return value


class StaticWeights(EngineeringModel):
    """Signed static-weight differences, expressed in ounces."""

    top_bottom_oz: FiniteFloat
    positive_negative_side_oz: FiniteFloat
    finger_thumb_oz: FiniteFloat


class CalculationResult(EngineeringModel):
    """Calculated finished-ball mass properties and diagnostic messages.

    Radius-of-gyration outputs are in inches. The center of mass uses the same
    Cartesian coordinate system and inch units as hole and insert positions.
    """

    finished_mass_g: PositiveFiniteFloat
    low_rg: PositiveFiniteFloat
    intermediate_rg: PositiveFiniteFloat
    high_rg: PositiveFiniteFloat
    total_diff: NonNegativeFiniteFloat
    intermediate_diff: NonNegativeFiniteFloat
    principal_axis_vectors: tuple[Vector3, Vector3, Vector3]
    calculated_center_of_mass: Vector3
    warnings: list[str] = Field(default_factory=list)

    @field_validator("principal_axis_vectors")
    @classmethod
    def principal_axes_must_be_unit_vectors(
        cls, value: tuple[Vector3, Vector3, Vector3]
    ) -> tuple[Vector3, Vector3, Vector3]:
        """Require exactly three normalized principal-axis vectors."""
        for vector in value:
            magnitude = math.sqrt(sum(component * component for component in vector))
            if not math.isclose(magnitude, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("each principal axis must be a 3D unit vector")
        return value

    @field_validator("warnings")
    @classmethod
    def warnings_must_not_be_blank(cls, value: list[str]) -> list[str]:
        """Prevent empty warning messages from entering result records."""
        if any(not warning.strip() for warning in value):
            raise ValueError("warnings cannot contain blank messages")
        return value
