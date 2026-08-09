"""Drilling calculation interface reserved for verified physical models."""

from __future__ import annotations

from .models import HoleSpec


def calculate_hole_mass_properties(hole: HoleSpec) -> None:
    """Calculate removed mass properties once drilling physics is specified."""
    raise NotImplementedError("drilling physics has not been specified")
