"""High-level calculation interface reserved for a verified physical model."""

from __future__ import annotations

from collections.abc import Sequence

from .models import BallSpec, CalculationResult, HoleSpec, InsertSpec


def calculate_finished_ball(
    ball: BallSpec,
    holes: Sequence[HoleSpec],
    inserts: Sequence[InsertSpec] = (),
) -> CalculationResult:
    """Calculate finished properties once the governing physics is specified."""
    raise NotImplementedError("post-drilling calculation physics has not been specified")
