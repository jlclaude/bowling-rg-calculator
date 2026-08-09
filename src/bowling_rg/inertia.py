"""Inertia calculation interface reserved for verified physical models."""

from __future__ import annotations

from .models import BallSpec


def build_undrilled_inertia_tensor(ball: BallSpec) -> None:
    """Build an undrilled inertia tensor once the model is formally specified."""
    raise NotImplementedError("undrilled inertia physics has not been specified")
