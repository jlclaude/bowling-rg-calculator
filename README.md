# Bowling RG Calculator

An engineering application for estimating post-drilling radius-of-gyration and
differential properties of bowling balls. The current engine composes mass,
first moments, and inertia tensors for finite cylindrical holes and inserts.

## Requirements

- Python 3.12
- NumPy
- Pydantic 2
- Streamlit
- pytest

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The package uses a `src` layout. For local development without packaging it,
add it to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src"
pytest
streamlit run app.py
```

## Coordinate and unit conventions

- Mass is stored in grams unless a field explicitly ends in `_lb` or `_oz`.
- Length is stored in inches unless a field explicitly states otherwise.
- Positions are Cartesian 3-vectors; direction and orientation vectors must be
  normalized.
- Static-weight differences are signed values.
- `InsertGeometry` uses `shape="cylinder"` with `diameter` and `depth` dimensions
  for calculations currently supported by the engine.

All internal mass-property calculations use kilograms, meters, and kg·m².
`HoleSpec.surface_entry_position` is the outer entry point and its axis points
inward; insert locations identify cylinder centroids. PAP and dual-angle layout
mapping are intentionally not implemented yet, so all positions and directions
must already be expressed in the ball's principal-axis coordinate frame.
