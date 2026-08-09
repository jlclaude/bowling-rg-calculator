# Bowling RG Calculator

An engineering application for estimating post-drilling radius-of-gyration and
differential properties of bowling balls. This version provides validated data
models plus generic, verified rigid-body tensor mathematics. It intentionally
does **not** convert bowling specifications into tensors or implement drilling,
layout, PAP, pin, PSA, or manufacturer-specific conventions.

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
- `InsertGeometry` stores a shape label and named dimensions without assuming a
  mass-property formula.

The coordinate-frame definition, drilling-volume treatment, initial inertia
reconstruction, and insert composition rules must be documented and verified
before the placeholder calculation functions are implemented.

## Generic inertia utilities

`bowling_rg.inertia` operates exclusively in kilograms, meters, and kg·m². It
provides unit conversions, RG/moment conversion, principal and rotated tensors,
parallel-axis translation, arbitrarily oriented finite-cylinder centroid
tensors, and symmetric eigendecomposition. The bowling-specific
`build_undrilled_inertia_tensor` function remains deliberately unimplemented.
