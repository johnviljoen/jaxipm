"""jaxipm: GPU-batched nonlinear interior-point optimization in JAX."""

import json
from importlib import resources


def default_params() -> dict:
    """Return a fresh copy of the default (IPOPT-style) solver parameters."""
    with resources.files(__package__).joinpath("params.json").open() as f:
        return json.load(f)
