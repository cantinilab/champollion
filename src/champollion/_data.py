import warnings

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse


def get_representation(adata, rep):
    """Extract a dense numpy representation from ``.X``, ``.layers``, or ``.obsm``."""
    location, key = representation_location(adata, rep)
    if location == "X":
        array = adata.X
    elif location == "layers":
        array = adata.layers[key]
    elif location == "obsm":
        array = adata.obsm[key]
    else:
        raise ValueError(f"Unknown representation location {location!r}.")
    if hasattr(array, "values"):
        array = array.values
    if issparse(array):
        array = array.todense()
    return np.asarray(array)


def representation_location(adata, rep):
    """Resolve a representation string to an AnnData storage location and key."""
    if rep is None or rep == "X":
        return "X", "X"
    if rep.startswith("layers/"):
        return "layers", rep.removeprefix("layers/")
    if rep.startswith("obsm/"):
        return "obsm", rep.removeprefix("obsm/")
    if rep in adata.obsm:
        if rep in adata.layers:
            warnings.warn(
                f"Representation {rep!r} exists in both .obsm and .layers; "
                ".obsm is used by default. Use 'obsm/{rep}' or 'layers/{rep}' "
                "to disambiguate.",
                UserWarning,
                stacklevel=2,
            )
        return "obsm", rep
    if rep in adata.layers:
        return "layers", rep
    raise KeyError(f"Representation {rep!r} was not found in .X, .obsm, or .layers.")


def get_feature_names(adata, rep, n_features, explicit_names=None):
    """Return feature names for a representation and validate their length."""
    if explicit_names is not None:
        names = pd.Index(explicit_names)
        source = "explicit"
    else:
        location, key = representation_location(adata, rep)
        if location in {"X", "layers"}:
            names = pd.Index(adata.var_names)
            source = "var_names"
        else:
            rep_name = key
            names = pd.Index([f"{rep_name}_{idx}" for idx in range(n_features)])
            source = "generated"
    if len(names) != n_features:
        raise ValueError(
            f"Feature names for representation {rep!r} have length {len(names)}, "
            f"but the representation has {n_features} columns."
        )
    if names.has_duplicates:
        duplicates = names[names.duplicated()].unique().tolist()
        raise ValueError(
            f"Feature names for representation {rep!r} must be unique; "
            f"found duplicates: {duplicates}."
        )
    return names, source


def as_float_tensor(array, device):
    """Convert an array-like object to a float32 torch tensor on ``device``."""
    return torch.tensor(array, dtype=torch.float32, device=device)


def align_fully_paired_modalities(x_adata, y_adata):
    """Validate and align the paired bridge cells by observation name."""
    if x_adata.n_obs != y_adata.n_obs:
        raise ValueError(
            "Bridge data must have the same number of observations in both modalities."
        )
    if list(x_adata.obs_names) == list(y_adata.obs_names):
        return x_adata, y_adata
    if set(x_adata.obs_names) != set(y_adata.obs_names):
        raise ValueError(
            "Bridge data must contain the same observation names in both modalities."
        )
    return x_adata, y_adata[x_adata.obs_names].copy()
