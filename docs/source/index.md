# Champollion

Champollion learns an interpretable cross-modality cost from paired bridge cells and uses it to integrate unpaired single-cell profiles across modalities.

```{image} _static/champollion_model.png
:alt: Champollion model overview
:align: center
```

Champollion is designed for scverse workflows. The model is fitted on a paired bridge stored in a `MuData` object, then applied to modality-specific `AnnData` objects containing unpaired cells. It returns integration outputs that are useful at several levels: cell-to-cell transport plans for transferring annotations, barycentric projections for joint visualization, pairwise costs for quality control, and the learned interaction matrix `A` for biological interpretation.

## Install

```bash
pip install champollion
```

For the development version:

```bash
git clone git@github.com:cantinilab/champollion.git
cd champollion
pip install .
```

## Minimal Example

```python
from champollion import Champollion

model = Champollion(
    epsilon=1.0,
    gamma=0.001,
    lambda_prior=20.0,
    device="auto",
)

model.fit(
    mdata_bridge,
    x_mod="rna",
    y_mod="atac",
    x_rep="X_pca",
    y_rep="X_lsi",
    prior_x_rep="X_prior",
    prior_y_rep="X_prior",
)

result = model.transport(
    {"rna": adata_rna, "atac": adata_atac},
    reps={"rna": "X_pca", "atac": "X_lsi"},
    prior_reps={"rna": "X_prior", "atac": "X_prior"},
)
```

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
user_guide/data
user_guide/transport
user_guide/priors
user_guide/keops
user_guide/interactions
user_guide/persistence
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
