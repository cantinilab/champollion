# Champollion

Champollion learns an interpretable cross-modality cost from paired bridge cells and uses it to integrate unpaired single-cell profiles across modalities. It is designed for scverse workflows with `AnnData` and `MuData`, and exposes utilities for annotation transfer, barycentric projection, and interpretation of the learned interaction matrix.

```{image} _static/champollion_model.png
:alt: Champollion model overview
:align: center
```

## Install

Install the latest stable release from PyPI (recommended):

```bash
pip install champollion-omics
```

Optional Weights & Biases logging can be installed with:

```bash
pip install "champollion-omics[wandb]"
```

To install the latest development version from GitHub instead:

```bash
pip install "git+https://github.com/cantinilab/champollion.git"
```

Champollion is designed to take advantage of a GPU and runs much faster with one, typically taking only a few minutes on datasets with thousands of cells.

## Getting Started

For a concrete application, see the tutorials; the example below is only a compact overview of the API. Champollion is fitted on a paired bridge stored in a `MuData` object. It can then be applied to modality-specific `AnnData` objects containing unpaired cells.

```python
from champollion import Champollion

model = Champollion(
    epsilon=1.0,
    gamma=0.001,
    lambda_prior=20.0,
    use_keops=False,
    device="auto",
    random_state=0,
)

model.fit(
    mdata_bridge,
    modality_1="rna",
    modality_2="atac",
    x_1_rep="X_pca",
    x_2_rep="X_lsi",
    y_prior_1_rep="X_prior",
    y_prior_2_rep="X_prior",
)

result = model.transport(
    {"rna": adata_rna, "atac": adata_atac},
    x_reps={"rna": "X_pca", "atac": "X_lsi"},
    y_prior_reps={"rna": "X_prior", "atac": "X_prior"},
)

predicted_atac_labels = result.transfer_obs(
    key="cell_type",
    source="rna",
    kind="categorical",
)
```

The learned matrix can be inspected directly:

```python
A = model.A_dataframe()
top_links = model.top_interactions("CD18", modality="protein", k=10)
```

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
user_guide/data
user_guide/transport
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
