# Champollion

[![PyPI version](https://img.shields.io/pypi/v/champollion-omics)](https://pypi.org/project/champollion-omics/)
[![Tests](https://github.com/cantinilab/champollion/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/cantinilab/champollion/actions/workflows/tests.yml)
[![Lint](https://github.com/cantinilab/champollion/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/cantinilab/champollion/actions/workflows/lint.yml)
[![Docs](https://github.com/cantinilab/champollion/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/cantinilab/champollion/actions/workflows/docs.yml)
[![Build](https://github.com/cantinilab/champollion/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/cantinilab/champollion/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/cantinilab/champollion/branch/main/graph/badge.svg)](https://codecov.io/gh/cantinilab/champollion)

Champollion learns an interpretable cross-modality cost from paired bridge cells and uses it to integrate unpaired single-cell profiles across modalities. It is designed for scverse workflows with `AnnData` and `MuData`, and exposes utilities for annotation transfer, barycentric projection, and interpretation of the learned interaction matrix.

![Champollion model overview](docs/source/_static/champollion_model.png)

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

## Documentation

Documentation is available at [champollion-omics.readthedocs.io](https://champollion-omics.readthedocs.io/en/latest/).

Two tutorials are available there to demonstrate classic use cases of the package: fitting Champollion on bridge cells, transporting unpaired cells, transferring annotations, building a joint visualization, interpreting the learnt `A` matrix.

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

## Citation

The preprint is in preparation. Citation information will be added before the official release.
if you're looking for the repository to reproduce the results in the article, please see the [champollion_reproducibility](https://github.com/cantinilab/champollion_reproducibility) repository!
