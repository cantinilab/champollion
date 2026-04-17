# Quickstart

Champollion has two steps:

1. Fit the cost on paired bridge cells.
2. Transport unpaired cells between the same two modalities.

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
```

The transport result stores Sinkhorn potentials and exposes the cost and plan lazily:

```python
cost = result.cost
plan = result.plan
diagnostics = result.plan_diagnostics
```

Annotations can be transferred from one modality to the other:

```python
predicted_labels = result.transfer_obs(
    key="cell_type",
    source="rna",
    kind="categorical",
)
```

Continuous representations can be projected barycentrically:

```python
atac_in_rna_space = result.project(
    source="atac",
    target_rep="X_pca",
)
```
