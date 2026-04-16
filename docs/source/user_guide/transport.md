# Integration Outputs

`Champollion.transport` returns a `TransportResult`.

```python
result = model.transport(
    {"rna": adata_rna, "atac": adata_atac},
)
```

The result is the main integration object returned by Champollion. It stores the Sinkhorn potentials and computes the cost and plan lazily, so computational biologists can inspect the alignment, transfer biological metadata across assays, project cells into a shared representation space, and identify the strongest cross-modality matches without having to manually manipulate optimal transport objects.

```python
f, g = result.potentials
cost = result.cost
plan = result.plan
```

## Annotation Transfer

Categorical annotations are transferred by transporting one-hot class probabilities and taking the most likely class. This makes it possible to transfer cell type labels, perturbation labels, sample annotations, or other `.obs` fields from a well-annotated modality to a less annotated one.

```python
predicted = result.transfer_obs(
    key="cell_type",
    source="rna",
    kind="categorical",
)
```

Continuous annotations are transferred by barycentric averaging. This can be useful for pseudotime, module scores, quality-control covariates, or any scalar annotation stored in `.obs`:

```python
score = result.transfer_obs(
    key="pseudotime",
    source="rna",
    kind="continuous",
)
```

## Barycentric Projection

Representations can be projected across modalities:

```python
atac_in_rna_pca = result.project(
    source="atac",
    target_rep="X_pca",
)
```

This is useful for building joint visualizations with representations from both modalities. For example, ATAC cells can be projected into RNA PCA space and visualized together with RNA cells using UMAP.

## Matching

Use `top_matches` or `assignment_confidence` to inspect strongest cell-to-cell correspondences. These utilities help diagnose whether integration is sharp, ambiguous, or concentrated around expected biological populations.
