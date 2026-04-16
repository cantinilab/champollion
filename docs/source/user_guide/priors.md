# Prior Representations

Champollion can add a prior cost term based on prior representations for the two modalities.

Prior information provides a common ground for comparing cells across assays before the cross-modality cost is learned. In practice, these priors often come from sparse known connections between features across modalities: ATAC peaks can be mapped to nearby genes through gene activities, transcripts can be paired with their encoded proteins, or other feature-level links can be defined from biological knowledge.

```python
model.fit(
    mdata_bridge,
    x_mod="rna",
    y_mod="atac",
    x_rep="X_pca",
    y_rep="X_lsi",
    prior_x_rep="X_prior",
    prior_y_rep="X_prior",
)
```

If priors are used during `fit`, matching prior representations are expected during `transport` unless the same representation names should be reused:

```python
result = model.transport(
    {"rna": adata_rna, "atac": adata_atac},
    prior_reps={"rna": "X_prior", "atac": "X_prior"},
)
```

Prior representations are centered and scaled cell-wise before computing the correlation-based prior cost. Cells with zero norm are left unnormalized to avoid introducing missing values.

The prior contribution is controlled by `lambda_prior`.
