# Data Model

Champollion expects a paired bridge during `fit` and unpaired modality-specific data during `transport`.

The model is modular with respect to the single-cell representations used as input. The main representations can be raw feature matrices, normalized features, PCA, LSI, protein embeddings, or latent factors learned by another model. This lets users choose the representation that best matches the biological question and dataset scale.

## Bridge Cells

The bridge is passed as a `MuData` object. The two selected modalities must contain the same observations. If observation names match but are ordered differently, Champollion reorders the second modality to match the first.

```python
model.fit(
    mdata_bridge,
    x_mod="rna",
    y_mod="atac",
    x_rep="X_pca",
    y_rep="X_lsi",
)
```

## Unpaired Cells

After fitting, transport inputs are passed as a dictionary keyed by the same modality names used during `fit`.

```python
result = model.transport(
    {"rna": adata_rna, "atac": adata_atac},
)
```

Champollion deliberately requires modality names at transport time to avoid accidentally swapping modalities.

## Representations

Representations can be specified with:

- `"X"` for `adata.X`
- `"layers/counts"` for `adata.layers["counts"]`
- `"obsm/X_pca"` for `adata.obsm["X_pca"]`
- a shorthand key when unambiguous

For `X` and `layers`, feature names are read from `adata.var_names`. For `obsm`, feature names are generated as `<rep>_<idx>` unless explicit names are provided with `feature_names`.
