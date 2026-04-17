# Data Inputs

Champollion expects a paired bridge during `fit` and unpaired modality-specific data during `transport`.

The model is modular with respect to the single-cell representations used as input. Champollion does not impose a preprocessing pipeline: the main representations can be raw feature matrices, normalized features, PCA, LSI, or any cell low-dimensional embeddings learned by another model. This lets users choose the representation that best matches the biological question and dataset scale.

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

## Preprocessing

Champollion accepts any representation that can be stored in an `AnnData` object, so preprocessing can be adapted to the dataset, modality, and downstream interpretation goals. In the experiments reported in the paper, we used the following choices.

For RNA/ATAC integration, both modalities were log-normalized, scaled with `scanpy.pp.scale`, and embedded with PCA (due to the extremely high dimensionality of ATAC data). In a separate case study, we used DRVI embeddings for both modalities instead of PCA-based representations.

For RNA/ADT integration with a CITE-seq bridge, ADT counts were normalized with Muon's implementation of centered log-ratio (CLR) normalization. RNA was log-normalized and scaled, then represented either with PCA or with 4,000 highly variable genes when direct feature-level interpretability was preferred.

## Prior Representations

Champollion can add a prior cost term based on prior representations for the two modalities. 
Prior information provides a common ground for directly comparing cells across assays, complementing the learned cross-modal cost. By incorporating this external knowledge, it helps guide the matching and improves robustness, particularly when the bridge data alone is insufficient to ensure a reliable integration.
In practice, these priors often come from sparse known connections between features across modalities: ATAC peaks can be mapped to nearby genes through gene activities, transcripts can be paired with their encoded proteins, or other feature-level links can be defined from biological knowledge.

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

For RNA/ATAC integration, we first used Signac to compute gene activities from ATAC profiles. To build the prior representation, we subsetted both RNA profiles and ATAC gene activities to the common genes (only in the prior representations, not for the main representations), log-normalized them, concatenated the two cell-by-gene matrices, and ran PCA on the concatenated representation.

For RNA/ADT integration, we manually mapped each surface protein to its coding gene using GeneCards, then restricted both modalities to the resulting gene-protein pairs. The prior cost was the correlation distance, `1 - r` where `r` is Pearson's correlation, between every cell in modality 1 and every cell in modality 2, computed with `scipy.spatial.distance.cdist` (Virtanen et al., 2020).

Prior representations are centered and scaled cell-wise before computing the correlation-based prior cost. Cells with zero norm are left unnormalized to avoid introducing missing values. The prior contribution is controlled by `lambda_prior`.
