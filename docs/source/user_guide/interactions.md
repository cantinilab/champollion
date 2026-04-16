# Interpreting the Learned Matrix

Champollion learns a sparse bilinear interaction matrix $\mathbf{A}$ between the two modality representation spaces.

The matrix $\mathbf{A}$ captures dependencies between features across modalities. A positive entry indicates that the corresponding features tend to be expressed in the same cells, while lower or negative values indicate weaker or opposing associations. These associations can arise for many reasons, such as shared regulation, co-expression driven by a common cause, or direct regulatory interactions. The model does not distinguish between these mechanisms, but instead provides a map of cross-modal associations that can be further investigated.

```python
A = model.A_dataframe()
```

The returned object is a `pandas.DataFrame` with row and column names corresponding to the feature names recorded during `fit`.

Because Champollion can use any single-cell representation as input, the interpretation of $\mathbf{A}$ depends on the representation. If raw or biologically named features are used, $\mathbf{A}$ can be read directly as feature associations across modalities, which can help discover markers or prioritize feature pairs for downstream enrichment analysis. If low-dimensional representations are used, $\mathbf{A}$ links dimensions or factors; this can still be biologically meaningful when those factors are interpretable, for example with models such as [DRVI](https://drvi.readthedocs.io/latest/).

Use `top_interactions` to inspect the strongest interactions for a feature:

```python
model.top_interactions(
    feature="CD18",
    modality="protein",
    k=10,
    sign="both",
)
```

`sign="positive"` restricts to positive weights, `sign="negative"` restricts to negative weights, and `sign="both"` sorts by absolute value.
