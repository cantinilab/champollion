# Interpreting the Learned Matrix

Champollion learns a sparse bilinear interaction matrix `A` between the two modality representation spaces.

This matrix is one of the main interpretability outputs of the method. It tells which dimensions of one modality representation are encouraged or discouraged to match dimensions of the other modality when constructing the cross-modality cost.

```python
A = model.A_dataframe()
```

The returned object is a `pandas.DataFrame` with row and column names corresponding to the feature names recorded during `fit`.

Because Champollion can use any single-cell representation as input, the interpretation of `A` depends on the representation. If raw or biologically named features are used, `A` can be read directly as feature associations across modalities, which can help discover markers or prioritize feature pairs for downstream enrichment analysis. If low-dimensional representations are used, `A` links dimensions or factors; this can still be biologically meaningful when those factors are interpretable, for example with models such as [DRVI](https://drvi.readthedocs.io/latest/).

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
