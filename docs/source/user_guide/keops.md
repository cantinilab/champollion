# KeOps

For large datasets, setting `use_keops=True` represents pairwise costs and transport plans symbolically with KeOps LazyTensors.

```python
model = Champollion(use_keops=True)
```

In KeOps mode, the Sinkhorn potentials remain ordinary torch tensors, but `result.cost` and `result.plan` may be symbolic objects.

```python
result.cost_is_symbolic
result.plan_is_symbolic
```

High-level operations such as annotation transfer, barycentric projection, top matches, and assignment confidence are designed to work with symbolic plans when possible.

Dense materialization is explicit and guarded by a size limit:

```python
dense_plan = result.materialize_plan(max_entries=50_000 * 50_000)
```

Increase `max_entries` only when you intentionally want to allocate the full dense matrix.
