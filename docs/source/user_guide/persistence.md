# Saving and Loading

Save a fitted model with:

```python
model.save("champollion_model.pt")
```

Load it later with:

```python
from champollion import Champollion

model = Champollion.load("champollion_model.pt", device="auto")
```

The saved file contains hyperparameters, the learned matrix `A`, modality names, representation names, feature names, and schema metadata.

It does not store bridge data, dense costs, dense plans, optimizers, or cached fit internals. Transport on new unpaired data is independent from bridge costs and bridge potentials except through the learned matrix `A` and recorded schema.
