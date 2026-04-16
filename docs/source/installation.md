# Installation

Champollion requires Python 3.12 and PyTorch. GPU support is optional but recommended for larger datasets.

## PyPI

After the official release:

```bash
pip install champollion
```

Optional Weights & Biases logging:

```bash
pip install "champollion[wandb]"
```

## Development Install

```bash
git clone git@github.com:cantinilab/champollion.git
cd champollion
pip install .
```

To run the test suite from a source checkout:

```bash
pip install -e ".[docs]"
pytest
```

## GPU Notes

Champollion uses PyTorch tensors internally. If you plan to run on a GPU, install a PyTorch build compatible with your CUDA runtime and hardware before installing Champollion, or verify the version selected by `pip`.
