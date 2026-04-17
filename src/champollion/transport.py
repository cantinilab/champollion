import numpy as np
import pandas as pd
import torch
from pykeops.torch import LazyTensor

from champollion._data import get_representation
from champollion._ot import full_cost, transport_plan
from champollion.prior import compute_prior_cost

DEFAULT_MAX_MATERIALIZE_ENTRIES = 50_000 * 50_000


class TransportResult:
    """Result of transporting cells between the two modalities.

    A result stores the Sinkhorn potentials and lazily exposes the pairwise
    cost and transport plan. In dense mode these are torch tensors. In KeOps
    mode they may be symbolic LazyTensor objects, so use the high-level methods
    such as :meth:`transfer_obs`, :meth:`project`, and :meth:`top_matches` when
    possible.
    """

    def __init__(
        self,
        model,
        x,
        y,
        f,
        g,
        prior_cost=None,
        cost=None,
        plan=None,
        x_obs_names=None,
        y_obs_names=None,
        x_adata=None,
        y_adata=None,
        prior_x=None,
        prior_y=None,
        plan_diagnostics=None,
    ):
        self._model = model
        self._x = x
        self._y = y
        self._prior_cost = prior_cost
        self._cost = cost
        self._plan = plan
        self.f = f
        self.g = g
        self.x_obs_names = x_obs_names
        self.y_obs_names = y_obs_names
        self.x_adata = x_adata
        self.y_adata = y_adata
        self._prior_x = prior_x
        self._prior_y = prior_y
        self._materialized_cost = None
        self._materialized_plan = None
        self.plan_diagnostics = plan_diagnostics

    @property
    def potentials(self):
        """Tuple of Sinkhorn potentials ``(f, g)``."""
        return self.f, self.g

    @property
    def is_symbolic(self):
        """Whether the transport plan is represented symbolically."""
        return not isinstance(self.plan, torch.Tensor)

    @property
    def cost_is_symbolic(self):
        """Whether the cost object is represented symbolically."""
        return not isinstance(self.cost, torch.Tensor)

    @property
    def plan_is_symbolic(self):
        """Whether the plan object is represented symbolically."""
        return not isinstance(self.plan, torch.Tensor)

    @property
    def cost(self):
        """Pairwise transport cost, computed lazily when needed."""
        if self._cost is None:
            self._cost = full_cost(
                x=self._x,
                y=self._y,
                A=self._model.A_,
                prior_cost=self._prior_cost,
                lambda_prior=self._model.lambda_prior,
                use_keops=self._model.use_keops,
            )
        return self._cost

    @property
    def plan(self):
        """Transport plan, computed lazily when needed."""
        if self._plan is None:
            self._plan = transport_plan(
                cost=self.cost,
                f=self.f,
                g=self.g,
                epsilon=self._model.epsilon,
                n_x=self._x.shape[0],
                n_y=self._y.shape[0],
                use_keops=self._model.use_keops,
            )
        return self._plan

    def clear_cost(self):
        """Drop cached cost objects so they can be recomputed later."""
        self._cost = None
        self._materialized_cost = None

    def clear_plan(self):
        """Drop cached plan objects so they can be recomputed later."""
        self._plan = None
        self._materialized_plan = None

    def materialize_cost(self, max_entries=DEFAULT_MAX_MATERIALIZE_ENTRIES):
        """Return a dense tensor for the pairwise cost.

        This explicitly allocates an ``n_x`` by ``n_y`` tensor, even when the
        result was computed with KeOps. Large materializations are blocked by
        default.

        Parameters
        ----------
        max_entries
            Maximum allowed number of dense matrix entries. Increase this
            value only when the dense allocation is intentional.

        Returns
        -------
        torch.Tensor
            Dense pairwise cost matrix.
        """
        self._check_materialization_limit(max_entries=max_entries, what="cost")
        if self._materialized_cost is None:
            prior_cost = None
            if self._prior_x is not None and self._prior_y is not None:
                prior_cost = compute_prior_cost(
                    self._prior_x,
                    self._prior_y,
                    device=self._x.device,
                    use_keops=False,
                )
            self._materialized_cost = full_cost(
                x=self._x,
                y=self._y,
                A=self._model.A_,
                prior_cost=prior_cost,
                lambda_prior=self._model.lambda_prior,
                use_keops=False,
            )
        return self._materialized_cost

    def materialize_plan(self, max_entries=DEFAULT_MAX_MATERIALIZE_ENTRIES):
        """Return a dense tensor for the transport plan.

        Parameters
        ----------
        max_entries
            Maximum allowed number of dense matrix entries. Increase this
            value only when the dense allocation is intentional.

        Returns
        -------
        torch.Tensor
            Dense transport plan with total mass close to one.
        """
        self._check_materialization_limit(max_entries=max_entries, what="plan")
        if self._materialized_plan is None:
            cost = self.materialize_cost(max_entries=max_entries)
            self._materialized_plan = transport_plan(
                cost=cost,
                f=self.f,
                g=self.g,
                epsilon=self._model.epsilon,
                n_x=self._x.shape[0],
                n_y=self._y.shape[0],
                use_keops=False,
            )
        return self._materialized_plan

    def transfer_obs(
        self,
        key,
        source,
        kind="auto",
        target_key=None,
        inplace=False,
        return_probabilities=False,
    ):
        """Transfer an ``.obs`` annotation from one modality to the other.

        Categorical annotations are transferred by transporting one-hot class
        probabilities and taking the argmax. Continuous annotations are
        transferred by barycentric averaging.

        Parameters
        ----------
        key
            Observation column to transfer from the source AnnData.
        source
            Source modality name. The target is the other modality.
        kind
            ``"auto"``, ``"categorical"``, or ``"continuous"``.
        target_key
            Column name used when ``inplace=True``. Defaults to
            ``f"{key}_champollion"``.
        inplace
            If ``True``, write the transferred annotation to the target AnnData.
        return_probabilities
            For categorical transfer, return both predictions and class
            probabilities.

        Returns
        -------
        pandas.Series or dict
            Transferred values, or a dictionary with ``"prediction"`` and
            ``"probabilities"`` for categorical transfers when requested.
        """
        source_adata, target_adata = self._get_source_target_adatas(source)
        values = source_adata.obs[key]
        if kind == "auto":
            kind = (
                "continuous" if pd.api.types.is_numeric_dtype(values) else "categorical"
            )
        if kind == "categorical":
            result = self._transfer_categorical(values=values, source=source)
            if inplace:
                if target_key is None:
                    target_key = f"{key}_champollion"
                target_adata.obs[target_key] = result["prediction"]
            if return_probabilities:
                return result
            return result["prediction"]
        if kind == "continuous":
            result = self.apply(values.to_numpy()[:, None], source=source)
            series = pd.Series(
                result[:, 0],
                index=target_adata.obs_names,
                name=target_key or f"{key}_champollion",
            )
            if inplace:
                target_adata.obs[series.name] = series
            return series
        raise ValueError("kind must be 'auto', 'categorical', or 'continuous'.")

    def project(self, rep="X", source=None, target_key=None, inplace=False):
        """Barycentrically project a source representation onto target cells.

        Parameters
        ----------
        rep
            Representation in the source AnnData to project.
        source
            Source modality name. The target is the other modality.
        target_key
            Target ``.obsm`` key used when ``inplace=True``.
        inplace
            If ``True``, write the projected matrix to target ``.obsm``.

        Returns
        -------
        numpy.ndarray
            Projected representation with one row per target cell.
        """
        if source is None:
            raise ValueError(f"source must be one of {self._model.modalities_}.")
        source_adata, target_adata = self._get_source_target_adatas(source)
        values = get_representation(source_adata, rep)
        projected = self.apply(values, source=source)
        if inplace:
            if target_key is None:
                target_key = f"X_champollion_{source}"
            target_adata.obsm[target_key] = projected
        return projected

    def apply(self, values, source):
        """Apply the normalized transport plan to arbitrary source values.

        Parameters
        ----------
        values
            Array-like matrix with one row per source cell.
        source
            Source modality name.

        Returns
        -------
        numpy.ndarray
            Transport-weighted values with one row per target cell.
        """
        source = self._resolve_source(source)
        values = torch.as_tensor(
            np.asarray(values).copy(), dtype=torch.float32, device=self._x.device
        )
        if values.ndim == 1:
            values = values[:, None]
        plan = self.plan
        if isinstance(plan, torch.Tensor):
            weights = self.normalized_plan(source=source)
            return (weights @ values).detach().cpu().numpy()
        return self._apply_symbolic(values=values, source=source).detach().cpu().numpy()

    def normalized_plan(self, source):
        """Return a row-normalized dense plan for source-to-target transfer.

        This method requires a dense plan. In KeOps mode, use ``apply`` or the
        higher-level transfer/projection methods instead of calling this
        directly.
        """
        plan = self._dense_plan()
        source = self._resolve_source(source)
        if source == self._model.y_mod_:
            weights = plan
        elif source == self._model.x_mod_:
            weights = plan.T
        else:
            raise ValueError(f"source must be one of {self._model.modalities_}.")
        row_sums = weights.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=torch.finfo(weights.dtype).tiny)
        return weights / row_sums

    def top_matches(self, source, k=5):
        """Return top transported source cells for each target cell.

        Parameters
        ----------
        source
            Source modality name.
        k
            Number of matches per target cell.

        Returns
        -------
        pandas.DataFrame
            Target-indexed table with match observation names and normalized
            weights.
        """
        source_adata, target_adata = self._get_source_target_adatas(source)
        source = self._resolve_source(source)
        plan = self.plan
        if isinstance(plan, torch.Tensor):
            weights = self.normalized_plan(source=source)
            k = min(k, weights.shape[1])
            values, indices = torch.topk(weights, k=k, dim=1)
        else:
            k = min(k, len(source_adata))
            values, indices = self._symbolic_topk(source=source, k=k)
        source_names = np.asarray(source_adata.obs_names)
        columns = {}
        for rank in range(k):
            columns[f"match_{rank + 1}"] = source_names[indices[:, rank].cpu().numpy()]
            columns[f"weight_{rank + 1}"] = values[:, rank].detach().cpu().numpy()
        return pd.DataFrame(columns, index=target_adata.obs_names)

    def _transfer_categorical(self, values, source):
        _, target_adata = self._get_source_target_adatas(source)
        categorical = pd.Categorical(values)
        one_hot = np.eye(len(categorical.categories), dtype=np.float32)[
            categorical.codes
        ]
        probabilities = self.apply(one_hot, source=source)
        predictions = categorical.categories[np.argmax(probabilities, axis=1)]
        prediction = pd.Series(
            pd.Categorical(predictions, categories=categorical.categories),
            index=target_adata.obs_names,
            name=f"{values.name}_champollion",
        )
        probabilities = pd.DataFrame(
            probabilities,
            index=target_adata.obs_names,
            columns=categorical.categories,
        )
        return {"prediction": prediction, "probabilities": probabilities}

    def _get_source_target_adatas(self, source):
        if self.x_adata is None or self.y_adata is None:
            raise RuntimeError(
                "This TransportResult does not hold AnnData references, so "
                "AnnData-based transfer methods are unavailable."
            )
        source = self._resolve_source(source)
        if source == self._model.x_mod_:
            return self.x_adata, self.y_adata
        if source == self._model.y_mod_:
            return self.y_adata, self.x_adata
        raise ValueError(f"source must be one of {self._model.modalities_}.")

    def _resolve_source(self, source):
        if source in self._model.modalities_:
            return source
        raise ValueError(f"source must be one of {self._model.modalities_}.")

    def _dense_plan(self):
        plan = self.plan
        if not isinstance(plan, torch.Tensor):
            raise RuntimeError(
                "The transport plan is symbolic in KeOps mode. Use "
                "materialize_plan() to explicitly build a dense tensor, or use "
                "transfer_obs(), project(), and top_matches(), which support "
                "symbolic reductions."
            )
        return plan

    def _check_materialization_limit(self, max_entries, what):
        n_entries = self._x.shape[0] * self._y.shape[0]
        if max_entries is not None and n_entries > max_entries:
            raise RuntimeError(
                f"Blocked materialize_{what}() because it would allocate a dense "
                f"{self._x.shape[0]} x {self._y.shape[0]} tensor "
                f"({n_entries} entries), which is larger than max_entries="
                f"{max_entries}. Increase max_entries if you still want to "
                "materialize this result."
            )

    def _apply_symbolic(self, values, source):
        plan = self.plan
        if source == self._model.y_mod_:
            values_j = LazyTensor(values[None, :, :])
            numerator = (plan * values_j).sum(dim=1)
            denominator = plan.sum(dim=1)
        elif source == self._model.x_mod_:
            values_i = LazyTensor(values[:, None, :])
            numerator = (plan * values_i).sum(dim=0)
            denominator = plan.sum(dim=0)
        else:
            raise ValueError(f"source must be one of {self._model.modalities_}.")
        return numerator / torch.clamp(
            denominator, min=torch.finfo(numerator.dtype).tiny
        )

    def _symbolic_topk(self, source, k):
        plan = self.plan
        if source == self._model.y_mod_:
            dim = 1
            denominator = plan.sum(dim=1)
        elif source == self._model.x_mod_:
            dim = 0
            denominator = plan.sum(dim=0)
        else:
            raise ValueError(f"source must be one of {self._model.modalities_}.")
        indices = (-plan).argKmin(K=k, dim=dim)
        values = -(-plan).Kmin(K=k, dim=dim)
        values = values / torch.clamp(denominator, min=torch.finfo(values.dtype).tiny)
        return values, indices
