from pathlib import Path

import numpy as np
import pandas as pd
import torch

from champollion._data import (
    align_fully_paired_modalities,
    as_float_tensor,
    get_feature_names,
    get_representation,
)
from champollion._iot import LassoIOT
from champollion._optim import AdamOptimizer
from champollion._ot import (
    full_cost,
    sinkhorn_potentials,
    transport_plan_diagnostics,
)
from champollion.prior import compute_prior_cost
from champollion.transport import TransportResult


class Champollion:
    """Integrates unpaired single-cell multimodal data using a small set paired of cells
    called bridge.

    Champollion learns a bilinear cross-modality cost matrix ``A`` from the
    paired bridge cells, then uses the learned cost to transport unpaired cells
    between two modalities. Inputs are expected to be AnnData-like objects,
    usually stored in a MuData object for the bridge and passed as a
    modality-keyed dictionary for transport.

    Parameters
    ----------
    epsilon
        Entropic regularization strength used in the optimal transport problem. The
        default value of 1 should not be changed as it amounts only to a rescaling of
        the problem.
    gamma
        Lasso regularization weight applied to the learned matrix ``A``.
    lambda_prior
        Weight of the optional prior cost term. This is the lambda parameter
        from the paper.
    use_keops
        If ``True``, use KeOps LazyTensors for symbolic cost and plan
        operations to reduce dense memory use.
    device
        Torch device. Use ``"auto"`` to select CUDA when available and CPU
        otherwise.
    random_state
        Optional random seed passed to torch before fitting.
    max_iter
        Number of Adam optimization iterations for fitting ``A``.
    learning_rate
        Adam learning rate.
    sinkhorn_tol
        Tolerance used when checking Sinkhorn marginal convergence.
    log_every
        Number of fit iterations between metric logging and convergence checks.
    max_iter_sink
        Maximum number of Sinkhorn iterations used for transport after fitting.
    transport_log_every
        Number of transport Sinkhorn iterations between convergence checks.
    monitor_gradient_norm
        If truthy, enable gradient-norm stopping during fitting.
    gradient_norm_tol
        Gradient norm threshold used when gradient-norm stopping is enabled.
    wandb_log
        If ``True``, log metrics to Weights & Biases. Requires installing the
        optional ``wandb`` extra.
    verbose
        If ``True``, print fit and transport progress.
    prior_weight
        Deprecated alias for ``lambda_prior``.
    """

    SAVE_FORMAT_VERSION = 1

    def __init__(
        self,
        epsilon=1.0,
        gamma=0.01,
        lambda_prior=20.0,
        use_keops=False,
        device="auto",
        random_state=None,
        max_iter=300_000,
        learning_rate=1e-3,
        sinkhorn_tol=1e-3,
        log_every=40,
        max_iter_sink=1000,
        transport_log_every=10,
        monitor_gradient_norm=None,
        gradient_norm_tol=1e-3,
        wandb_log=False,
        verbose=False,
        prior_weight=None,
    ):
        if prior_weight is not None:
            if lambda_prior != 20.0:
                raise ValueError(
                    "Use only one of 'lambda_prior' or the deprecated "
                    "'prior_weight' alias."
                )
            lambda_prior = prior_weight
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambda_prior = lambda_prior
        self.use_keops = use_keops
        self.device = self._resolve_device(device)
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.sinkhorn_tol = sinkhorn_tol
        self.log_every = log_every
        self.max_iter_sink = max_iter_sink
        self.transport_log_every = transport_log_every
        self.monitor_gradient_norm = monitor_gradient_norm
        self.gradient_norm_tol = gradient_norm_tol
        self.wandb_log = wandb_log
        self.verbose = verbose

        self._backend = None
        self._trainer = None
        self._train_transport = None
        self.training_history_ = None
        self.A_ = None
        self.x_mod_ = None
        self.y_mod_ = None
        self.modalities_ = None
        self.reps_ = None
        self.prior_reps_ = None
        self.dims_ = None
        self.feature_names_ = None
        self.feature_name_sources_ = None
        self.uses_prior_ = None
        self.is_fitted_ = False

    def fit(
        self,
        mdata,
        x_mod,
        y_mod,
        x_rep="X",
        y_rep="X",
        prior_x_rep=None,
        prior_y_rep=None,
        feature_names=None,
    ):
        """Fit the cross-modality cost on the paired bridge cells.

        The bridge's two modalities must contain the same observations. If the
        observation names are the same but ordered differently, the second
        modality is reordered to match the first.

        Representations can be specified as ``"X"``, ``"obsm/key"``,
        ``"layers/key"``, or by shorthand key when unambiguous. If a
        representation comes from ``.X`` or ``.layers``, feature names are taken
        from ``adata.var_names``. If it comes from ``.obsm``, feature names are
        generated unless supplied with ``feature_names``.

        Parameters
        ----------
        mdata
            MuData-like object with a ``.mod`` mapping containing both
            modalities for the paired bridge cells.
        x_mod, y_mod
            Names of the two modalities to align.
        x_rep, y_rep
            Main representations used to learn the bilinear cost matrix.
        prior_x_rep, prior_y_rep
            Optional prior representations. Provide both or neither.
        feature_names
            Optional mapping from modality name to feature names for the main
            representations, mainly useful for ``.obsm`` representations.

        Returns
        -------
        Champollion
            The fitted model.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        explicit_feature_names = self._resolve_feature_name_overrides(feature_names)
        x_adata, y_adata = align_fully_paired_modalities(
            mdata.mod[x_mod],
            mdata.mod[y_mod],
        )
        x = as_float_tensor(get_representation(x_adata, x_rep), self.device)
        y = as_float_tensor(get_representation(y_adata, y_rep), self.device)
        x_feature_names, x_feature_source = get_feature_names(
            x_adata,
            x_rep,
            x.shape[1],
            explicit_names=explicit_feature_names.get(x_mod),
        )
        y_feature_names, y_feature_source = get_feature_names(
            y_adata,
            y_rep,
            y.shape[1],
            explicit_names=explicit_feature_names.get(y_mod),
        )
        prior_cost, prior_x, prior_y = self._extract_prior_data(
            x_adata=x_adata,
            y_adata=y_adata,
            prior_x_rep=prior_x_rep,
            prior_y_rep=prior_y_rep,
        )

        self._backend = LassoIOT(
            d_x=x.shape[1],
            d_y=y.shape[1],
            n_p=x.shape[0],
            epsilon=self.epsilon,
            gamma=self.gamma,
            lamb=self.lambda_prior,
            device=self.device,
            use_keops=self.use_keops,
        )
        self._trainer = AdamOptimizer(
            x=x,
            y=y,
            prior_cost=prior_cost,
            max_iter=self.max_iter,
            log_n_steps=self.log_every,
            sink_tol=self.sinkhorn_tol,
            monitor_gradient_norm=self.monitor_gradient_norm,
            gradient_norm_tol=self.gradient_norm_tol,
            wandb_log=self.wandb_log,
            verbose=self.verbose,
            lr=self.learning_rate,
        )
        self._trainer.fit(self._backend)

        train_cost = self._backend.get_full_cost(x, y, prior_cost=prior_cost)
        f, g = self._backend.get_potentials(train_cost)
        train_plan_diagnostics = self._plan_diagnostics(
            cost=train_cost,
            f=f,
            g=g,
        )
        self._train_transport = TransportResult(
            model=self,
            x=x,
            y=y,
            prior_cost=prior_cost,
            cost=train_cost,
            f=f.detach(),
            g=g.detach(),
            x_obs_names=x_adata.obs_names.copy(),
            y_obs_names=y_adata.obs_names.copy(),
            x_adata=x_adata,
            y_adata=y_adata,
            prior_x=prior_x,
            prior_y=prior_y,
            plan_diagnostics=train_plan_diagnostics,
        )
        self.training_history_ = self._trainer.logs
        self.A_ = self._backend.get_A().detach()
        self.x_mod_ = x_mod
        self.y_mod_ = y_mod
        self.modalities_ = (x_mod, y_mod)
        self.reps_ = {x_mod: x_rep, y_mod: y_rep}
        self.prior_reps_ = {x_mod: prior_x_rep, y_mod: prior_y_rep}
        self.dims_ = {x_mod: x.shape[1], y_mod: y.shape[1]}
        self.feature_names_ = {x_mod: x_feature_names, y_mod: y_feature_names}
        self.feature_name_sources_ = {
            x_mod: x_feature_source,
            y_mod: y_feature_source,
        }
        self.uses_prior_ = prior_x_rep is not None or prior_y_rep is not None
        self.is_fitted_ = True
        return self

    def transport(
        self,
        adatas,
        reps=None,
        prior_reps=None,
        store_cost=True,
        store_plan=False,
        max_iter_sink=None,
        log_every=None,
        feature_names=None,
    ):
        """Compute transport between unpaired modality-specific AnnData objects.

        This step uses the learned matrix ``A`` and the representation schema recorded
        during ``fit``. The input dictionary must contain exactly the two modality names
        used during ``fit``.

        Parameters
        ----------
        adatas
            Dictionary mapping the modality names used during ``fit`` to
            AnnData objects.
        reps
            Optional mapping from modality name to main representation. If
            omitted, the representation names specified during ``fit`` are reused.
        prior_reps
            Optional mapping from modality name to prior representation. If the
            model was fitted with priors and this is omitted, the prior
            representation names specified during ``fit`` are reused.
        store_cost
            Whether to keep the cost object in the returned result. With KeOps,
            this may be symbolic rather than dense.
        store_plan
            Whether to compute and store the transport plan immediately. If
            ``False``, the plan is computed lazily on first access.
        max_iter_sink
            Override the model's transport Sinkhorn iteration limit.
        log_every
            Override the model's transport Sinkhorn check frequency.
        feature_names
            Optional feature-name overrides for transport representations.

        Returns
        -------
        TransportResult
            Object containing potentials, diagnostics, lazy cost/plan access,
            and downstream transfer/projection utilities.
        """
        self._check_is_fitted()
        self._validate_transport_modalities(adatas)
        reps = self._resolve_transport_reps(reps)
        prior_reps = self._resolve_transport_prior_reps(prior_reps)
        explicit_feature_names = self._resolve_feature_name_overrides(feature_names)

        x_adata = adatas[self.x_mod_]
        y_adata = adatas[self.y_mod_]
        x = as_float_tensor(get_representation(x_adata, reps[self.x_mod_]), self.device)
        y = as_float_tensor(get_representation(y_adata, reps[self.y_mod_]), self.device)
        self._validate_transport_dimensions(x=x, y=y, reps=reps)
        self._validate_transport_feature_names(
            adatas=adatas,
            reps=reps,
            dims={self.x_mod_: x.shape[1], self.y_mod_: y.shape[1]},
            explicit_feature_names=explicit_feature_names,
        )
        prior_cost, prior_x, prior_y = self._extract_prior_data(
            x_adata=x_adata,
            y_adata=y_adata,
            prior_x_rep=prior_reps[self.x_mod_],
            prior_y_rep=prior_reps[self.y_mod_],
        )
        cost = full_cost(
            x=x,
            y=y,
            A=self.A_,
            prior_cost=prior_cost,
            lambda_prior=self.lambda_prior,
            use_keops=self.use_keops,
        )
        f, g = self._solve_transport_potentials(
            cost=cost,
            max_iter_sink=self.max_iter_sink
            if max_iter_sink is None
            else max_iter_sink,
            log_every=self.transport_log_every if log_every is None else log_every,
        )
        diagnostics = self._plan_diagnostics(cost=cost, f=f, g=g)
        result = TransportResult(
            model=self,
            x=x,
            y=y,
            prior_cost=prior_cost,
            cost=cost if store_cost else None,
            f=f.detach(),
            g=g.detach(),
            x_obs_names=x_adata.obs_names.copy(),
            y_obs_names=y_adata.obs_names.copy(),
            x_adata=x_adata,
            y_adata=y_adata,
            prior_x=prior_x,
            prior_y=prior_y,
            plan_diagnostics=diagnostics,
        )
        if store_plan:
            _ = result.plan
        return result

    def training_transport(self):
        """Return transport quantities for the paired bridge cells.

        Returns
        -------
        TransportResult
            Result object for the paired cells used in ``fit``.
        """
        self._check_is_fitted()
        return self._train_transport

    def save(self, path):
        """Save the fitted model state needed for future transport.

        The saved file contains hyperparameters, ``A``, modality names,
        representation names, feature names, and schema metadata. It does not
        store bridge data, dense costs, dense plans, optimizers, or cached fit
        internals.

        Parameters
        ----------
        path
            Destination path passed to ``torch.save``.
        """
        self._check_is_fitted()
        path = Path(path)
        payload = {
            "format_version": self.SAVE_FORMAT_VERSION,
            "hyperparameters": {
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "lambda_prior": self.lambda_prior,
                "use_keops": self.use_keops,
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "sinkhorn_tol": self.sinkhorn_tol,
                "log_every": self.log_every,
                "max_iter_sink": self.max_iter_sink,
                "transport_log_every": self.transport_log_every,
                "monitor_gradient_norm": self.monitor_gradient_norm,
                "gradient_norm_tol": self.gradient_norm_tol,
                "wandb_log": self.wandb_log,
                "verbose": self.verbose,
            },
            "state": {
                "A": self.A_.detach().cpu(),
                "x_mod": self.x_mod_,
                "y_mod": self.y_mod_,
                "modalities": tuple(self.modalities_),
                "reps": dict(self.reps_),
                "prior_reps": dict(self.prior_reps_),
                "dims": dict(self.dims_),
                "feature_names": {
                    mod: list(names) for mod, names in self.feature_names_.items()
                },
                "feature_name_sources": dict(self.feature_name_sources_),
                "uses_prior": self.uses_prior_,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device="auto", use_keops=None):
        """Load a fitted Champollion model saved with :meth:`save`.

        Parameters
        ----------
        path
            Path to a saved Champollion model.
        device
            Device on which to load the learned matrix ``A``.
        use_keops
            Optional override for the saved KeOps setting.

        Returns
        -------
        Champollion
            Fitted model ready for ``transport``.
        """
        resolved_device = cls._resolve_device(device)
        payload = torch.load(path, map_location=resolved_device)
        if payload.get("format_version") != cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Champollion save format version "
                f"{payload.get('format_version')!r}."
            )
        hyperparameters = dict(payload["hyperparameters"])
        if "prior_weight" in hyperparameters and "lambda_prior" not in hyperparameters:
            hyperparameters["lambda_prior"] = hyperparameters.pop("prior_weight")
        if use_keops is not None:
            hyperparameters["use_keops"] = use_keops
        model = cls(**hyperparameters, device=device)

        state = payload["state"]
        model.A_ = state["A"].to(model.device)
        model.x_mod_ = state["x_mod"]
        model.y_mod_ = state["y_mod"]
        model.modalities_ = tuple(state["modalities"])
        model.reps_ = dict(state["reps"])
        model.prior_reps_ = dict(state["prior_reps"])
        model.dims_ = dict(state["dims"])
        model.feature_names_ = {
            mod: pd.Index(names) for mod, names in state["feature_names"].items()
        }
        model.feature_name_sources_ = dict(state["feature_name_sources"])
        model.uses_prior_ = state["uses_prior"]
        model._backend = None
        model._trainer = None
        model._train_transport = None
        model.training_history_ = None
        model.is_fitted_ = True
        return model

    def A_dataframe(self):
        """Return the learned matrix ``A`` as a labeled DataFrame.

        Returns
        -------
        pandas.DataFrame
            Matrix with rows named by the first modality's features and columns
            named by the second modality's features.
        """
        self._check_is_fitted()
        return pd.DataFrame(
            self.A_.detach().cpu().numpy(),
            index=self.feature_names_[self.x_mod_],
            columns=self.feature_names_[self.y_mod_],
        )

    def save_A(self, path, format="auto"):
        """Save the learned matrix ``A`` with feature labels.

        Parameters
        ----------
        path
            Output path.
        format
            One of ``"auto"``, ``"csv"``, ``"tsv"``, ``"parquet"``,
            ``"pkl"``, or ``"pickle"``. ``"auto"`` infers the format from the
            path suffix.
        """
        path = Path(path)
        if format == "auto":
            format = path.suffix.removeprefix(".")
        A = self.A_dataframe()
        if format in {"csv", "tsv"}:
            sep = "\t" if format == "tsv" else ","
            A.to_csv(path, sep=sep)
        elif format == "parquet":
            A.to_parquet(path, index=True)
        elif format in {"pkl", "pickle"}:
            A.to_pickle(path)
        else:
            raise ValueError(
                "format must be 'auto', 'csv', 'tsv', 'parquet', or 'pickle'."
            )

    def top_interactions(self, feature, modality, k=10, direction="both"):
        """Return top weighted interactions for one feature in ``A``.

        Parameters
        ----------
        feature
            Feature name in the queried modality.
        modality
            Modality containing ``feature``.
        k
            Maximum number of interactions to return.
        direction
            ``"positive"``, ``"negative"``, or ``"both"``. ``"both"`` ranks by
            absolute weight.

        Returns
        -------
        pandas.DataFrame
            Interaction table with source/target modalities, feature names,
            signed weights, and absolute weights.
        """
        self._check_is_fitted()
        if modality not in self.modalities_:
            raise ValueError(f"modality must be one of {self.modalities_}.")
        if direction not in {"positive", "negative", "both"}:
            raise ValueError("direction must be 'positive', 'negative', or 'both'.")
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k < 1:
            raise ValueError("k must be >= 1.")

        feature_names = self.feature_names_[modality]
        matches = np.flatnonzero(feature_names == feature)
        if len(matches) == 0:
            raise KeyError(
                f"Feature {feature!r} was not found in modality {modality!r}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Feature {feature!r} appears multiple times in modality {modality!r}; "
                "feature names must be unique for top_interactions."
            )
        idx = int(matches[0])
        A = self.A_.detach().cpu().numpy()
        if modality == self.x_mod_:
            weights = A[idx, :]
            other_mod = self.y_mod_
            other_features = self.feature_names_[other_mod]
        else:
            weights = A[:, idx]
            other_mod = self.x_mod_
            other_features = self.feature_names_[other_mod]

        if direction == "positive":
            keep = weights > 0
            order_values = weights
        elif direction == "negative":
            keep = weights < 0
            order_values = -weights
        else:
            keep = weights != 0
            order_values = np.abs(weights)

        candidate_idx = np.flatnonzero(keep)
        if len(candidate_idx) == 0:
            return pd.DataFrame(
                columns=[
                    "source_modality",
                    "source_feature",
                    "target_modality",
                    "target_feature",
                    "weight",
                    "abs_weight",
                ]
            )
        order = np.argsort(order_values[candidate_idx])[::-1][:k]
        selected_idx = candidate_idx[order]
        return pd.DataFrame(
            {
                "source_modality": modality,
                "source_feature": feature,
                "target_modality": other_mod,
                "target_feature": other_features[selected_idx],
                "weight": weights[selected_idx],
                "abs_weight": np.abs(weights[selected_idx]),
            }
        ).reset_index(drop=True)

    def _resolve_feature_name_overrides(self, feature_names):
        if feature_names is None:
            return {}
        expected = set(self.modalities_ or feature_names)
        observed = set(feature_names)
        unexpected = observed - expected
        if unexpected:
            raise ValueError(
                f"feature_names contains unexpected modalities: {sorted(unexpected)}."
            )
        return feature_names

    def _validate_transport_modalities(self, adatas):
        expected = set(self.modalities_)
        observed = set(adatas)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            msg = [f"Transport adatas must contain exactly {sorted(expected)}."]
            if missing:
                msg.append(f"Missing modalities: {missing}.")
            if extra:
                msg.append(f"Unexpected modalities: {extra}.")
            raise ValueError(" ".join(msg))

    def _resolve_transport_reps(self, reps):
        if reps is None:
            return dict(self.reps_)
        expected = set(self.modalities_)
        observed = set(reps)
        if observed != expected:
            raise ValueError(
                "reps must provide one representation for each modality used "
                "during fit: "
                f"{sorted(expected)}."
            )
        return dict(reps)

    def _resolve_transport_prior_reps(self, prior_reps):
        expected = set(self.modalities_)
        if self.uses_prior_:
            if prior_reps is None:
                return dict(self.prior_reps_)
            if set(prior_reps) != expected:
                raise ValueError(
                    "prior_reps must provide one prior representation for "
                    "each modality used during fit: "
                    f"{sorted(expected)}."
                )
            if any(prior_reps[mod] is None for mod in self.modalities_):
                raise ValueError(
                    "This model was fitted with priors; prior representations "
                    "cannot be None at transport time."
                )
            return dict(prior_reps)
        if prior_reps is not None and any(
            prior_reps.get(mod) is not None for mod in prior_reps
        ):
            raise ValueError(
                "This model was fitted without priors; prior_reps cannot be used "
                "at transport time."
            )
        return {self.x_mod_: None, self.y_mod_: None}

    def _validate_transport_dimensions(self, x, y, reps):
        observed = {self.x_mod_: x.shape[1], self.y_mod_: y.shape[1]}
        for mod, dim in observed.items():
            if dim != self.dims_[mod]:
                raise ValueError(
                    f"Representation {reps[mod]!r} for modality {mod!r} has "
                    f"{dim} features, but Champollion was fitted with "
                    f"{self.dims_[mod]} features for that modality."
                )

    def _validate_transport_feature_names(
        self, adatas, reps, dims, explicit_feature_names
    ):
        for mod in self.modalities_:
            names, _ = get_feature_names(
                adatas[mod],
                reps[mod],
                dims[mod],
                explicit_names=explicit_feature_names.get(mod),
            )
            if not names.equals(self.feature_names_[mod]):
                raise ValueError(
                    f"Feature names for modality {mod!r} do not match the "
                    "feature names recorded during fit. Expected "
                    f"{list(self.feature_names_[mod])}, got {list(names)}."
                )

    def _extract_prior_data(self, x_adata, y_adata, prior_x_rep, prior_y_rep):
        if prior_x_rep is None and prior_y_rep is None:
            return None, None, None
        if prior_x_rep is None or prior_y_rep is None:
            raise ValueError("Both prior_x_rep and prior_y_rep must be provided.")
        prior_x = get_representation(x_adata, prior_x_rep)
        prior_y = get_representation(y_adata, prior_y_rep)
        if prior_x.ndim != 2 or prior_y.ndim != 2:
            raise ValueError(
                "Prior representations must be two-dimensional arrays. "
                f"Got shapes {prior_x.shape} and {prior_y.shape}."
            )
        if prior_x.shape[1] != prior_y.shape[1]:
            raise ValueError(
                "Prior representations must have the same number of columns. "
                f"Got {prior_x.shape[1]} for {prior_x_rep!r} and "
                f"{prior_y.shape[1]} for {prior_y_rep!r}."
            )
        prior_cost = compute_prior_cost(
            prior_x, prior_y, device=self.device, use_keops=self.use_keops
        )
        return prior_cost, prior_x, prior_y

    def _solve_transport_potentials(self, cost, max_iter_sink, log_every):
        f, g, conv_flag = sinkhorn_potentials(
            cost=cost,
            epsilon=self.epsilon,
            max_iter=max_iter_sink,
            tol=self.sinkhorn_tol,
            log_every=log_every,
            use_keops=self.use_keops,
            device=self.device,
        )
        if not conv_flag and self.verbose:
            print("Transport sinkhorn loop reached max_iter_sink")
        return f, g

    def _plan_diagnostics(self, cost, f, g):
        return transport_plan_diagnostics(
            cost=cost,
            f=f,
            g=g,
            epsilon=self.epsilon,
            use_keops=self.use_keops,
            n_x=f.shape[0],
            n_y=g.shape[0],
            tol=self.sinkhorn_tol,
        )

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Champollion must be fitted before calling this method.")

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
