import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/champollion-keops-cache")

anndata = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from champollion import Champollion, TransportResult, process_prior_data
from champollion.prior import compute_prior_cost

FEATURE_NAMES = {
    "rna": ["rna_factor_0", "rna_factor_1", "rna_factor_2"],
    "adt": ["adt_factor_0", "adt_factor_1"],
}


def _adata(matrix, obs_prefix, var_names=None):
    adata = anndata.AnnData(np.asarray(matrix, dtype=np.float32))
    adata.obs_names = [f"{obs_prefix}_{idx}" for idx in range(adata.n_obs)]
    if var_names is not None:
        adata.var_names = var_names
    return adata


def _make_bridge(seed=0, n_cells=5, d_rna=3, d_adt=2, d_prior=4):
    rng = np.random.default_rng(seed)
    rna = _adata(
        rng.normal(size=(n_cells, d_rna)),
        "bridge",
        var_names=[f"gene_{idx}" for idx in range(d_rna)],
    )
    adt = _adata(
        rng.normal(size=(n_cells, d_adt)),
        "bridge",
        var_names=[f"protein_{idx}" for idx in range(d_adt)],
    )
    rna.obsm["latent"] = rng.normal(size=(n_cells, d_rna)).astype(np.float32)
    adt.obsm["latent"] = rng.normal(size=(n_cells, d_adt)).astype(np.float32)
    rna.obsm["prior"] = rng.normal(size=(n_cells, d_prior)).astype(np.float32)
    adt.obsm["prior"] = rng.normal(size=(n_cells, d_prior)).astype(np.float32)
    return SimpleNamespace(mod={"rna": rna, "adt": adt})


def _make_unpaired(seed=1, n_rna=4, n_adt=3, d_rna=3, d_adt=2, d_prior=4):
    rng = np.random.default_rng(seed)
    rna = _adata(
        rng.normal(size=(n_rna, d_rna)),
        "rna",
        var_names=[f"gene_{idx}" for idx in range(d_rna)],
    )
    adt = _adata(
        rng.normal(size=(n_adt, d_adt)),
        "adt",
        var_names=[f"protein_{idx}" for idx in range(d_adt)],
    )
    rna.obsm["latent"] = rng.normal(size=(n_rna, d_rna)).astype(np.float32)
    adt.obsm["latent"] = rng.normal(size=(n_adt, d_adt)).astype(np.float32)
    rna.obsm["prior"] = rng.normal(size=(n_rna, d_prior)).astype(np.float32)
    adt.obsm["prior"] = rng.normal(size=(n_adt, d_prior)).astype(np.float32)
    rna.obs["score"] = np.linspace(0.0, 1.0, n_rna)
    adt.obs["cell_type"] = ["type_a", "type_b", "type_a"][:n_adt]
    adt.obs["score"] = np.linspace(1.0, 2.0, n_adt)
    return {"rna": rna, "adt": adt}


def _fit_model(seed=0, use_keops=False, use_prior=True):
    bridge = _make_bridge(seed=seed)
    model = Champollion(
        epsilon=1.0,
        gamma=0.05,
        lambda_prior=0.5,
        use_keops=use_keops,
        device="cpu",
        random_state=seed,
        max_iter=2,
        learning_rate=1e-2,
        sinkhorn_tol=0.0,
        log_every=1,
        max_iter_sink=3,
        transport_log_every=1,
        verbose=False,
    )
    kwargs = {}
    if use_prior:
        kwargs = {"prior_x_rep": "prior", "prior_y_rep": "prior"}
    model.fit(
        bridge,
        x_mod="rna",
        y_mod="adt",
        x_rep="latent",
        y_rep="latent",
        feature_names=FEATURE_NAMES,
        **kwargs,
    )
    return model


def test_public_dense_fit_transport_and_transfer_workflow():
    model = _fit_model(seed=10, use_keops=False, use_prior=True)

    assert model.is_fitted_
    assert model.modalities_ == ("rna", "adt")
    assert model.A_.shape == (3, 2)
    assert list(model.A_dataframe().index) == [
        "rna_factor_0",
        "rna_factor_1",
        "rna_factor_2",
    ]

    bridge_result = model.training_transport()
    assert isinstance(bridge_result, TransportResult)
    assert bridge_result.cost.shape == (5, 5)
    assert bridge_result.plan.shape == (5, 5)
    assert bridge_result.plan_diagnostics["finite"]
    assert bridge_result.plan_diagnostics["mass_abs_error"] < 1e-4

    adatas = _make_unpaired(seed=11)
    result = model.transport(
        adatas,
        reps={"rna": "latent", "adt": "latent"},
        prior_reps={"rna": "prior", "adt": "prior"},
        feature_names=FEATURE_NAMES,
        max_iter_sink=3,
        log_every=1,
        store_cost=False,
        store_plan=False,
    )

    assert isinstance(result, TransportResult)
    assert result.f.shape == (4,)
    assert result.g.shape == (3,)
    assert result.cost.shape == (4, 3)
    assert result.plan.shape == (4, 3)
    assert torch.isfinite(result.cost).all()
    assert torch.isfinite(result.plan).all()
    assert result.plan_diagnostics["finite"]
    assert result.plan_diagnostics["mass_abs_error"] < 1e-4

    labels = result.transfer_obs("cell_type", source="adt")
    assert list(labels.index) == list(adatas["rna"].obs_names)
    assert set(labels.astype(str)).issubset({"type_a", "type_b"})

    label_result = result.transfer_obs(
        "cell_type",
        source="adt",
        return_probabilities=True,
        inplace=True,
        target_key="predicted_cell_type",
    )
    assert "predicted_cell_type" in adatas["rna"].obs
    assert label_result["probabilities"].shape == (4, 2)

    scores = result.transfer_obs("score", source="adt", kind="continuous")
    assert scores.shape == (4,)
    assert list(scores.index) == list(adatas["rna"].obs_names)

    projected = result.project("latent", source="adt", inplace=True, target_key="X_adt")
    assert projected.shape == (4, 2)
    assert "X_adt" in adatas["rna"].obsm

    matches = result.top_matches(source="adt", k=2)
    assert matches.shape == (4, 4)
    assert list(matches.index) == list(adatas["rna"].obs_names)


def test_schema_validation_and_prior_error_paths():
    model = _fit_model(seed=20, use_keops=False, use_prior=True)
    adatas = _make_unpaired(seed=21)

    with pytest.raises(ValueError, match="Missing modalities"):
        model.transport({"rna": adatas["rna"]})

    with pytest.raises(ValueError, match="Unexpected modalities"):
        model.transport(
            {"rna": adatas["rna"], "adt": adatas["adt"], "protein": adatas["adt"]}
        )

    with pytest.raises(ValueError, match="was fitted with 3 features"):
        model.transport(
            {"rna": adatas["adt"], "adt": adatas["rna"]},
            reps={"rna": "latent", "adt": "latent"},
            prior_reps={"rna": "prior", "adt": "prior"},
            max_iter_sink=1,
            log_every=1,
        )

    bad_rna = adatas["rna"].copy()
    bad_rna.obsm["latent"] = bad_rna.obsm["latent"][:, ::-1].copy()
    with pytest.raises(ValueError, match="Feature names"):
        model.transport(
            {"rna": bad_rna, "adt": adatas["adt"]},
            reps={"rna": "latent", "adt": "latent"},
            prior_reps={"rna": "prior", "adt": "prior"},
            feature_names={
                "rna": ["rna_factor_2", "rna_factor_1", "rna_factor_0"],
                "adt": ["adt_factor_0", "adt_factor_1"],
            },
            max_iter_sink=1,
            log_every=1,
        )

    with pytest.raises(ValueError, match="prior representations cannot be None"):
        model.transport(
            adatas,
            reps={"rna": "latent", "adt": "latent"},
            prior_reps={"rna": None, "adt": None},
            feature_names=FEATURE_NAMES,
            max_iter_sink=1,
            log_every=1,
        )

    no_prior_model = _fit_model(seed=22, use_keops=False, use_prior=False)
    with pytest.raises(ValueError, match="fitted without priors"):
        no_prior_model.transport(
            adatas,
            reps={"rna": "latent", "adt": "latent"},
            prior_reps={"rna": "prior", "adt": "prior"},
            feature_names=FEATURE_NAMES,
            max_iter_sink=1,
            log_every=1,
        )


def test_X_and_layer_representations_use_var_names_and_can_be_reused():
    rng = np.random.default_rng(50)
    rna = _adata(
        rng.normal(size=(4, 3)),
        "bridge",
        var_names=["gene_a", "gene_b", "gene_c"],
    )
    adt = _adata(
        rng.normal(size=(4, 2)),
        "bridge",
        var_names=["protein_a", "protein_b"],
    )
    rna.layers["counts"] = (rna.X + 5).copy()
    adt.layers["counts"] = (adt.X + 5).copy()
    bridge = SimpleNamespace(mod={"rna": rna, "adt": adt})

    x_model = Champollion(
        device="cpu",
        random_state=50,
        max_iter=1,
        log_every=1,
        max_iter_sink=2,
        transport_log_every=1,
        sinkhorn_tol=0.0,
        verbose=False,
    ).fit(bridge, x_mod="rna", y_mod="adt")

    assert x_model.reps_ == {"rna": "X", "adt": "X"}
    assert list(x_model.feature_names_["rna"]) == ["gene_a", "gene_b", "gene_c"]
    assert list(x_model.feature_names_["adt"]) == ["protein_a", "protein_b"]
    assert list(x_model.A_dataframe().index) == ["gene_a", "gene_b", "gene_c"]
    assert list(x_model.A_dataframe().columns) == ["protein_a", "protein_b"]

    rna_test = _adata(
        rng.normal(size=(2, 3)),
        "rna",
        var_names=["gene_a", "gene_b", "gene_c"],
    )
    adt_test = _adata(
        rng.normal(size=(3, 2)),
        "adt",
        var_names=["protein_a", "protein_b"],
    )
    x_result = x_model.transport(
        {"rna": rna_test, "adt": adt_test},
        max_iter_sink=2,
        log_every=1,
    )
    assert x_result.plan.shape == (2, 3)

    layer_model = Champollion(
        device="cpu",
        random_state=51,
        max_iter=1,
        log_every=1,
        max_iter_sink=2,
        transport_log_every=1,
        sinkhorn_tol=0.0,
        verbose=False,
    ).fit(
        bridge,
        x_mod="rna",
        y_mod="adt",
        x_rep="layers/counts",
        y_rep="counts",
    )

    assert layer_model.reps_ == {"rna": "layers/counts", "adt": "counts"}
    assert list(layer_model.feature_names_["rna"]) == ["gene_a", "gene_b", "gene_c"]
    assert list(layer_model.feature_names_["adt"]) == ["protein_a", "protein_b"]

    rna_test.layers["counts"] = (rna_test.X + 5).copy()
    adt_test.layers["counts"] = (adt_test.X + 5).copy()

    reused_reps = layer_model.transport(
        {"rna": rna_test, "adt": adt_test},
        max_iter_sink=2,
        log_every=1,
    )
    explicit_reps = layer_model.transport(
        {"rna": rna_test, "adt": adt_test},
        reps={"rna": "layers/counts", "adt": "counts"},
        max_iter_sink=2,
        log_every=1,
    )
    assert reused_reps.plan.shape == (2, 3)
    assert torch.allclose(reused_reps.cost, explicit_reps.cost)
    assert torch.allclose(reused_reps.plan, explicit_reps.plan)

    rna_bad = rna_test.copy()
    rna_bad.var_names = ["gene_c", "gene_b", "gene_a"]
    with pytest.raises(ValueError, match="Feature names"):
        layer_model.transport(
            {"rna": rna_bad, "adt": adt_test},
            max_iter_sink=1,
            log_every=1,
        )


def test_save_load_and_A_interpretation_utilities(tmp_path):
    model = _fit_model(seed=30, use_keops=False, use_prior=False)
    model.A_ = torch.tensor(
        [
            [3.0, -1.0],
            [0.0, 2.0],
            [-4.0, 0.5],
        ],
        dtype=torch.float32,
    )

    A = model.A_dataframe()
    assert A.loc["rna_factor_0", "adt_factor_0"] == pytest.approx(3.0)

    top = model.top_interactions("rna_factor_0", modality="rna", k=2)
    assert list(top["target_feature"]) == ["adt_factor_0", "adt_factor_1"]
    assert top["weight"].tolist() == pytest.approx([3.0, -1.0])

    positive = model.top_interactions(
        "rna_factor_0",
        modality="rna",
        k=2,
        direction="positive",
    )
    assert list(positive["target_feature"]) == ["adt_factor_0"]

    negative = model.top_interactions(
        "adt_factor_0",
        modality="adt",
        k=2,
        direction="negative",
    )
    assert list(negative["target_feature"]) == ["rna_factor_2"]
    assert negative["weight"].tolist() == pytest.approx([-4.0])

    with pytest.raises(KeyError, match="not found"):
        model.top_interactions("missing", modality="rna")

    path = tmp_path / "champollion.pt"
    model.save(path)
    loaded = Champollion.load(path, device="cpu")
    assert loaded.is_fitted_
    assert loaded._backend is None
    assert loaded._trainer is None
    assert torch.allclose(loaded.A_, model.A_)
    assert loaded.modalities_ == model.modalities_
    assert loaded.feature_names_["rna"].equals(model.feature_names_["rna"])

    adatas = _make_unpaired(seed=31)
    before = model.transport(
        adatas,
        reps={"rna": "latent", "adt": "latent"},
        feature_names=FEATURE_NAMES,
        max_iter_sink=3,
        log_every=1,
    )
    after = loaded.transport(
        adatas,
        reps={"rna": "latent", "adt": "latent"},
        feature_names=FEATURE_NAMES,
        max_iter_sink=3,
        log_every=1,
    )
    assert torch.allclose(after.cost, before.cost)
    assert torch.allclose(after.plan, before.plan)


def test_prior_processing_is_finite_and_warns_for_zero_mean_cost():
    prior = np.array(
        [
            [np.nan, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    original = prior.copy()
    processed = process_prior_data(prior)

    assert np.isnan(original[0, 0])
    assert np.isfinite(processed).all()
    assert processed[0].tolist() == [0.0, 0.0, 0.0]
    assert processed[2].tolist() == [0.0, 0.0, 0.0]

    with pytest.warns(UserWarning, match="Prior cost mean is numerically zero"):
        cost = compute_prior_cost(
            np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float32),
            np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float32),
            device=torch.device("cpu"),
        )
    assert cost.shape == (2, 2)
    assert torch.count_nonzero(cost) == 0


def test_keops_symbolic_transport_and_materialization_guards():
    pytest.importorskip("pykeops")

    model = _fit_model(seed=40, use_keops=True, use_prior=True)
    adatas = _make_unpaired(seed=41, n_rna=2, n_adt=3)
    result = model.transport(
        adatas,
        reps={"rna": "latent", "adt": "latent"},
        prior_reps={"rna": "prior", "adt": "prior"},
        feature_names=FEATURE_NAMES,
        max_iter_sink=2,
        log_every=1,
    )

    assert result.is_symbolic
    assert result.cost_is_symbolic
    assert result.plan_is_symbolic
    assert result.plan_diagnostics["finite"]

    with pytest.raises(RuntimeError, match="Blocked materialize_cost"):
        result.materialize_cost(max_entries=1)

    with pytest.raises(RuntimeError, match="Blocked materialize_plan"):
        result.materialize_plan(max_entries=1)

    dense_cost = result.materialize_cost()
    dense_plan = result.materialize_plan()
    assert dense_cost.shape == (2, 3)
    assert dense_plan.shape == (2, 3)
    assert torch.isfinite(dense_cost).all()
    assert torch.isfinite(dense_plan).all()

    labels = result.transfer_obs("cell_type", source="adt")
    assert labels.shape == (2,)
    projected = result.project("latent", source="adt")
    assert projected.shape == (2, 2)
