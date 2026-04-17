import math

import torch
from pykeops.torch import LazyTensor


def learnt_cost(x_1, x_2, A, use_keops):
    if use_keops:
        Ax_2 = x_2 @ A.T
        x_1_i = LazyTensor(x_1[:, None, :])
        Ax_2_j = LazyTensor(Ax_2[None, :, :])
        return -(x_1_i * Ax_2_j).sum(dim=2)
    return -(x_1 @ A @ x_2.T)


def full_cost(x_1, x_2, A, prior_cost=None, lambda_prior=1.0, use_keops=False):
    cost = learnt_cost(x_1=x_1, x_2=x_2, A=A, use_keops=use_keops)
    if prior_cost is not None:
        return cost + lambda_prior * prior_cost
    return cost


def c_transform_potential(cost, potential, log_n, epsilon, use_keops):
    if use_keops:
        expanded_potential = LazyTensor(potential.view(potential.shape[0], 1), axis=0)
    else:
        expanded_potential = potential[:, None]
    out_potential = -epsilon * (
        ((-cost + expanded_potential) / epsilon) - log_n
    ).logsumexp(dim=0)
    return out_potential.view(-1)


def marginal_from_potentials(cost, f, g, epsilon, use_keops):
    n_1 = f.shape[0]
    n_2 = g.shape[0]
    f_tilde = c_transform_potential(
        cost=cost.T,
        potential=g,
        log_n=math.log(n_2),
        epsilon=epsilon,
        use_keops=use_keops,
    )
    return ((f - f_tilde) / epsilon).exp() / n_1


def transport_plan(cost, f, g, epsilon, n_1=None, n_2=None, use_keops=False):
    if n_1 is None:
        n_1 = f.shape[0]
    if n_2 is None:
        n_2 = g.shape[0]
    if use_keops:
        pot_f = LazyTensor(f.view(-1, 1, 1))
        pot_g = LazyTensor(g.view(1, -1, 1))
    else:
        pot_f = f[:, None]
        pot_g = g[None, :]
    return ((pot_f + pot_g - cost) / epsilon).exp() / (n_1 * n_2)


def transport_plan_diagnostics(
    cost, f, g, epsilon, use_keops=False, n_1=None, n_2=None, tol=1e-3
):
    plan = transport_plan(
        cost=cost,
        f=f,
        g=g,
        epsilon=epsilon,
        n_1=n_1,
        n_2=n_2,
        use_keops=use_keops,
    )
    marginal_1 = plan.sum(dim=1).view(-1)
    marginal_2 = plan.sum(dim=0).view(-1)
    expected_1 = torch.ones_like(marginal_1) / marginal_1.shape[0]
    expected_2 = torch.ones_like(marginal_2) / marginal_2.shape[0]
    marginal_1_abs_error = (marginal_1 - expected_1).abs()
    marginal_2_abs_error = (marginal_2 - expected_2).abs()
    mass = marginal_1.sum()
    mass_error = (mass - 1).abs()
    finite = (
        torch.isfinite(mass)
        & torch.isfinite(marginal_1).all()
        & torch.isfinite(marginal_2).all()
    )
    passed = (
        finite
        & (mass_error <= tol)
        & (marginal_1_abs_error.sum() <= tol)
        & (marginal_2_abs_error.sum() <= tol)
    )
    return {
        "mass": float(mass.detach().cpu()),
        "mass_abs_error": float(mass_error.detach().cpu()),
        "x_1_marginal_l1_error": float(marginal_1_abs_error.sum().detach().cpu()),
        "x_2_marginal_l1_error": float(marginal_2_abs_error.sum().detach().cpu()),
        "finite": bool(finite.detach().cpu()),
        "passed": bool(passed.detach().cpu()),
        "tol": tol,
    }


def sinkhorn_potentials(cost, epsilon, max_iter, tol, log_every, use_keops, device):
    n_1, n_2 = cost.shape
    g = torch.zeros(n_2, device=device)
    conv_flag = False
    for k in range(max_iter):
        f = c_transform_potential(
            cost=cost.T,
            potential=g,
            log_n=math.log(n_2),
            epsilon=epsilon,
            use_keops=use_keops,
        )
        g = c_transform_potential(
            cost=cost,
            potential=f,
            log_n=math.log(n_1),
            epsilon=epsilon,
            use_keops=use_keops,
        )
        if (k + 1) % log_every == 0:
            marginal_1 = marginal_from_potentials(
                cost=cost,
                f=f,
                g=g,
                epsilon=epsilon,
                use_keops=use_keops,
            )
            weights_1 = torch.ones_like(marginal_1) / len(marginal_1)
            err_1 = torch.linalg.norm(weights_1 - marginal_1, ord=1).item()
            conv_flag = err_1 < tol
            if conv_flag:
                break
    return f, g, conv_flag
