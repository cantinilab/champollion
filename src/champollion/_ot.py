import math

import torch
from pykeops.torch import LazyTensor


def learnt_cost(x, y, A, use_keops):
    if use_keops:
        ay = y @ A.T
        x_i = LazyTensor(x[:, None, :])
        ay_j = LazyTensor(ay[None, :, :])
        return (x_i * ay_j).sum(dim=2)
    return x @ A @ y.T


def full_cost(x, y, A, prior_cost=None, lambda_prior=1.0, use_keops=False):
    cost = learnt_cost(x=x, y=y, A=A, use_keops=use_keops)
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
    n_x = f.shape[0]
    n_y = g.shape[0]
    f_tilde = c_transform_potential(
        cost=cost.T,
        potential=g,
        log_n=math.log(n_y),
        epsilon=epsilon,
        use_keops=use_keops,
    )
    return ((f - f_tilde) / epsilon).exp() / n_x


def transport_plan(cost, f, g, epsilon, n_x=None, n_y=None, use_keops=False):
    if n_x is None:
        n_x = f.shape[0]
    if n_y is None:
        n_y = g.shape[0]
    if use_keops:
        pot_f = LazyTensor(f.view(-1, 1, 1))
        pot_g = LazyTensor(g.view(1, -1, 1))
    else:
        pot_f = f[:, None]
        pot_g = g[None, :]
    return ((pot_f + pot_g - cost) / epsilon).exp() / (n_x * n_y)


def transport_plan_diagnostics(
    cost, f, g, epsilon, use_keops=False, n_x=None, n_y=None, tol=1e-3
):
    plan = transport_plan(
        cost=cost,
        f=f,
        g=g,
        epsilon=epsilon,
        n_x=n_x,
        n_y=n_y,
        use_keops=use_keops,
    )
    marginal_x = plan.sum(dim=1).view(-1)
    marginal_y = plan.sum(dim=0).view(-1)
    expected_x = torch.ones_like(marginal_x) / marginal_x.shape[0]
    expected_y = torch.ones_like(marginal_y) / marginal_y.shape[0]
    x_abs_error = (marginal_x - expected_x).abs()
    y_abs_error = (marginal_y - expected_y).abs()
    mass = marginal_x.sum()
    mass_error = (mass - 1).abs()
    finite = (
        torch.isfinite(mass)
        & torch.isfinite(marginal_x).all()
        & torch.isfinite(marginal_y).all()
    )
    passed = (
        finite
        & (mass_error <= tol)
        & (x_abs_error.sum() <= tol)
        & (y_abs_error.sum() <= tol)
    )
    return {
        "mass": float(mass.detach().cpu()),
        "mass_abs_error": float(mass_error.detach().cpu()),
        "x_marginal_l1_error": float(x_abs_error.sum().detach().cpu()),
        "y_marginal_l1_error": float(y_abs_error.sum().detach().cpu()),
        "finite": bool(finite.detach().cpu()),
        "passed": bool(passed.detach().cpu()),
        "tol": tol,
    }


def sinkhorn_potentials(cost, epsilon, max_iter, tol, log_every, use_keops, device):
    n_x, n_y = cost.shape
    g = torch.zeros(n_y, device=device)
    conv_flag = False
    for k in range(max_iter):
        f = c_transform_potential(
            cost=cost.T,
            potential=g,
            log_n=math.log(n_y),
            epsilon=epsilon,
            use_keops=use_keops,
        )
        g = c_transform_potential(
            cost=cost,
            potential=f,
            log_n=math.log(n_x),
            epsilon=epsilon,
            use_keops=use_keops,
        )
        if (k + 1) % log_every == 0:
            marginal_x = marginal_from_potentials(
                cost=cost,
                f=f,
                g=g,
                epsilon=epsilon,
                use_keops=use_keops,
            )
            weights_x = torch.ones_like(marginal_x) / len(marginal_x)
            err_x = torch.linalg.norm(weights_x - marginal_x, ord=1).item()
            conv_flag = err_x < tol
            if conv_flag:
                break
    return f, g, conv_flag
