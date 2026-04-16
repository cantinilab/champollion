import math

import torch
from pykeops.torch import LazyTensor

from champollion._ot import (
    c_transform_potential,
    full_cost,
    marginal_from_potentials,
    transport_plan,
)


class LassoIOT:
    def __init__(self, d_x, d_y, n_p, epsilon, gamma, lamb, device, use_keops):
        self.use_keops = use_keops

        self.n_p = n_p
        self.d_x = d_x
        self.d_y = d_y
        self.device = device

        self.epsilon = epsilon
        self.gamma = gamma
        self.lamb = lamb

        f = torch.randn(self.n_p, device=self.device) / 10
        u = torch.randn(self.d_x, self.d_y, device=self.device) / 10
        v = torch.randn(self.d_x, self.d_y, device=self.device) / 10
        self.params = {
            "f": f.requires_grad_(True),
            "u": u.requires_grad_(True),
            "v": v.requires_grad_(True),
        }

    def get_uv(self):
        return self.params["u"], self.params["v"]

    def get_A(self):
        u, v = self.get_uv()
        return u * v

    def regularization_loss(self):
        u, v = self.get_uv()
        return (u.norm() ** 2 + v.norm() ** 2) * self.gamma / 2

    def get_trace(self, x, y):
        A = self.get_A()
        if self.use_keops:
            ay = y @ A.T
            x_i = LazyTensor(x[:, None, :])
            ay_i = LazyTensor(ay[:, None, :])
            trace = (x_i * ay_i).sum(dim=2).sum(dim=0) / self.n_p
        else:
            trace = (torch.einsum("ij,jk,ik->i", x, A, y) / self.n_p).sum()
        return trace

    def get_learnt_cost(self, x, y):
        return full_cost(
            x=x,
            y=y,
            A=self.get_A(),
            prior_cost=None,
            use_keops=self.use_keops,
        )

    def get_optim_params(self):
        return list(self.params.values())

    def get_full_cost(self, x, y, prior_cost):
        return full_cost(
            x=x,
            y=y,
            A=self.get_A(),
            prior_cost=prior_cost,
            lambda_prior=self.lamb,
            use_keops=self.use_keops,
        )

    def get_plan(self, cost, f=None, g=None, n_x=None, n_y=None):
        if f is None or g is None:
            f, g = self.get_potentials(cost=cost)
        if n_x is None or n_y is None:
            n_x = self.n_p
            n_y = self.n_p
        return transport_plan(
            cost=cost,
            f=f,
            g=g,
            epsilon=self.epsilon,
            n_x=n_x,
            n_y=n_y,
            use_keops=self.use_keops,
        )

    def get_potentials(self, cost):
        f = self.params["f"]
        g = c_transform_potential(
            cost=cost,
            potential=f,
            log_n=math.log(self.n_p),
            epsilon=self.epsilon,
            use_keops=self.use_keops,
        )
        return f, g

    def get_marginal(self, f, g, cost):
        return marginal_from_potentials(
            cost=cost,
            f=f,
            g=g,
            epsilon=self.epsilon,
            use_keops=self.use_keops,
        )

    def iot_loss(self, x, y, prior_cost, return_marginal=False):
        trace = self.get_trace(x=x, y=y)
        cost = self.get_full_cost(x=x, y=y, prior_cost=prior_cost)
        f, g = self.get_potentials(cost=cost)

        n_x = x.shape[0]
        n_y = y.shape[0]

        outputs = {"marginal_x": None}
        outputs["trace_term"] = trace.detach().item()
        outputs["obj_loss"] = trace - ((f / n_x).sum() + (g / n_y).sum())
        if return_marginal:
            with torch.no_grad():
                outputs["marginal_x"] = self.get_marginal(f=f, g=g, cost=cost)
        return outputs

    def eval(self):
        for param in self.get_optim_params():
            param.requires_grad_(False)


def c_transf_potential(cost, potential, log_n, eps, use_keops):
    return c_transform_potential(
        cost=cost,
        potential=potential,
        log_n=log_n,
        epsilon=eps,
        use_keops=use_keops,
    )
