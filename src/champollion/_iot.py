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
    def __init__(self, d_1, d_2, n_p, epsilon, gamma, lamb, device, use_keops):
        self.use_keops = use_keops

        self.n_p = n_p
        self.d_1 = d_1
        self.d_2 = d_2
        self.device = device

        self.epsilon = epsilon
        self.gamma = gamma
        self.lamb = lamb

        f = torch.randn(self.n_p, device=self.device) / 10
        u = torch.randn(self.d_1, self.d_2, device=self.device) / 10
        v = torch.randn(self.d_1, self.d_2, device=self.device) / 10
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

    def get_trace(self, x_1, x_2):
        A = self.get_A()
        if self.use_keops:
            Ax_2 = x_2 @ A.T
            x_1_i = LazyTensor(x_1[:, None, :])
            Ax_2_i = LazyTensor(Ax_2[:, None, :])
            trace = -(x_1_i * Ax_2_i).sum(dim=2).sum(dim=0) / self.n_p
        else:
            trace = -(torch.einsum("ij,jk,ik->i", x_1, A, x_2) / self.n_p).sum()
        return trace

    def get_learnt_cost(self, x_1, x_2):
        return full_cost(
            x_1=x_1,
            x_2=x_2,
            A=self.get_A(),
            prior_cost=None,
            use_keops=self.use_keops,
        )

    def get_optim_params(self):
        return list(self.params.values())

    def get_full_cost(self, x_1, x_2, prior_cost):
        return full_cost(
            x_1=x_1,
            x_2=x_2,
            A=self.get_A(),
            prior_cost=prior_cost,
            lambda_prior=self.lamb,
            use_keops=self.use_keops,
        )

    def get_plan(self, cost, f=None, g=None, n_1=None, n_2=None):
        if f is None or g is None:
            f, g = self.get_potentials(cost=cost)
        if n_1 is None or n_2 is None:
            n_1 = self.n_p
            n_2 = self.n_p
        return transport_plan(
            cost=cost,
            f=f,
            g=g,
            epsilon=self.epsilon,
            n_1=n_1,
            n_2=n_2,
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

    def iot_loss(self, x_1, x_2, prior_cost, return_marginal=False):
        trace = self.get_trace(x_1=x_1, x_2=x_2)
        cost = self.get_full_cost(x_1=x_1, x_2=x_2, prior_cost=prior_cost)
        f, g = self.get_potentials(cost=cost)

        n_1 = x_1.shape[0]
        n_2 = x_2.shape[0]

        outputs = {"marginal_1": None}
        outputs["trace_term"] = trace.detach().item()
        outputs["obj_loss"] = trace - ((f / n_1).sum() + (g / n_2).sum())
        if return_marginal:
            with torch.no_grad():
                outputs["marginal_1"] = self.get_marginal(f=f, g=g, cost=cost)
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
