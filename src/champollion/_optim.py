import torch
from tqdm import tqdm


class AdamOptimizer:
    def __init__(
        self,
        x_1,
        x_2,
        prior_cost=None,
        max_iter=2000,
        log_n_steps=10,
        sink_tol=1e-3,
        monitor_gradient_norm=None,
        gradient_norm_tol=1e-3,
        wandb_log=False,
        verbose=False,
        **optim_kwargs,
    ):
        self.x_1 = x_1
        self.x_2 = x_2
        self.prior_cost = prior_cost
        self.device = x_1.device

        self.max_iter = max_iter
        self.sink_tol = sink_tol
        self.optim_kwargs = optim_kwargs
        self.log_n_steps = log_n_steps
        self.wandb_log = wandb_log
        self.verbose = verbose
        self.logs = dict()
        self.monitor_gradient_norm = monitor_gradient_norm
        self.gradient_norm_tol = gradient_norm_tol
        self._wandb = None
        if self.wandb_log:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "wandb_log=True requires the optional dependency 'wandb'. "
                    "Install wandb to enable Weights & Biases logging."
                ) from exc
            self._wandb = wandb

    def log(self, name, value):
        key = f"train_{name}"
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(value)
        if self._wandb is not None:
            self._wandb.log({key: value})
        if self.verbose:
            print(f"{key}: {value}")

    def summary_update(self, name, value):
        key = f"train_{name}"
        if self._wandb is not None:
            self._wandb.summary.update({key: value})
        if self.verbose:
            print(f"{key}: {value}")

    def log_plan_marginal_mass(self, marginal_1):
        weights_1 = torch.ones_like(marginal_1) / len(marginal_1)
        with torch.no_grad():
            err_1 = torch.linalg.norm(weights_1 - marginal_1, ord=1).item()
        self.log(name="err_1", value=err_1)
        plan_mass = marginal_1.sum().item()
        self.log(name="plan_mass", value=plan_mass)
        return plan_mass, err_1

    def fit(self, iot_model):
        optimizer = torch.optim.Adam(iot_model.get_optim_params(), **self.optim_kwargs)
        conv_flag = False
        for i in tqdm(range(self.max_iter), disable=not self.verbose):
            optimizer.zero_grad()
            reg_loss = iot_model.regularization_loss()
            log_flag = i % self.log_n_steps == 0
            loss_dict = iot_model.iot_loss(
                x_1=self.x_1,
                x_2=self.x_2,
                prior_cost=self.prior_cost,
                return_marginal=log_flag,
            )
            loss = loss_dict["obj_loss"] + reg_loss
            loss.backward()
            optimizer.step()
            if log_flag:
                self.log(name="total_loss", value=loss.item())
                self.log(name="obj_loss", value=loss_dict["obj_loss"].item())
                self.log(name="reg_loss", value=reg_loss.item())
                self.log(name="trace_term", value=loss_dict["trace_term"])
                marginal_1 = loss_dict["marginal_1"]
                _, err_1 = self.log_plan_marginal_mass(marginal_1=marginal_1)

                conv_flag = err_1 < self.sink_tol
                if self.monitor_gradient_norm:
                    gradient_norm = torch.linalg.norm(
                        torch.cat(
                            [p.grad.flatten() for p in iot_model.get_optim_params()]
                        ),
                        ord=2,
                    ).item()
                    self.log(name="gradient_norm", value=gradient_norm)
                    if self.gradient_norm_tol is not None:
                        grad_threshold = max(
                            self.gradient_norm_tol,
                            self.gradient_norm_tol * iot_model.gamma,
                        )
                        grad_norm_conv = gradient_norm < grad_threshold
                        conv_flag = conv_flag and grad_norm_conv

                if conv_flag:
                    if self.verbose:
                        print(f"Converged after {i + 1} iterations")
                    self.summary_update(name="n_iterations", value=i + 1)
                    break
        if not conv_flag and self.verbose:
            print("Training reached max_iter")

        cost = iot_model.get_full_cost(
            x_1=self.x_1,
            x_2=self.x_2,
            prior_cost=self.prior_cost,
        )
        plan = iot_model.get_plan(cost=cost)
        plan_mass, _ = self.log_plan_marginal_mass(marginal_1=plan.sum(dim=1))
        A_norm = (iot_model.get_A().norm(p=1)).item()
        self.summary_update(name="A_final_norm", value=A_norm)

        plan_entropy = -(plan * (plan + 1e-12).log()).sum(dim=1).sum()
        max_entropy = -plan_mass * torch.log(
            torch.tensor(
                [plan_mass / (plan.shape[0] * plan.shape[1])], device=self.device
            )
        )
        plan_entropy_ratio = plan_entropy / max_entropy
        self.summary_update(name="plan_entropy_ratio", value=plan_entropy_ratio)
        iot_model.eval()
