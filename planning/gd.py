import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device
from models.latWM import sde
from models.latWM import transform, invTrans


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = obs_0.shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)

        return actions

    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = self.preprocessor.transform_obs(obs_0).to(self.device)
        trans_obs_g = self.preprocessor.transform_obs(obs_g).to(self.device)

        actions = self.init_actions(trans_obs_0, actions).to(self.device)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_evals = actions.shape[0]
        estimated_observations = []

        for i in range(self.opt_steps):
            optimizer.zero_grad()
            
            import pdb; pdb.set_trace()
            query_frame = trans_obs_0.squeeze(0)
            with torch.no_grad():
                for i in range(self.horizon):
                    x_init = torch.randn(*trans_obs_0.shape, device=self.device)
                    sample = sde.euler_maruyama(sde.ODE(self.wm), x_init=x_init, sample_steps=50, y=None, condition=actions[:,1:], img_cond=query_frame, target_time_index=torch.zeros([1, 1], device=self.device), alpha=0, adversarial=False)
                    transformed_sample = torch.clip(invTrans(sample.squeeze(0)), 0, 1)
                    estimated_observations.append(transformed_sample.permute(0, 2, 3, 1).detach().cpu().numpy())
                    query_frame = transform(transformed_sample)


            import pdb; pdb.set_trace()            
            loss = self.objective_fn(i_z_obses, trans_obs_g)  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            with torch.no_grad():
                actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                actions_new += (
                    torch.randn_like(actions_new) * self.action_noise
                )  # Add Gaussian noise
                actions.copy_(actions_new)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        return actions, np.full(n_evals, np.inf)  # all actions are valid
