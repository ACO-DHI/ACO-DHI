from typing import Any, Optional, Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.acodhi.policy import ACODHIPolicy


class ACODHI(REINFORCE):


    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[ACODHIPolicy] = None,
        baseline: Union[REINFORCEBaseline, str] = "no",
        policy_kwargs: dict = {},
        baseline_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = ACODHIPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)

    def shared_step(
        self,
        batch: Any,
        batch_idx: int,
        phase: str,
        dataloader_idx: Union[int, None] = None,
    ):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase)

        # Compute loss
        if phase == "train":
            out["loss"] = -(out["advantage"] * out["log_likelihood"]).mean()

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
