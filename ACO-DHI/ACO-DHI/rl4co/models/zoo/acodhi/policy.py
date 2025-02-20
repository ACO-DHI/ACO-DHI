from typing import Callable
import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Type, Union

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.acodhi.antsystem import AntSystem
from rl4co.utils.utils import merge_with_defaults
from rl4co.utils.ops import batchify, unbatchify
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.acodhi.decoder import ACODHIDecoder
from rl4co.models.zoo.acodhi.encoder import GraphHAttentionEncoder
from rl4co.models.zoo.acodhi.refiner import ACODHIRefiner


class ACODHIPolicy(NonAutoregressivePolicy):


    def __init__(
        self,
        encoder: nn.Module = None,
        refiner:  nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_station: int = 9,
        num_encoder_layers: int = 3,
        num_heads: int = 1,
        node_dim: int = 2,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        env_name: str = "tsp",
        init_embedding: nn.Module = None,
        sdpa_fn: Callable = None,
        sdpa_fn_encoder: Callable = None,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,

        aco_class: Optional[Type[AntSystem]] = None,
        aco_kwargs: dict = {},
        train_with_local_search: bool = False,
        n_ants: Optional[Union[int, dict]] = None,
        n_iterations: Optional[Union[int, dict]] = None,
        ls_reward_aug_W: float = 0.95,
        **unused_kwargs,
    ):


        if encoder is None:
            encoder = GraphHAttentionEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_station=num_station,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_encoder is None else sdpa_fn_encoder,
 
            )

        if decoder is None:
            decoder = ACODHIDecoder(embed_dim=embed_dim, num_heads=num_heads)
        

        
        

        super(ACODHIPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
            **unused_kwargs,
        )
        
        # self.firstencoder = FirstEncoder(embed_dim=embed_dim)

        self.aco_class = AntSystem if aco_class is None else aco_class
        self.aco_kwargs = aco_kwargs
        self.train_with_local_search = train_with_local_search
        self.n_ants = merge_with_defaults(n_ants, train=30, val=48, test=48)
        self.n_iterations = merge_with_defaults(n_iterations, train=1, val=5, test=10)
        self.ls_reward_aug_W = ls_reward_aug_W
        self.refiner = ACODHIRefiner(embed_dim=embed_dim, num_heads=num_heads, node_dim=node_dim)

        

    
    def forward(
        self,
        td_initial: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        calc_reward: bool = True,
        phase: str = "train",
        actions=None,
        return_actions: bool = True,
        return_hidden: bool = True,
        **kwargs,
    ):
        """
        Forward method. During validation and testing, the policy runs the ACO algorithm to construct solutions.
        See :class:`NonAutoregressivePolicy` for more details during the training phase.
        """
        n_ants = self.n_ants[phase]
        distance_matrix = AntSystem.calculate_distance(self,td_initial)
        # emb = self.firstencoder(td_initial)
        # Instantiate environment if needed
        if (phase != "train" or self.ls_reward_aug_W > 0) and (env is None or isinstance(env, str)):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        if phase == "train":
            select_start_nodes_fn = partial(
                self.aco_class.select_start_node_fn, start_node=self.aco_kwargs.get("start_node", None)
            )
            kwargs.update({"select_start_nodes_fn": select_start_nodes_fn})
            #  we just use the constructive policy
            outdict = super().forward(
                distance_matrix,
                td_initial,
                env,
                
                # emb=emb,
                self.refiner,
                phase=phase,
                decode_type="multistart_sampling",
                calc_reward=calc_reward,
                num_starts=n_ants,
                actions=actions,
                return_actions=return_actions,
                return_hidden=return_hidden,
                **kwargs,
            )

            # manually compute the advantage
            reward = unbatchify(outdict["reward"], n_ants)
            advantage = reward - reward.mean(dim=1, keepdim=True)

            if self.ls_reward_aug_W > 0 and self.train_with_local_search:
                heatmap_logits = outdict["hidden"]
                aco = self.aco_class(
                    heatmap_logits,
                    n_ants=n_ants,
                    temperature=self.aco_kwargs.get("temperature", self.temperature),
                    **self.aco_kwargs,
                )
                
                actions = outdict["actions"]
                _, ls_reward = aco.local_search(batchify(td_initial, n_ants), env, actions)

                ls_reward = unbatchify(ls_reward, n_ants)
                ls_advantage = ls_reward - ls_reward.mean(dim=1, keepdim=True)
                advantage = advantage * (1 - self.ls_reward_aug_W) + ls_advantage * self.ls_reward_aug_W

            outdict["advantage"] = advantage
            outdict["log_likelihood"] = unbatchify(outdict["log_likelihood"], n_ants)

            return outdict

        # heatmap_logits, _ = self.encoder(td_initial)
        

        h, init_h = self.encoder(td_initial)
        final_h = self.refiner(td_initial,h)

        heatmap_logits=self.decoder(td_initial,final_h)

        heatmap_logits = heatmap_logits+distance_matrix





        aco = self.aco_class(
            heatmap_logits,
            self.encoder,
            self.refiner,
            self.decoder,
            h,
           
            n_ants=self.n_ants[phase],
            temperature=self.aco_kwargs.get("temperature", self.temperature),
            **self.aco_kwargs,
        )
        td, actions, reward = aco.run(td_initial, env, self.n_iterations[phase])

        out = {}
        if calc_reward:
            out["reward"] = reward
        if return_actions:
            out["actions"] = actions

        return out

