import math

import torch
from typing import Any, Callable, Optional, Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.improvement.base import ImprovementDecoder
from rl4co.models.nn.attention import MultiHeadCompat
from rl4co.models.nn.mlp import MLP
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ACODHIDecoder(ImprovementDecoder):

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.hidden_dim = embed_dim

        # for MHC sublayer (NFE aspect)
        self.compater_node = MultiHeadCompat(
            num_heads, embed_dim, embed_dim, embed_dim, embed_dim
        )


        self.norm_factor = 1 / math.sqrt(1 * self.hidden_dim)
        # for Max-Pooling sublayer
        self.project_graph_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # for feed-forward aggregation (FFA)sublayer
        self.value_head = MLP(
            input_dim=2 * self.n_heads,
            output_dim=1,
            num_neurons=[32, 32],
            dropout_probs=[0.05, 0.00],
        )

    def forward(self, td: TensorDict, final_h: Tensor) -> Tensor:

        batch_size, graph_size, dim = final_h.size()
        # Max-Pooling sublayer
        h_node_refined = self.project_node_node(final_h) + self.project_graph_node(
            final_h.max(1)[0]
        )[:, None, :].expand(batch_size, graph_size, dim)

        # MHC sublayer
        compatibility = torch.zeros(
            (batch_size, graph_size, graph_size, self.n_heads * 2),
            device=h_node_refined.device,
        )

        # compatibility[:, :, :, self.n_heads :] = self.compater_node(
        #     h_node_refined
        # ).permute(1, 2, 3, 0)
        compatibility = self.compater_node(
            h_node_refined
        )
        # FFA sublater
        # return self.value_head(self.norm_factor * compatibility).squeeze(-1)
        return self.norm_factor * compatibility[0]
    
    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, Any, RL4COEnvBase]:

        return td, env, hidden



