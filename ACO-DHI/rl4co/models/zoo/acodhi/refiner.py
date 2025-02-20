

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor
from rl4co.models.nn.mlp import MLP
from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
import torch
from typing import Optional, Type, Union

from rl4co.utils.ops import batchify, unbatchify
from tensordict import TensorDict
from torch import Tensor
from einops import rearrange


from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ACODHIRefiner(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        node_dim: int = 2,
        # n_ants: int =1,
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,

    ):
        super(ACODHIRefiner, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim
        self.n_heads = num_heads
        # self.n_ants =n_ants

        self.tour = nn.Linear(60, self.embed_dim, bias=False)
        self.FF_tour = nn.Sequential(
            nn. Linear(2*self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.embed_dim)
        )


        self.FF = nn.Sequential(
            nn. Linear(2*self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.embed_dim)
        )



        # for Max-Pooling sublayer
        self.project_graph_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits

    def calculate_distance(self, td_initial):
        x = td_initial["locs"][:, :, 0].unsqueeze(2)
        y = td_initial["locs"][:, :, 1].unsqueeze(2)

        dx = (x - x.transpose(1, 2)) ** 2
        dy = (y - y.transpose(1, 2)) ** 2

        distance_matrix = torch.sqrt(dx + dy)

        diagonal_mask = torch.eye(distance_matrix.size(1)).to(distance_matrix.device) * 1e9
        distance_matrix = distance_matrix + diagonal_mask
        distance_matrix[:, :, :td_initial["num_station"][0]+1] = 1e9
        distance_matrix[:, :td_initial["num_station"][0]+1, :] = 1e9
        reciprocal_distance_matrix = torch.reciprocal(distance_matrix)
        # result = reciprocal_distance_matrix - diagonal_mask


        

        return torch.log(distance_matrix)       

 

    def forward(self, 
                td: TensorDict,
                final_h: Tensor,
                # num_starts: int = 0,
                # decode_type: str = "multistart_sampling",
                # env: Optional[Union[str, RL4COEnvBase]] = None,
                # return_entropy: bool = False,
                **decoding_kwargs,
                 ) -> Tensor:
       
        dis = self.calculate_distance(td)
        
        dis_refined = self.tour(dis)

        cat_context = torch.cat((dis_refined, final_h), -1)

        ref_embedding = self.FF_tour(cat_context).squeeze(-1)
        ref_embedding = torch.nan_to_num(ref_embedding, nan=0.0)


        return ref_embedding


