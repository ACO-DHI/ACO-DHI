

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


class DeepACORefiner(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        node_dim: int = 3,
        # n_ants: int =1,
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,

    ):
        super(DeepACORefiner, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim
        self.n_heads = num_heads
        # self.n_ants =n_ants
        self.tour1 = nn.Linear(node_dim, self.embed_dim, bias=False)

        self.tour = nn.Linear(15, self.embed_dim, bias=False)
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

    # def calculate_distance(self, td_initial):
    #     x = td_initial["locs"][:, :, 0].unsqueeze(2)
    #     y = td_initial["locs"][:, :, 1].unsqueeze(2)

    #     dx = (x - x.transpose(1, 2)) ** 2
    #     dy = (y - y.transpose(1, 2)) ** 2

    #     distance_matrix = torch.sqrt(dx + dy)

    #     diagonal_mask = torch.eye(distance_matrix.size(1)).to(distance_matrix.device) * 1e9
    #     distance_matrix = distance_matrix + diagonal_mask
    #     distance_matrix[:, :, :td_initial["num_station"][0]+1] = 1e9
    #     distance_matrix[:, :td_initial["num_station"][0]+1, :] = 1e9
    #     reciprocal_distance_matrix = torch.reciprocal(distance_matrix)
    #     # result = reciprocal_distance_matrix - diagonal_mask


        

    #     return torch.log(distance_matrix)       

        # # for feed-forward aggregation (FFA)sublayer
        # self.value_head = MLP(
        #     input_dim=2 * self.embed_dim,
        #     output_dim=128,
        #     num_neurons=[128, 128],
        #     dropout_probs=[0.05, 0.00],
        # )
    

    def forward(self, 
                td: TensorDict,
                final_h: Tensor,
                # refdis: any,
                # num_starts: int = 0,
                # decode_type: str = "multistart_sampling",
                # env: Optional[Union[str, RL4COEnvBase]] = None,
                # return_entropy: bool = False,
                **decoding_kwargs,
                 ) -> Tensor:
        batch_size, graph_size, dim = final_h.size()
        # decode_strategy: DecodingStrategy = get_decoding_strategy(
        #     decode_type,
        #     temperature=decoding_kwargs.pop("temperature", self.temperature),
        #     tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
        #     mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
        #     store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
        #     num_starts=decoding_kwargs.pop("store_all_logp", num_starts),
        #     **decoding_kwargs,
        # )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy

        # td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # if  num_starts > 1:
        #     td = unbatchify(td, num_starts)

        # RemainingBattery = td["max_length"]-td["used_length"]  #就是初始电量
        # RemainingCapacity = td["vehicle_capacity"]-td["used_capacity"]

      #  h_node_refined = self.project_node_node(final_h) + self.project_graph_node(
          #  final_h.max(1)[0])[:, None, :].expand(batch_size, graph_size, dim)
        
        # veh_context = torch.cat((RemainingBattery, RemainingCapacity),-1)
        # veh_context = rearrange(veh_context, "(s b) l -> b s l", b=batch_size)

        # veh_context_refined = self.tour(veh_context)
        # dis = self.calculate_distance(td)
        # # 检查第0维度所有值是否相同
        # if (dis[0, :, :] == dis).all():
        #     # 使用 `unsqueeze` 函数增加维度
        #      dis = dis[0, :, :].unsqueeze(0)
        
        # dis_refined = self.tour(refdis)    #修改，一样的只留一个


        RemainingBattery = td["max_length"]-td["used_length"]  #shengyudianliang
        RemainingCapacity = td["vehicle_capacity"]-td["used_capacity"]  #shengyurongliang
        Remainingtime = td["current_time"]   #dangqianshijian

        h_node_refined = self.project_node_node(final_h) + self.project_graph_node(
            final_h.max(1)[0])[:, None, :]
        
        veh_context = torch.cat((RemainingBattery, RemainingCapacity,Remainingtime),-1)
        veh_context = rearrange(veh_context, "(s b) l -> b s l", b=batch_size)

        # veh_context_refined = self.tour(veh_context)
        
        veh_context_refined = self.tour1(veh_context).expand(batch_size, graph_size, dim)

        cat_context = torch.cat((veh_context_refined, h_node_refined), -1)

        ref_embedding = self.FF_tour(cat_context).squeeze(-1)
 

        # cat_context = torch.cat((dis_refined, final_h), -1)

        # ref_embedding = self.FF_tour(cat_context).squeeze(-1)
        # ref_embedding = torch.nan_to_num(ref_embedding, nan=0.0)

        # finalembcat = torch.cat((ref_embedding, final_h), -1)
        # finalemb = self.FF(finalembcat).squeeze(-1)

        return ref_embedding


