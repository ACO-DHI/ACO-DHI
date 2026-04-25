import math

from typing import Callable, Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from rl4co.models.nn.env_embeddings.init import EVRPInitEmbedding
from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork
from rl4co.models.common import ImprovementEncoder
from rl4co.models.nn.attention import MultiHeadCompat
from rl4co.models.nn.ops import AdaptiveSequential, Normalization
from rl4co.utils.pylogger import get_pylogger
from rl4co.models.common.constructive import AutoregressiveEncoder
log = get_pylogger(__name__)


class Synth_Attention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int,num_station:int) -> None:
        super().__init__()
        

        hidden_dim = input_dim // n_heads

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_station=num_station

        self.W_query_custom = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_query_custom_1 = nn.Parameter(torch.Tensor(n_heads, input_dim,hidden_dim))
        self.W_key_custom = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_val_custom = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))

        self.W_query_charge = nn.Parameter(torch.Tensor(n_heads, input_dim,hidden_dim))
        self.W_query_charge_1 = nn.Parameter(torch.Tensor(n_heads, input_dim,hidden_dim))
        self.W_key_charge = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_val_charge = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        # self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))

        self.score_aggr = nn.Sequential(
            nn.Linear(2 * n_heads, 2 * n_heads),
            nn.ReLU(inplace=True),
            nn.Linear(2 * n_heads, n_heads),
        )

        self.W_out = nn.Parameter(torch.Tensor(n_heads, hidden_dim, input_dim))

        self.init_parameters()

    # used for init nn.Parameter
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h should be (batch_size, n_query, input_dim)
        batch_size, n_query, input_dim = h_fea.size()


        # hflat = h_fea.contiguous().view(-1, input_dim)

        # shp = (self.n_heads, batch_size, n_query, self.hidden_dim)

        # Calculate queries, (n_heads, batch_size, n_query, hidden_dim)
        # Q = torch.matmul(hflat, self.W_query).view(shp)   #改为异构就可以
        # K = torch.matmul(hflat, self.W_key).view(shp)
        # V = torch.matmul(hflat, self.W_val).view(shp)





        num_task=n_query-self.num_station-1

        #客户
        hflat_custom =h_fea[:,1+self.num_station:].contiguous().view(-1, input_dim)
        qflat_custom = h_fea[:,1+self.num_station:].contiguous().view(-1, input_dim)
        shp_q_custom = (self.n_heads, batch_size,  num_task, -1)
        shp_custom = (self.n_heads, batch_size, num_task, -1)

        K_custom = torch.matmul(hflat_custom, self.W_key_custom).view(shp_custom)
        V_custom=torch.matmul(hflat_custom, self.W_val_custom).view(shp_custom)
        
        #充电站
        hflat_station = h_fea[:,:1+self.num_station].contiguous().view(-1, input_dim)
        qflat_station= h_fea[:,:1+self.num_station].contiguous().view(-1, input_dim)
        shp_station=(self.n_heads, batch_size,self.num_station+1,-1)
        shp_q_station = (self.n_heads, batch_size,  self.num_station+1, -1)

        # Q_charge=torch.matmul(qflat_station, self.W_query_charge).view(shp_q_station)
        K_charge=torch.matmul(hflat_station, self.W_key_charge).view(shp_station)
        V_station=torch.matmul(hflat_station, self.W_val_charge).view(shp_station)

        #仓库
        


        #targe->targe
        Q_custom1=torch.matmul(qflat_custom, self.W_query_custom_1).view(shp_q_custom)
        compatibility1 = torch.cat((torch.matmul(Q_custom1, K_custom.transpose(2, 3)), aux_att_score[:, :, 1+self.num_station:, 1+self.num_station:]), 0)
        # compatibility1 = self.norm_factor * torch.matmul(Q_custom1, K_custom.transpose(2, 3))
        attn_raw1 = compatibility1.permute(1, 2, 3, 0)  
        attn1 = self.score_aggr(attn_raw1).permute(3, 0, 1, 2)
        heads_targe = torch.matmul(
            F.softmax(attn1, dim=-1), V_custom
        )

        # # attn1=torch.softmax(compatibility1, dim=-1)
        # heads_targe = torch.matmul(attn1,V_custom)

        #targe->station
        Q_custom2=torch.matmul(qflat_custom, self.W_query_custom).view(shp_q_custom)
        compatibility2 = torch.cat((torch.matmul(Q_custom2, K_charge.transpose(2, 3)), aux_att_score[:, :, 1+self.num_station:, :1+self.num_station]), 0)
        # compatibility2 = self.norm_factor * torch.matmul(Q_custom2, K_charge.transpose(2, 3))
        attn_raw2 = compatibility2.permute(1, 2, 3, 0)  
        attn2 = self.score_aggr(attn_raw2).permute(3, 0, 1, 2)
        # attn2=torch.softmax(compatibility2, dim=-1)
        # heads_targe+=torch.matmul(attn2,V_station)
        heads_targe += torch.matmul(
            F.softmax(attn2, dim=-1), V_station
        )

        # #station->targe
        Q_charge3=torch.matmul(qflat_station, self.W_query_charge_1).view(shp_q_station)
        compatibility3 = torch.cat((torch.matmul(Q_charge3, K_custom.transpose(2, 3)), aux_att_score[:, :, :1+self.num_station, 1+self.num_station:]), 0)
        # compatibility3 =  torch.matmul(Q_charge3, K_custom.transpose(2, 3))

        attn_raw3 = compatibility3.permute(1, 2, 3, 0)  
        attn3 = self.score_aggr(attn_raw3).permute(3, 0, 1, 2)
        # attn3=torch.softmax(compatibility3, dim=-1)
        heads_station = torch.matmul(
            F.softmax(attn3, dim=-1), V_custom
        )
        # heads_station = torch.matmul(attn3,V_custom)

        heads=torch.cat((heads_station,heads_targe),dim=2)
        # out=torch.mm(
        #     heads.permute(1, 2, 0, 3)
        #     .contiguous()
        #     .view(-1, self.n_heads * self.hidden_dim),
        #     self.W_out.view(-1, self.input_dim),
        # ).view(batch_size, n_query, self.input_dim)




        # Calculate compatibility (n_heads, batch_size, n_query, n_key)
        # compatibility = torch.cat((torch.matmul(Q, K.transpose(2, 3)), aux_att_score), 0)   #torch.matmul(Q, K.transpose(2, 3)#1，1，15，128

        # attn_raw = compatibility.permute(1, 2, 3, 0)  
        # attn = self.score_aggr(attn_raw).permute(3, 0, 1, 2) 

        # heads = torch.matmul(
        #     F.softmax(attn, dim=-1), V
        # )  # (n_heads, batch_size, n_query, hidden_dim)

        h_wave = torch.mm(
            heads.permute(1, 2, 0, 3)  # (batch_size, n_query, n_heads, hidden_dim)
            .contiguous()
            .view(
                -1, self.n_heads * self.hidden_dim
            ),  # (batch_size * n_query, n_heads * hidden_dim)
            self.W_out.view(-1, self.input_dim),  # (n_heads * hidden_dim, input_dim)
        ).view(batch_size, n_query, self.input_dim)

        return h_wave, aux_att_score


class SynthAttNormSubLayer(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, num_station:int, normalization: str) -> None:
        super().__init__()

        self.SynthAtt = Synth_Attention(n_heads, input_dim,num_station)

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention and Residual connection
        h_wave, aux_att_score = self.SynthAtt(h_fea, aux_att_score)

        # Normalization
        return self.Norm(h_wave + h_fea), aux_att_score


class FFNormSubLayer(nn.Module):
    def __init__(
        self, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.FF = (
            nn.Sequential(
                nn.Linear(input_dim, feed_forward_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, input_dim, bias=False),
            )
            if feed_forward_hidden > 0
            else nn.Linear(input_dim, input_dim, bias=False)
        )

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class N2SEncoderLayer(nn.Module):
    def __init__(
        self, n_heads: int, input_dim: int, num_station:int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.SynthAttNorm_sublayer = SynthAttNormSubLayer(
            n_heads, input_dim, num_station,normalization
        )

        self.FFNorm_sublayer = FFNormSubLayer(
            input_dim, feed_forward_hidden, normalization
        )

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_wave, aux_att_score = self.SynthAttNorm_sublayer(h_fea, aux_att_score)
        return self.FFNorm_sublayer(h_wave), aux_att_score


class DeepACOEncoder(AutoregressiveEncoder):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        num_station: int = 4,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
    ):
        super(DeepACOEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.init_embedding = EVRPInitEmbedding(embed_dim)

        self.pos_net = MultiHeadCompat(num_heads, embed_dim, feedforward_hidden)
        self.tour = nn.Linear(20, embed_dim, bias=False)

        self.net = AdaptiveSequential(
            *(
                N2SEncoderLayer(
                    num_heads,
                    embed_dim,
                    num_station,
                    feedforward_hidden,
                    normalization,
                )
                for _ in range(num_layers)
            )
        )

    def calculate_distance(self, td_initial):
            x = td_initial["locs"][:, :, 0].unsqueeze(2)
            y = td_initial["locs"][:, :, 1].unsqueeze(2)

            dx = (x - x.transpose(1, 2)) ** 2
            dy = (y - y.transpose(1, 2)) ** 2

            distance_matrix = torch.sqrt(dx + dy)

            diagonal_mask = torch.eye(distance_matrix.size(1)).to(distance_matrix.device) * 1e9
            distance_matrix = distance_matrix + diagonal_mask
            # distance_matrix[:, :, :td_initial["num_station"][0]+1] = 1e9
            # distance_matrix[:, :td_initial["num_station"][0]+1, :] = 1e9
            batch_size, graph_size, dim = distance_matrix.size()

            tw_matrix = (td_initial["time_windows"][..., 1]).unsqueeze(1).repeat(1,  dim,1)

            distance_matrix_tw = (distance_matrix * tw_matrix) / (distance_matrix + tw_matrix)

            return torch.log(distance_matrix_tw) 


    def forward(self, td: Tensor) -> Tuple[Tensor, Tensor]:
        init_h = self.init_embedding(td)
        batch_size, graph_size, dim = init_h.size()

        refdis = self.calculate_distance(td)

        # dis_refined = self.tour(refdis)

        # RemainingBattery = td["max_length"]-td["used_length"]  #就是初始电量
        # RemainingCapacity = td["vehicle_capacity"]-td["used_capacity"]
        # veh_context = torch.cat((RemainingBattery, RemainingCapacity),-1)
        # veh_context = rearrange(veh_context, "(s b) l -> b s l", b=batch_size)

        init_e = self.tour(refdis).expand(batch_size, graph_size, dim)

        embed_e = self.pos_net(init_e)   #辅助信息

        final_h, final_p = self.net(init_h, embed_e)

        return final_h, final_p
