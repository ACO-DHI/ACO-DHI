
from rl4co.models.nn.graph.attnnet import Normalization, SkipConnection
from rl4co.models.zoo.acodhi.attention import HMHA
import torch.nn as nn
from rl4co.models.nn.env_embeddings.init import EVRPInitEmbedding
from rl4co.envs import RL4COEnvBase


class HMHALayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        num_station,
        feed_forward_hidden=512,
        normalization="batch",
    ):
        super(HMHALayer, self).__init__(
            SkipConnection(HMHA(num_heads, embed_dim,num_station, embed_dim)),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class GraphHAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_station,
        init_embedding=None,
        num_heads=8,        
        embed_dim=128,
        num_layers=3,
        env_name=None,
        normalization="batch",
        feedforward_hidden=512,
        sdpa_fn=None,
    ):
        super(GraphHAttentionEncoder, self).__init__()
        
        # substitute env_name with pdp if none
        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.init_embedding = EVRPInitEmbedding(embed_dim)
        

        self.layers = nn.Sequential(
            *(
                HMHALayer(
                    num_heads,
                    embed_dim,
                    num_station,
                    feedforward_hidden,
                    normalization,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)  # (batch_size, graph_size, embed_dim)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds
