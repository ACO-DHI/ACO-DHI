import math


import torch
import torch.nn as nn

class HMHA(nn.Module):
    def __init__(
        self,
        num_heads,
        input_dim,
        num_station,
        embed_dim=None,
        val_dim=None,
        key_dim=None
    ):
        super(HMHA, self).__init__()
        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // num_heads
        if key_dim is None:
            key_dim = val_dim
            
        self.num_station=num_station
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        
        self.W_query_custom = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_query_custom_1 = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key_custom = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val_custom = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))

        self.W_query_charge = nn.Parameter(torch.Tensor(num_heads, input_dim,key_dim))
        self.W_query_charge_1 = nn.Parameter(torch.Tensor(num_heads, input_dim,key_dim))
        self.W_key_charge = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val_charge = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))

        # self.W_query_depot = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        # self.W_query_depot_1 = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))

        self.W_out = nn.Parameter(torch.Tensor(num_heads, key_dim, embed_dim))
        self.init_parameters()
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    def forward(self, q,h=None, mask=None):
        if h is None:
            h = q
        batch_size, graph_size, input_dim =h.size()
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"
        num_task=graph_size-self.num_station-1


        hflat_custom = h[:,1+self.num_station:].contiguous().view(-1, input_dim)
        qflat_custom = q[:,1+self.num_station:].contiguous().view(-1, input_dim)
        shp_q_custom = (self.num_heads, batch_size,  num_task, -1)
        shp_custom = (self.num_heads, batch_size, num_task, -1)

        K_custom = torch.matmul(hflat_custom, self.W_key_custom).view(shp_custom)
        V_custom=torch.matmul(hflat_custom, self.W_val_custom).view(shp_custom)
        

        hflat_station = h[:,:1+self.num_station].contiguous().view(-1, input_dim)
        qflat_station= q[:,:1+self.num_station].contiguous().view(-1, input_dim)
        shp_station=(self.num_heads, batch_size,self.num_station+1,-1)
        shp_q_station = (self.num_heads, batch_size,  self.num_station+1, -1)

        Q_charge=torch.matmul(qflat_station, self.W_query_charge).view(shp_q_station)
        K_charge=torch.matmul(hflat_station, self.W_key_charge).view(shp_station)
        V_station=torch.matmul(hflat_station, self.W_val_charge).view(shp_station)




        #targe->targe
        Q_custom=torch.matmul(qflat_custom, self.W_query_custom_1).view(shp_q_custom)
        compatibility = self.norm_factor * torch.matmul(Q_custom, K_custom.transpose(2, 3))
        attn=torch.softmax(compatibility, dim=-1)
        heads_targe = torch.matmul(attn,V_custom)

        #targe->station
        Q_custom=torch.matmul(qflat_custom, self.W_query_custom).view(shp_q_custom)
        compatibility = self.norm_factor * torch.matmul(Q_custom, K_charge.transpose(2, 3))
        attn=torch.softmax(compatibility, dim=-1)
        heads_targe+=torch.matmul(attn,V_station)

        # #station->targe
        Q_charge=torch.matmul(qflat_station, self.W_query_charge_1).view(shp_q_station)
        compatibility = self.norm_factor * torch.matmul(Q_charge, K_custom.transpose(2, 3))
        attn=torch.softmax(compatibility, dim=-1)
        heads_station = torch.matmul(attn,V_custom)

        heads=torch.cat((heads_station,heads_targe),dim=2)
        out=torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.num_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, graph_size, self.embed_dim)
        return out