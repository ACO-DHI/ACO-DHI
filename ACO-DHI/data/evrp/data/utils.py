import os
import re
import numpy as np
import torch
from tensordict.tensordict import TensorDict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURR_DIR))




def load_file_to_tensordict(path):
    dirs = os.listdir( path)
    demand_all=torch.zeros((0,100))
    depot_all=torch.zeros((0,2))
    stations_all=torch.zeros((0,21,2))
    loc_all=torch.zeros((0,100,2))
    max_length_all=[]
    for file in dirs:
        filename=path+file
        f=open(filename,'r')
        content =f.read()
        capacity = float(re.search("C Vehicle load capacity /(\d+\.?\d*)/", content, re.MULTILINE).group(1))
        energy_capacity = float(re.search("Q Vehicle fuel tank capacity /(\d+\.?\d*)/", content, re.MULTILINE).group(1))
        energy_consumption = float(re.search("r fuel consumption rate /(\d+\.?\d*)/", content, re.MULTILINE).group(1))
        max_length=energy_capacity/energy_consumption/3
        max_length_all.append(max_length)
        depot=re.findall(r"d          (-?\d+\.?\d* ?)       (-?\d+\.?\d*)", content, re.MULTILINE)
        depot=torch.Tensor([[float(a),float(b)] for a,b in depot])
        stations=re.findall(r"f          (-?\d+\.?\d* ?)       (-?\d+\.?\d*)", content, re.MULTILINE)
        stations=torch.Tensor([[float(a),float(b)] for a,b in stations])
        customs=re.findall(r"c          (-?\d+\.?\d* ?)       (-?\d+\.?\d* ?)       (\d+\.?\d*)", content, re.MULTILINE)
        customs=torch.Tensor([[float(a),float(b),float(c)] for a,b,c in customs])
        demand=customs[:,-1]/capacity
        demand_all=torch.cat((demand_all,demand.unsqueeze(0)))
        customs=customs[:,:-1]
        nodes=torch.cat((depot,stations,customs),dim=-2)
        bias=torch.min(nodes,dim=0).values.unsqueeze(0)
        depot=(depot-bias)/max_length
        depot_all=torch.cat((depot_all,depot))
        stations=(stations-bias)/max_length
        stations_all=torch.cat((stations_all,stations.unsqueeze(0)))
        customs=(customs-bias)/max_length
        loc_all=torch.cat((loc_all,customs.unsqueeze(0)))
    td=TensorDict(
        {
            "locs":loc_all,
            "depot":depot_all,
            "stations":stations_all,
            "demand":demand_all,
            "factor":torch.Tensor(max_length_all)
        },
        batch_size=[len(dirs)]
    )
    return td

def load_evrp_to_tensordict(filename):
    f=open(filename,'r')
    content =f.read()
    vehicles = torch.Tensor([int(re.search("VEHICLES: (\d+)", content, re.MULTILINE).group(1))]).unsqueeze(0)
    optimalValue = float(re.search("OPTIMAL_VALUE: (\d+)", content, re.MULTILINE).group(1))
    capacity = float(re.search("CAPACITY: (\d+)", content, re.MULTILINE).group(1))
    dimension = int(re.search("DIMENSION: (\d+)", content, re.MULTILINE).group(1))
    station_number = int(re.search("STATIONS: (\d+)", content, re.MULTILINE).group(1))
    energy_capacity = float(re.search("ENERGY_CAPACITY: (\d+)", content, re.MULTILINE).group(1))
    energy_consumption = float(re.search("ENERGY_CONSUMPTION: (\d+\.?\d*)", content, re.MULTILINE).group(1))
    max_length=energy_capacity/energy_consumption/3
    # max_length=energy_capacity/energy_consumption/100
    demand = re.findall(r"^(\d+) (\d+)$", content, re.MULTILINE)
    demand = torch.Tensor([float(b) for a, b in demand][1:]).unsqueeze(0)
    nodes = re.findall(r"^(\d+) (\d+) (\d+)", content, re.MULTILINE)
    nodes = torch.Tensor([[float(b),float(c)] for a, b,c in nodes])
    bias=torch.min(nodes,dim=0).values.unsqueeze(0)
    nodes=(nodes-bias)/max_length
    # nodes=nodes/max_length
    depot=nodes[0].unsqueeze(0)
    stations=nodes[-station_number:].unsqueeze(0)
    locs=nodes[1:-station_number].unsqueeze(0)
    td=TensorDict(
        {
            "locs":locs,
            "depot":depot,
            "stations":stations,
            "demand":demand/capacity,
            # "max_length": torch.Tensor([max_length]),
            # "factor":torch.Tensor([100])
            "factor":torch.Tensor([max_length]),
        },
        batch_size=[1]
    )
    return td








def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = dict(x)
    batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)


def save_tensordict_to_npz(tensordict, filename, compress: bool = False):
    """Save a TensorDict to a npz file
    We assume that the TensorDict contains a dictionary of tensors
    """
    x_dict = {k: v.numpy() for k, v in tensordict.items()}
    if compress:
        np.savez_compressed(filename, **x_dict)
    else:
        np.savez(filename, **x_dict)


def check_extension(filename, extension=".npz"):
    """Check that filename has extension, otherwise add it"""
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def load_solomon_instance(name, path=None, edge_weights=False):
    """Load solomon instance from a file"""
    import vrplib

    if not path:
        path = "data/solomon/instances/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.txt"
    if not os.path.isfile(file_path):
        vrplib.download_instance(name=name, path=path)
    return vrplib.read_instance(
        path=file_path,
        instance_format="solomon",
        compute_edge_weights=edge_weights,
    )


def load_solomon_solution(name, path=None):
    """Load solomon solution from a file"""
    import vrplib

    if not path:
        path = "data/solomon/solutions/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.sol"
    if not os.path.isfile(file_path):
        vrplib.download_solution(name=name, path=path)
    return vrplib.read_solution(path=file_path)