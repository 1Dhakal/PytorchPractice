import torch
# from sympy.physics.quantum.identitysearch import random_identity_search
# from torch_geometric.data import Data
#
# # 1. Define a mini graph with 3 nodes and edges between them
# edge_index = torch.tensor([[0,1,1,2], # source nodes
#                            [1,0,2,1]], # target nodes
#                           dtype=torch.long)
# # 2. Features for each node (3 nodes, each with 1 feature)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index)
# # print(data)
#
# data.validate(raise_on_error=True)

# from torch_geometric.datasets import TUDataset
# dataset = TUDataset(root='/tmp/ENZYMES', name = 'ENZYMES')
#
# print(dataset)
# print(dataset.num_classes)
# print(dataset.num_node_features)
# data = dataset[0]
#
# dataset = dataset.shuffle()

# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='/tmp/Cora', name = 'Cora')
# print(dataset)
#
# data = dataset[0]
# print(data)
#
# print(data.is_undirected())
#
# if data.is_undirected():
#     print("data is Undirected")
# else :
#     print("No does Not")
#


from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

# dataset = TUDataset('root=/tmp/ENZYMES', name='ENZYMES', use_node_attr = True)
# loader = DataLoader(dataset, batch_size = 32, shuffle=True)
# #
# for data in loader:
#     x = scatter(data.x, data.batch, dim=0, reduce='mean')
#     print(x.size())


# from torch_geometric.datasets import ShapeNet
#
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
#
# dataset[0]



# from torch_geometric.datasets import ModelNet
# import torch_geometric.transforms as T
#
# # Preprocessing steps: build graph, normalize once
# pre_transform = T.Compose([
#     T.SamplePoints(num=1024),     # Sample 1024 points from mesh
#     T.KNNGraph(k=6),              # Build 6-nearest neighbor graph
#     T.NormalizeScale()            # Normalize scale to unit sphere
# ])
#
# # Runtime transform: add random jitter noise during each access
# transform = T.RandomJitter(sigma=0.01, clip=0.05)
#
# # Load dataset
# dataset = ModelNet(
#     root='/tmp/ModelNet',
#     name='10',
#     train=True,
#     pre_transform=pre_transform,
#     transform=transform
# )
#
# # Check one sample
# print(dataset[0])

#
# import torch_geometric.transforms as T
# from torch_geometric.datasets import TUDataset
#
# transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
#
# dataset = TUDataset(path, name='MUTAG', transform=transform)
# data = dataset[0]  # Implicitly transform data on every access.
#
# data = TUDataset(path, name='MUTAG')[0]
# data = transform(data)  # Explicitly transform data.



##############################   QUESTION   ########################
#Load the "IMDB-BINARY" dataset from the TUDataset benchmark suite and randomly split it into 80%/10%/10% training, validation and test graphs.

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset =TUDataset(root='/tmp/IMDB-BINARY', name = 'IMDB-BINARY')

#shuffle the dataset
dataset = dataset.shuffle()

num_graphs = len(dataset)
train_len = int(0.8 * num_graphs)
val_len = int(0.1 * num_graphs)
test_len = num_graphs - train_len - val_len

#split the database
train_dataset = dataset[:train_len]
val_dataset = dataset[train_len:train_len + val_len]
test_dataset = dataset[train_len + val_len:]
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Train:{len(train_dataset)}, val:{len(val_dataset)}, test: {len(test_dataset)}")

print(dataset)
data = dataset[0]
print(data)

