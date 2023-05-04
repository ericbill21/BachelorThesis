import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import wl_conv


edge_index = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]])
x = torch.zeros(5, dtype=torch.long).unsqueeze(-1)

data = Data(x=x, edge_index=edge_index)

wl = wl_conv.WLConv()

print('Hey')



