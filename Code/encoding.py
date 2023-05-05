import torch
from abc import ABC, abstractmethod
from wlnn import wl_algorithm
from torch import nn


class Encoding_Function(nn.Module):
    
    def __init__(self, embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.out_dim = embedding.embedding_dim
    
    @abstractmethod
    def forward(self, coloring):
        pass


class Counting_Encoding(Encoding_Function):
    def __init__(self, embedding) -> None:
        super().__init__(embedding)

    def forward(self, data):
        # We count the number of nodes for each color
        out = torch.zeros((data.num_graphs, self.out_dim), dtype=torch.float32)
        
        for i in range(data.num_graphs):
            for c in data.x[data.ptr[i]: data.ptr[i+1]]:
                if c < self.out_dim:
                    out[i][c] += 1

        return out
    
class Summation_Encoding(Encoding_Function):
    def __init__(self, embedding) -> None:
        super().__init__(embedding)

    def forward(self, data):
        out = torch.zeros((data.num_graphs, self.out_dim), dtype=torch.float32)
        
        for i in range(data.num_graphs):
            out[i] = torch.sum(data.x[data.ptr[i]: data.ptr[i+1]], dim=0)
        
        return out

class Mean_Encoding(Encoding_Function):
    def __init__(self, embedding) -> None:
        super().__init__(embedding)

    def forward(self, data):
        x = self.embedding(data.x).squeeze()
        out = torch.zeros((data.num_graphs, self.out_dim), dtype=torch.float32)
        
        for i in range(data.num_graphs):
            out[i] = torch.sum(x[data.ptr[i]: data.ptr[i+1]], dim=0) / (data.ptr[i+1] - data.ptr[i])
        
        return out
