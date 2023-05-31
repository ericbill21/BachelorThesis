#import HGP_SL_model
import math

import torch
import torch_geometric
from torch_geometric.nn.models import GAT, GIN, MLP


class generic_wlnn(torch.nn.Module):
    def __init__(
        self,
        pool_func: torch.nn.Module,
        encoding: torch.nn.Module,
        mlp: torch.nn.Module,
    ):
        super().__init__()

        self.embedding = encoding
        self.mlp = mlp
        self.pool = pool_func
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding(x).squeeze()
        x = self.pool(x, batch)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

    def reset_parameters(self):
        if hasattr(self.embedding, "reset_parameters"):
            self.embedding.reset_parameters()
        
        self.mlp.reset_parameters()

        if hasattr(self.pool, "reset_parameters"):
            self.pool.reset_parameters()

class generic_gnn(torch.nn.Module):
    def __init__(
        self,
        gnn: torch.nn.Module,
        pool_func: torch.nn.Module,
        mlp: torch.nn.Module,
    ):
        super().__init__()

        self.gnn = gnn
        self.mlp = mlp
        self.pool = pool_func
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp.reset_parameters()

        if hasattr(self.pool, "reset_parameters"):
            self.pool.reset_parameters()

def create_model(model_name: str,
                    input_dim: int,
                    output_dim: int,
                    mlp_kwargs: dict = {},
                    gnn_kwargs: dict = {},
                    encoding_kwargs: dict = {}):
    
    # Check if the model is a 1WL+NN model
    if model_name.startswith("1WL+NN:"):
        # Retrieve the correct encoding function
        if "Embedding" in model_name:
            assert encoding_kwargs['embedding_dim'] > 0 and encoding_kwargs['max_node_feature'] > 0, "Key 'embedding_dim' and 'max_node_feature' must be defined in encoding_kwargs as a postive integer greater zero."

            encoding = torch.nn.Embedding(
                num_embeddings=encoding_kwargs['max_node_feature'],
                embedding_dim=encoding_kwargs['embedding_dim'],
            )
            mlp_input_channels = encoding_kwargs['embedding_dim']
        else:
            # TODO: Check if it is problematic with the squeeze
            encoding = torch.nn.Identity()
            mlp_input_channels = 1

        # Retrieve the correct pooling function
        if "Sum" in model_name:
            pool_func = torch_geometric.nn.pool.global_add_pool
        elif "Max" in model_name:
            pool_func = torch_geometric.nn.pool.global_max_pool
        elif "Mean" in model_name:
            pool_func = torch_geometric.nn.pool.global_mean_pool
        elif "Set2Set" in model_name:
            assert encoding_kwargs["processing_steps"] > 0, "Key 'processing_steps' must be defined in encoding_kwargs as a postive integer greater zero."

            pool_func = torch_geometric.nn.aggr.Set2Set(in_channels=mlp_input_channels, **encoding_kwargs)
            mlp_input_channels = pool_func.out_channels
        else:
            raise ValueError(f"Invalid pooling function: {model_name}")

        # Construct the MLP
        if "hidden_channels" not in mlp_kwargs:
            mlp_kwargs["hidden_channels"] = mlp_input_channels
        mlp = MLP(in_channels=mlp_input_channels, out_channels=output_dim, **mlp_kwargs)

        # Construct the model
        model = generic_wlnn(encoding=encoding,
                             mlp=mlp,
                             pool_func=pool_func)
    
    # Check if the model is a GIN model
    elif model_name.startswith("GIN") or model_name.startswith("GAT") or model_name.startswith("GCN"):
        # Construct the GNN
        if model_name.startswith("GIN"):
            gnn = GIN(in_channels=input_dim, **gnn_kwargs)
        elif model_name.startswith("GAT"):
            gnn = GAT(in_channels=input_dim, **gnn_kwargs)
        elif model_name.startswith("GCN"):
            gnn = torch_geometric.nn.GCN(in_channels=input_dim, **gnn_kwargs)
        
        mlp_input_channels = gnn.out_channels

        # Retrieve the correct pooling function
        if "Max" in model_name:
            pool_func = torch_geometric.nn.pool.global_max_pool
        elif "Mean" in model_name:
            pool_func = torch_geometric.nn.pool.global_mean_pool
        elif "Sum" in model_name:
            pool_func = torch_geometric.nn.pool.global_add_pool
        elif "Set2Set" in model_name:
            assert encoding_kwargs["processing_steps"] > 0, "Key 'processing_steps' must be defined in encoding_kwargs as a postive integer greater zero."

            pool_func = torch_geometric.nn.aggr.Set2Set(in_channels=mlp_input_channels, **encoding_kwargs)
            mlp_input_channels = pool_func.out_channels
        else:
            raise ValueError(f"Invalid pooling function: {model_name}")

        # Construct the MLP: 
        if "hidden_channels" not in mlp_kwargs:
            mlp_kwargs["hidden_channels"] = mlp_input_channels
        mlp = MLP(in_channels=mlp_input_channels, out_channels=output_dim, **mlp_kwargs)

        # Construct the model
        model = generic_gnn(gnn=gnn,
                            mlp=mlp,
                            pool_func=pool_func)
    
    return model
