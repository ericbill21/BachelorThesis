#import HGP_SL_model
import torch
import torch_geometric
from torch_geometric.nn.models import GIN, MLP


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

def create_model(
    model_name: str,
    output_dim: int,
    is_classification: bool,
    input_dim: int = None,
    mlp_hidden_layer_conf: list = [],
    gnn_layers: int = None,
    activation_func: str = "relu",
    dropout: float = 0.0,
    mlp_norm: str = "batch_norm",
    jk: str = "cat",
    gnn_hidden_channels: int = 16,
    encoding_kwargs: dict = {},
    pool_func_kwargs: dict = {},
):
    
    # Check if the model is a 1WL+NN model
    if model_name.startswith("1WL+NN:"):
        # Retrieve the correct encoding function
        if "Embedding" in model_name:
            assert encoding_kwargs['embedding_dim'] > 0, "Key 'embedding_dim' must be defined in encoding_kwargs as a postive integer greater zero."

            encoding = torch.nn.Embedding(
                num_embeddings=encoding_kwargs['largest_color'],
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
            assert pool_func_kwargs["processing_steps"] > 0, "Key 'processing_steps' must be defined in encoding_kwargs as a postive integer greater zero."

            pool_func = torch_geometric.nn.aggr.Set2Set(in_channels=mlp_input_channels, **pool_func_kwargs)
            mlp_input_channels = pool_func.out_channels

        # Construct the MLP
        mlp = MLP(channel_list=[mlp_input_channels] + mlp_hidden_layer_conf + [output_dim],
                    act=activation_func,
                    dropout=dropout,
                    mlp_norm=mlp_norm,
        )

        # Construct the model
        model = generic_wlnn(encoding=encoding,
                             mlp=mlp,
                             pool_func=pool_func)
    
    # Check if the model is a GIN model
    elif model_name.startswith("GIN"):
        # Construct the GIN
        gin = GIN(
            in_channels=input_dim,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_layers,
            dropout=dropout,
            mlp_norm=mlp_norm,
            act=activation_func,
            jk=jk,
        )

        # Remove the last linear layer that would otherwise remove all jk information
        if jk == "cat":
            delattr(gin, "lin")
            mlp_input_channels = gin.out_channels * gin.num_layers
        elif jk == "last" or jk == "sum" or jk == "mean" or jk == "max":
            mlp_input_channels = gin.out_channels
        else:
            mlp_input_channels = gin.out_channels

        # Retrieve the correct pooling function
        if model_name == "GIN:Max":
            pool_func = torch_geometric.nn.pool.global_max_pool
        elif model_name == "GIN:Mean":
            pool_func = torch_geometric.nn.pool.global_mean_pool
        elif model_name == "GIN:Sum":
            pool_func = torch_geometric.nn.pool.global_add_pool
        elif model_name == "GIN:Set2Set":
            assert pool_func_kwargs["processing_steps"] > 0, "Key 'processing_steps' must be defined in encoding_kwargs as a postive integer greater zero."

            pool_func = torch_geometric.nn.aggr.Set2Set(in_channels=mlp_input_channels, **pool_func_kwargs)
            mlp_input_channels = pool_func.out_channels

        # Construct the MLP
        mlp = MLP(channel_list=[mlp_input_channels] + mlp_hidden_layer_conf + [output_dim],
                    act=activation_func,
                    dropout=dropout,
                    mlp_norm=mlp_norm,
        )

        # Construct the model
        model = generic_gnn(gnn=gin,
                            mlp=mlp,
                            pool_func=pool_func)
    
    elif model_name == "HGP_SL":

        class Object(object):
            pass

        dummy_object = Object()
        dummy_object.num_features = input_dim
        dummy_object.nhid = 128
        dummy_object.num_classes = output_dim
        dummy_object.pooling_ratio = 0.5
        dummy_object.dropout_ratio = 0.0
        dummy_object.sample_neighbor = True
        dummy_object.sparse_attention = True
        dummy_object.structure_learning = True
        dummy_object.lamb = 1.0

        #model = HGP_SL_model.Model(dummy_object)
    

    return model
