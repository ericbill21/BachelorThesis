import torch
import torch_geometric
from torch_geometric.nn.models import GIN, MLP


def load_model(model_name: str,
                output_dim: int,
                is_classification: bool,
                input_dim: int = None,
                largest_color: int = None,
                embedding_dim: int = None, 
                mlp_hidden_layer_conf: list = [], 
                gnn_layers: int = None, 
                activation_func: str = 'relu', 
                dropout: float = 0.0, 
                mlp_norm: str = 'batch_norm', 
                jk: str = 'cat', 
                gnn_hidden_channels: int = 16):

    if model_name == "1WL+NN:Embedding-Sum":

        if is_classification:

            if largest_color is None:
                raise ValueError("largest_color must be specified for classification tasks")
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified for classification tasks")

            model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                                    (torch.nn.Embedding(num_embeddings=largest_color, embedding_dim=embedding_dim), 'x -> x'),
                                    (torch.squeeze, 'x -> x'),
                                    (torch_geometric.nn.pool.global_add_pool, 'x, batch -> x'),
                                    (MLP(channel_list=[embedding_dim] + mlp_hidden_layer_conf + [output_dim], act=activation_func, dropout=dropout, mlp_norm=mlp_norm), 'x -> x'),
                                    (torch.nn.Softmax(dim=1), 'x -> x')
                                ])
            
        else:
            raise ValueError("1WL+NN:Embedding-Max is not implemented for regression tasks")

    
    elif model_name == "1WL+NN:Embedding-Max":

        if is_classification:

            if largest_color is None:
                raise ValueError("largest_color must be specified for classification tasks")
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified for classification tasks")

            model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                                    (torch.nn.Embedding(num_embeddings=largest_color, embedding_dim=embedding_dim), 'x -> x'),
                                    (torch.squeeze, 'x -> x'),
                                    (torch_geometric.nn.pool.global_max_pool, 'x, batch -> x'),
                                    (MLP(channel_list=[embedding_dim] + mlp_hidden_layer_conf + [output_dim], act=activation_func, dropout=dropout, mlp_norm=mlp_norm), 'x -> x'),
                                    (torch.nn.Softmax(dim=1), 'x -> x')
                                ])
            
        else:
            raise ValueError("1WL+NN:Embedding-Max is not implemented for regression tasks")
    
    elif model_name == "1WL+NN:Embedding-Mean":

        if is_classification:

            if largest_color is None:
                raise ValueError("largest_color must be specified for classification tasks")
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified for classification tasks")

            model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                                    (torch.nn.Embedding(num_embeddings=largest_color, embedding_dim=embedding_dim), 'x -> x'),
                                    (torch.squeeze, 'x -> x'),
                                    (torch_geometric.nn.pool.global_mean_pool, 'x, batch -> x'),
                                    (MLP(channel_list=[embedding_dim] + mlp_hidden_layer_conf + [output_dim], act=activation_func, dropout=dropout, mlp_norm=mlp_norm), 'x -> x'),
                                    (torch.nn.Softmax(dim=1), 'x -> x')
                                ])
        
        else:
            raise ValueError("1WL+NN:Embedding-Max is not implemented for regression tasks")

    elif model_name == "GIN:Sum":
        gin = GIN(in_channels=input_dim,
                   hidden_channels=gnn_hidden_channels, 
                   num_layers=gnn_layers, 
                   dropout=dropout, 
                   mlp_norm=mlp_norm, 
                   act=activation_func, 
                   jk=jk)

        # Remove the last linear layer that would otherwise remove all jk information
        if jk == 'cat':
            delattr(gin, 'lin')
            input_dim = gin.out_channels * gin.num_layers
        elif jk == 'last' or jk == 'sum' or jk == 'mean' or jk == 'max':
            input_dim = gin.out_channels

        model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                        (gin, 'x, edge_index -> x'),
                        (torch_geometric.nn.pool.global_add_pool, 'x, batch -> x'),
                        (MLP(channel_list=[input_dim] + mlp_hidden_layer_conf + [output_dim], act=activation_func, dropout=dropout, mlp_norm=mlp_norm), 'x -> x'),
                        (torch.nn.Softmax(dim=1), 'x -> x')
                    ])
    
    elif model_name == "GIN:Max":
        gin = GIN(in_channels=input_dim,
                   hidden_channels=gnn_hidden_channels, 
                   num_layers=gnn_layers, 
                   dropout=dropout, 
                   mlp_norm=mlp_norm, 
                   act=activation_func, 
                   jk=jk)

        # Remove the last linear layer that would otherwise remove all jk information
        if jk == 'cat':
            delattr(gin, 'lin')
            input_dim = gin.out_channels * gin.num_layers
        elif jk == 'last' or jk == 'sum' or jk == 'mean' or jk == 'max':
            input_dim = gin.out_channels


        model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                        (gin, 'x, edge_index -> x'),
                        (torch_geometric.nn.pool.global_max_pool, 'x, batch -> x'),
                        (MLP(channel_list=[input_dim] + mlp_hidden_layer_conf + [output_dim], act=activation_func, dropout=dropout, mlp_norm=mlp_norm), 'x -> x'),
                        (torch.nn.Softmax(dim=1), 'x -> x')
                    ])
        
    elif model_name == "GIN:Mean":
        gin = GIN(in_channels=input_dim,
                   hidden_channels=gnn_hidden_channels, 
                   num_layers=gnn_layers, 
                   dropout=dropout, 
                   mlp_norm=mlp_norm, 
                   act=activation_func, 
                   jk=jk)

        # Remove the last linear layer that would otherwise remove all jk information
        if jk == 'cat':
            delattr(gin, 'lin')
            input_dim = gin.out_channels * gin.num_layers
        elif jk == 'last' or jk == 'sum' or jk == 'mean' or jk == 'max':
            input_dim = gin.out_channels

        model = torch_geometric.nn.Sequential('x, edge_index, batch', [
                        (gin, 'x, edge_index -> x'),
                        (torch_geometric.nn.pool.global_mean_pool, 'x, batch -> x'),
                        (MLP(channel_list=[input_dim] + mlp_hidden_layer_conf + [output_dim], act=activation_func, dropout=dropout, mlp_norm=mlp_norm), 'x -> x'),
                        (torch.nn.Softmax(dim=1), 'x -> x')
                    ])

    return model