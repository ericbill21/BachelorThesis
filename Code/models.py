import torch
import torch_geometric
from torch_geometric.nn.models import GIN, MLP

def load_model(model_name, output_dim, is_classification, device, largest_color = None, embedding_dim = None, mlp_hidden_layer_conf = []):

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
                                    (MLP(channel_list=[embedding_dim] + mlp_hidden_layer_conf + [output_dim]), 'x -> x'),
                                    (torch.nn.Softmax(dim=1), 'x -> x')
                                ]).to(device)
            
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
                                    (MLP(channel_list=[embedding_dim] + mlp_hidden_layer_conf + [output_dim]), 'x -> x'),
                                    (torch.nn.Softmax(dim=1), 'x -> x')
                                ]).to(device)
            
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
                                    (MLP(channel_list=[embedding_dim] + mlp_hidden_layer_conf + [output_dim]), 'x -> x'),
                                    (torch.nn.Softmax(dim=1), 'x -> x')
                                ]).to(device)
        
        else:
            raise ValueError("1WL+NN:Embedding-Max is not implemented for regression tasks")


    return model