import torch
from torch_geometric.nn.conv import wl_conv

class WLNN():

    def __init__(self, f_enc = None, mlp = None) -> None:
        self.wl = wl_conv.WLConv()
        self.encoding = f_enc
        self.mlp = mlp

        self.train_dataset = None
        self.test_dataset = None

    def set_encoding(self, f_enc):
        self.encoding = f_enc

    def set_mlp(self, mlp):
        self.mlp = mlp

    def get_total_number_of_colors(self):
        return len(self.wl.hashmap)
    
    
    
    def init_training(self, train_dataset, test_dataset, encode_dim):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.wl.reset_parameters()

        self.new_dataset = torch.zeros((train_dataset.len() + test_dataset.len(), encode_dim), dtype=torch.long)

        count = 0
        for graph in train_dataset + test_dataset:

            coloring = wl_algorithm(graph)

            # We encode the node features
            self.new_dataset[count] = self.encoding(coloring)
            count += 1
            
        
    def forward(self, x, edge_index, batch_size, batch_ptr):
        # We run the wl kernel on the graph
        wl_out = x.squeeze()
        for i in range(max(batch_ptr)):
            wl_out = self.wl.forward(x.squeeze(), edge_index)

        # We encode the node features
        # TODO: Adapt to the other encoding functions
        encoded_out = torch.zeros(batch_size, self.get_total_number_of_colors())

        for i in range(batch_size):
            encoded_out[i] = self.encoding(wl_out[batch_ptr[i]:batch_ptr[i+1]])

        # We run the MLP on the encoded node features
        out = self.mlp(encoded_out)

        return out
    

def constant_and_id_transformer(data):
    if data.x is None:
        data.x = torch.zeros(data.edge_index.shape[1], dtype=torch.long).unsqueeze(-1)
    
    return data

def wl_algorithm(wl, graph, total_iterations = -1):
        old_coloring = graph.x.squeeze()
        new_coloring = wl.forward(old_coloring, graph.edge_index)

        iteration = 0
        is_converged = (torch.sort(wl.histogram(old_coloring))[0] == torch.sort(wl.histogram(new_coloring))[0]).all()
        while not is_converged and iteration < total_iterations:
            # Calculate the new coloring
            old_coloring = new_coloring
            new_coloring = wl.forward(old_coloring, graph.edge_index)

            # Check if the coloring has converged
            iteration += 1
            is_converged = (torch.sort(wl.histogram(old_coloring))[0] == torch.sort(wl.histogram(new_coloring))[0]).all()

        return old_coloring


    
        