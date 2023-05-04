import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import wl_conv

def main():
    edge_index = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]])
    x = torch.zeros(5, dtype=torch.long).unsqueeze(-1)

    g1 = Data(x=x, edge_index=edge_index)

    g2 = g1.clone()

    wl = wl_conv.WLConv()


    old_coloring = g1.x.squeeze()

    is_converged = False
    iteration = 0
    while not is_converged and iteration < g1.num_nodes:
        new_coloring = wl.forward(old_coloring, g2.edge_index)

        is_converged = check_convergence(old_coloring, new_coloring)
        old_coloring = new_coloring

        iteration += 1



    print('hey')


def check_convergence(old, new):
    hashmap = {}

    for i in new:
        if i.item() not in hashmap:
            hashmap[i.item()] = 1
        else:
            hashmap[i.item()] += 1

    values_new = list(hashmap.values())
    values_new.sort()
    
    hashmap = {}

    for i in old:
        if i.item() not in hashmap:
            hashmap[i.item()] = 1
        else:
            hashmap[i.item()] += 1

    values_old = list(hashmap.values())
    values_old.sort()
    
    return values_new == values_old

if __name__ == '__main__':
    main()

