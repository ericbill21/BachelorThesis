import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import wl_conv
import matplotlib.pyplot as plt
from matplotlib import colors
from visualization import plot_loss

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

def main2():
    all_train_losses = {'sum': [-0.5468073625034756, -0.6351711050669352, -0.6904754702250163, -0.6852498830689324, -0.7070234841770596, -0.735215020444658, -0.7307437414593166, -0.7316404493649801, -0.732372088432312, -0.749060090912713, -0.7212119640244378, -0.7509085117446052, -0.7484979671902127, -0.7641528243488735, -0.7622791715463002, -0.7618747451570299, -0.7623144022623698, -0.7703672295146519, -0.7428427984979418], 'max': [-0.5557216180695428, -0.6170147879918416, -0.6455222427182727, -0.6694237494468689, -0.6890527031156752, -0.6822997552818723, -0.6925642593701681, -0.6874928760528565, -0.6950672319200304, -0.6787794378068712, -0.7151878152953254, -0.7294376267327203, -0.7260124945640564, -0.740850223965115, -0.7372573334640927, -0.729334455066257, -0.7282889509201049, -0.7375153777334426, -0.7396014147334629], 'mean': [-0.5872294388877021, -0.6696430865923564, -0.6939359421200223, -0.6966966830359564, -0.7303071800867716, -0.7152869831191169, -0.7365544663535224, -0.7312921063105265, -0.7316360772980584, -0.7360931735568577, -0.7443746389283075, -0.7432899022102356, -0.7076673581865098, -0.7064008829328748, -0.7463080848587884, -0.7468940199746026, -0.7380526126755609, -0.7247209050920275, -0.7245409647623698], 'gin': [-0.5341731990708245, -0.6328041399849785, -0.6643079378869798, -0.6673368748029073, -0.6763436500231425, -0.6782207292980618, -0.6605296762784322, -0.6843764201800029, -0.6852280677689446, -0.6767232686943478, -0.6755173087120057, -0.6748655237091912, -0.6875040794743432, -0.7000292507807414, -0.6855047292179531, -0.6975006413459778, -0.6782192394468519, -0.6795780293146769, -0.6976980336507161]}
    all_val_losses = {'sum': [-0.024215864241123198, -0.026777576804161072, -0.02756079852581024, -0.02544563353061676, -0.027187696695327758, -0.025436868071556092, -0.025886765718460082, -0.029074646830558777, -0.02909631609916687, -0.028066438734531404, -0.026936654448509217, -0.027278915345668793, -0.027130922079086305, -0.028003169298171995, -0.025023180842399597, -0.024888998866081237, -0.026403093338012697, -0.02787726402282715, -0.02430107057094574], 'max': [-0.01934876322746277, -0.02147516518831253, -0.022233429551124572, -0.02208070993423462, -0.02219173848628998, -0.022141787707805633, -0.022481976151466368, -0.023652947545051574, -0.019493204057216645, -0.020963580906391145, -0.023554641008377075, -0.024003230929374696, -0.023703970909118653, -0.023586073815822603, -0.024966351985931397, -0.02287929505109787, -0.023367741107940675, -0.02323045313358307, -0.022947918474674225], 'mean': [-0.026027539372444154, -0.027229883670806886, -0.027538172006607055, -0.026430086493492128, -0.025992285013198852, -0.024249302744865416, -0.026373622417449952, -0.025888838768005372, -0.02583968698978424, -0.027864712476730346, -0.02642217218875885, -0.02543590545654297, -0.025299429297447204, -0.02498704731464386, -0.02683103919029236, -0.026694164276123047, -0.026746106743812562, -0.02644378423690796, -0.023804482817649842], 'gin': [-0.02108349621295929, -0.02780924439430237, -0.02739986091852188, -0.028365805745124817, -0.02674174278974533, -0.028085376024246215, -0.02759762465953827, -0.028804957568645477, -0.028852410316467285, -0.02951201021671295, -0.028653459250926973, -0.02913817286491394, -0.02735539376735687, -0.02951569139957428, -0.026796218752861024, -0.027781317830085753, -0.028383585214614867, -0.029585680961608886, -0.02957548141479492]}
    all_test_accuracies = {'sum': [0.57, 0.59, 0.62, 0.61, 0.64, 0.65, 0.62, 0.65, 0.66, 0.63, 0.65, 0.59, 0.65, 0.61, 0.64, 0.61, 0.6, 0.61, 0.59], 'max': [0.53, 0.55, 0.59, 0.59, 0.58, 0.58, 0.58, 0.63, 0.55, 0.55, 0.61, 0.62, 0.62, 0.62, 0.59, 0.59, 0.61, 0.61, 0.58], 'mean': [0.59, 0.6, 0.64, 0.64, 0.59, 0.55, 0.58, 0.61, 0.61, 0.62, 0.61, 0.59, 0.6, 0.65, 0.64, 0.64, 0.64, 0.64, 0.6], 'gin': [0.56, 0.7, 0.6, 0.64, 0.58, 0.65, 0.63, 0.63, 0.62, 0.66, 0.63, 0.66, 0.64, 0.66, 0.63, 0.66, 0.64, 0.72, 0.73]}

    plot_loss(all_train_losses, all_test_accuracies, all_val_losses, all_test_accuracies)


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
    main2()
