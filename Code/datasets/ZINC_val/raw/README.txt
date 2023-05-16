README for dataset ZINC_val

=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

There are OPTIONAL files if the respective information is available:

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i

=== Description of the dataset === 

The node labels are atom types and the edge labels atom bond types.

Node labels:

'C': 0
'O': 1
'N': 2
'F': 3
'C H1': 4
'S': 5
'Cl': 6
'O -': 7
'N H1 +': 8
'Br': 9
'N H3 +': 10
'N H2 +': 11
'N +': 12
'N -': 13
'S -': 14
'I': 15
'P': 16
'O H1 +': 17
'N H1 -': 18
'O +': 19
'S +': 20
'P H1': 21
'P H2': 22
'C H2 -': 23
'P +': 24
'S H1 +': 25
'C H1 -': 26
'P H1 +': 27

Edge labels:

'SINGLE': 1
'DOUBLE': 2
'TRIPLE': 3

=== Source ===

https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_93.pdf
@article{bresson2019two,
title={A Two-Step Graph Convolutional Decoder for Molecule Generation},
author={Bresson, Xavier and Laurent, Thomas},
journal={ Workshop on Machine Learning and the Physical Sciences (NeurIPS 2019),
Vancouver, Canada, arXiv preprint arXiv:1906.03412},
year={2019}
}

The chemical property used is y(m) = logP(m) − SA(m) − cycle(m) taken from
this paper, section 3.2:
https://arxiv.org/pdf/1802.04364.pdf
@article{jin2018junction,
title={Junction tree variational autoencoder for molecular graph generation},
author={Jin, Wengong and Barzilay, Regina and Jaakkola, Tommi},
journal={arXiv preprint arXiv:1802.04364},
year={2018}
}
