import torch.nn as nn
import torch.nn.functional as F
from GCN_layers import GraphConvolution  # Replace with the actual import
from data_loader import edge_index_to_adjacency
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, data):
        x = data['movie'].x
        adj = edge_index_to_adjacency(data['user', 'rates', 'movie'].edge_index, data['user'].num_nodes + x.size(0))
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
