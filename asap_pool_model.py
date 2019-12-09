import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_scatter import scatter_mean, scatter_max
from asap_pool import ASAP_Pooling
import pdb

def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0) 
    return torch.cat((x_mean, x_max), dim=-1)

class ASAP_Pool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, **kwargs):
        super(ASAP_Pool, self).__init__()
        if type(ratio)!=list:
            ratio = [ratio for i in range(num_layers)]
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.pool1 = ASAP_Pooling(in_channels=hidden, ratio=ratio[0], **kwargs)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.pools.append(ASAP_Pooling(in_channels=hidden, ratio=ratio[i], **kwargs))
        self.lin1 = Linear(2*hidden, hidden) # 2*hidden due to readout layer
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_weight, batch, perm = self.pool1(x=x, edge_index=edge_index, edge_weight=None, batch=batch)
        xs = readout(x, batch)
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
            x, edge_index, edge_weight, batch, perm = pool(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
            xs += readout(x, batch)
        x = F.relu(self.lin1(xs))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        out = F.log_softmax(x, dim=-1)
        return out

    def __repr__(self):
        return self.__class__.__name__
