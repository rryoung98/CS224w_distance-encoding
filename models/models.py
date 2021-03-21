# from models.layers import *
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, TAGConv, GATConv
from models.mlp import MLP



"""
Things to do:

Determine apppropriate num_groups
Understand necessity of skip_connect

"""
class batch_norm(torch.nn.Module):
    def __init__(self, hidden_features, type_norm, skip_connect=False, num_groups=2,
                 skip_weight=0.005):
        super(batch_norm, self).__init__()
        self.type_norm = type_norm
        self.skip_connect = skip_connect
        self.num_groups = num_groups
        self.skip_weight = skip_weight
        self.hidden_features= hidden_features
        if self.type_norm == 'batch':
            self.bn = torch.nn.BatchNorm1d(hidden_features, momentum=0.3)
        elif self.type_norm == 'group':
            self.bn = torch.nn.BatchNorm1d(hidden_features*self.num_groups, momentum=0.3)
            self.group_func = torch.nn.Linear(hidden_features, self.num_groups, bias=True)
        else:
            pass

    def forward(self, x):
        if self.type_norm == 'None':
            return x
        elif self.type_norm == 'batch':
            # print(self.bn.running_mean.size())
            return self.bn(x)
        elif self.type_norm == 'group':
            if self.num_groups == 1:
                x_temp = self.bn(x)
            else:
                score_cluster = F.softmax(self.group_func(x), dim=1)
                x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)], dim=1)
                x_temp = self.bn(x_temp).view(-1, self.num_groups, self.hidden_features).sum(dim=1)
            x = x + x_temp * self.skip_weight
            return x

        else:
            raise Exception(f'the normalization has not been implemented')

class GNNModel(nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features, prop_depth, dropout=0.0, model_name='DE-GNN',type_norm='group'):
        super(GNNModel, self).__init__()
        self.num_layers, self.in_features, self.hidden_features, self.out_features, self.model_name = num_layers, in_features, hidden_features, out_features, model_name
        Layer = self.get_layer_class()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.type_norm = type_norm
        if self.model_name == 'DE-GNN':
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features, K=prop_depth))
        elif self.model_name == 'GIN':
            self.layers.append(
                Layer(MLP(num_layers=2, input_dim=in_features, hidden_dim=hidden_features, output_dim=hidden_features)))
        else:
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features))
        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                if self.model_name == 'DE-GNN':
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features, K=prop_depth))
                elif self.model_name == 'GIN':
                    self.layers.append(Layer(MLP(num_layers=2, input_dim=hidden_features, hidden_dim=hidden_features,
                                                 output_dim=hidden_features)))
                elif self.model_name == 'GAT':
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features, heads=8))
                else:
                    # for GCN and GraphSAGE
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features))
        # we're  building up the normalization layers here.
        if self.type_norm == 'group':
            self.layer_norms = nn.ModuleList([batch_norm(hidden_features,'group',skip_connect = True) for i in range(self.num_layers)])
        else:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for i in range(self.num_layers)])
        self.merger = nn.Linear(3 * hidden_features, hidden_features)
        self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        for i, layer in enumerate(self.layers):
            # edge_weight = None
            # # if not layer.normalize:
            # #     edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
            x = layer(x, edge_index, edge_weight=None)
            x = self.act(x)
            x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
            if self.model_name == 'DE-GNN':
                x = self.layer_norms[i](x)
        x = self.get_minibatch_embeddings(x, batch)
        x = self.feed_forward(x)
        return x

    def get_minibatch_embeddings(self, x, batch):
        device = x.device
        set_indices, batch, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch].to(device).sum(dim=0)
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        assert(index_bases.size(0) == set_indices.size(0))
        set_indices_batch = index_bases + set_indices
        # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
        x = x[set_indices_batch]  # shape [B, set_size, F]
        x = self.pool(x)
        return x

    def pool(self, x):
        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)
        # use mean/diff/max to pool each set's representations
        x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        for i, j in combinations(range(x.size(1)), 2):
            x_diff += torch.abs(x[:, i, :]-x[:, j, :])
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
        return x

    def get_layer_class(self):
        layer_dict = {'DE-GNN': TAGConv, 'GIN': GINConv, 'GCN': GCNConv, 'GraphSAGE': SAGEConv, 'GAT': GATConv}  # TAGConv essentially sums up GCN layerwise outputs, can use GCN instead 
        Layer = layer_dict.get(self.model_name)
        if Layer is None:
            raise NotImplementedError('Unknown model name: {}'.format(self.model_name))
        return Layer

    def short_summary(self):
        return 'Model: {}, #layers: {}, in_features: {}, hidden_features: {}, out_features: {}'.format(self.model_name, self.layers, self.in_features, self.hidden_features, self.out_features)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x
