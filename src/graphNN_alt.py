import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from config import *
from utils import get_time_stamp, check_and_mkdir, DEBUG, get_newest_checkpoint
from utils_network import check_select_device
from restore_training_data import CompressedHicFragDataSet
from torchviz import make_dot
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.pool import TopKPooling, SAGPooling
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import coalesce, dropout_edge, dropout_node


class Graph_HiC_Likelihood_alt(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=64, hidden_likelihood_dim=128, heads_num=4, num_edge_attr=5, output_dim=4):
        super().__init__()

        self.num_edge_attr = num_edge_attr

        self.conv_attention = GATConv(input_dim, hidden_dim, heads=heads_num)
        self.conv_list = nn.ModuleList([
            GCNConv(hidden_dim * heads_num, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.conv_list_1 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.conv_list_2 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv_list_3 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, hidden_dim) for _ in range(num_edge_attr)
        ])
        self.conv_list_4 = nn.ModuleList([
            GCNConv(hidden_dim * self.num_edge_attr, input_dim) for _ in range(num_edge_attr)
        ])
        
        self.link_likelihood = nn.Linear(hidden_dim * 2 * self.num_edge_attr, hidden_likelihood_dim)
        self.link_likelihood1 = nn.Linear(hidden_likelihood_dim, hidden_likelihood_dim)
        self.link_likelihood2 = nn.Linear(hidden_likelihood_dim, output_dim)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

        self.batchnorm_1 = torch.nn.BatchNorm1d(hidden_dim * heads_num)
        self.batchnorm_2 = torch.nn.BatchNorm1d(hidden_dim * self.num_edge_attr)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, batch:Batch):
        x, edge_index, edge_attr, edge_index_test = batch.x, batch.edge_index, batch.edge_attr, batch.edge_index_test
        x = self.relu(
            self.batchnorm_1(self.conv_attention(x, edge_index))   # [64, 64 * heads_num]   
        )
        x = self.relu(
            self.batchnorm_2(
                torch.concat([self.conv_list[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
            ))
        edge_index, edge_id  = dropout_edge(edge_index, p=0.1, force_undirected=True)
        edge_attr = edge_attr[edge_id, :]
        # x = self.relu(
        #     torch.concat([self.conv_list_1[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        # )
        # x = self.batchnorm_2(x)
        # x = self.relu(
        #     torch.concat([self.conv_list_2[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        # )
        # x = self.batchnorm_2(x)
        x = self.relu(
            torch.concat([self.conv_list_3[i](x, edge_index, edge_weight=edge_attr[:, i]) for i in range(self.num_edge_attr)], dim=-1)   # [64, 64 * self.num_edge_attr]
        )
        x = self.batchnorm_2(x)
        x = self.dropout(x) # [node_num, hidden_dim * self.num_edge_attr]
        x = self.predict_links(x, edge_index_test)  # torch.any(x.isnan())
        return x  #  node embedding
        
    def predict_links(self, x, edge_index):
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_embeddings_1 = torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)
        return 0.5 * (
            self.link_likelihood2(self.relu(self.link_likelihood1(self.relu(self.link_likelihood(edge_embeddings))))).squeeze(-1)+
            self.link_likelihood2(self.relu(self.link_likelihood1(self.relu(self.link_likelihood(edge_embeddings_1))))).squeeze(-1)
        )