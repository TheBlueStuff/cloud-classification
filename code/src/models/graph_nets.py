import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import dgl
from dgl.nn import GATConv, GraphConv


class GraphConvGNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.cnn = torch.nn.Sequential(
            *(list(models.resnet50(pretrained=True).children())[:-1])
        )

        self.gnn_layer_1 = GraphConv(2048, 1024)
        self.gnn_layer_2 = GraphConv(1024, 512)

        self.linear_1 = nn.Linear(2048 + 512, 1024)
        self.class_1 = nn.Linear(1024, num_classes)
        self.class_2 = nn.Linear(2048, num_classes)
        
        self.THRESHOLD = 0.7

    def forward(self, x, get_embeddings=False):
        
        deep_features = self.cnn(x).view(-1, 2048)
        g = self.build_graph(deep_features)

        x = F.relu(self.gnn_layer_1(g, deep_features).reshape(-1, 1024))
        x = self.gnn_layer_2(g, x)

        x = torch.cat([deep_features, x], dim=1)
        x = F.leaky_relu(self.linear_1(x))

        logits_1 = self.class_1(x)
        logits_2 = self.class_2(deep_features)

        return logits_1, logits_2
    
    
    def build_graph(self, dfs):
        norm = dfs.norm(dim=1).view(-1,1)
        batch_nodes = dfs/norm

        sim_matrix = batch_nodes @ batch_nodes.T
        adj_matrix = torch.where(sim_matrix > self.THRESHOLD, 1, 0)
        row, col = torch.where(adj_matrix==1)
        
        return dgl.graph((row, col))
    
    
    

class GATConvGNN(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_hidden, num_heads, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.num_heads = num_heads

        self.cnn = torch.nn.Sequential(
            *(list(models.resnet50(pretrained=True).children())[:-1])
        )
        
        
        self.gnn_layer_1 = GATConv(2048, hidden_dim, num_heads=4, residual=True)
        
        self.gnn_stack = nn.ModuleList(
                GATConv(hidden_dim, hidden_dim, num_heads=num_heads, residual=True)
                for i in range(num_hidden-2)
            )
        
        self.gnn_stack.append(GATConv(hidden_dim, hidden_dim, num_heads=1, residual=True))
        

        self.linear_1 = nn.Linear(2048 + hidden_dim, 1024)
        self.class_1 = nn.Linear(1024, num_classes)
        self.class_2 = nn.Linear(2048, num_classes)
        
        self.THRESHOLD = threshold

    def forward(self, x, get_embeddings=False):
        
        deep_features = self.cnn(x).view(-1, 2048)
        g = self.build_graph(deep_features)

        x = F.relu(self.gnn_layer_1(g, deep_features).sum(dim=1))
        
        for i, gnn_layer in enumerate(self.gnn_stack):
            x = gnn_layer(g, x).sum(dim=1)
            if i != len(self.gnn_stack)-1:
                x = F.relu(x)

        x = torch.cat([deep_features, x], dim=1)
        x = F.leaky_relu(self.linear_1(x))

        logits_1 = self.class_1(x)
        logits_2 = self.class_2(deep_features)

        return logits_1, logits_2
    
    
    def build_graph(self, dfs):
        norm = dfs.norm(dim=1).view(-1,1)
        batch_nodes = dfs/norm

        sim_matrix = batch_nodes @ batch_nodes.T
        adj_matrix = torch.where(sim_matrix > self.THRESHOLD, 1, 0)
        row, col = torch.where(adj_matrix==1)
        
        return dgl.graph((row, col))
        