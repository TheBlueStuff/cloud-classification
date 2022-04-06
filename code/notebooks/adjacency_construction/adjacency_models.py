import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import dgl
from dgl.nn import GATConv, GraphConv


def normalize_features(dfs):
    norm = dfs.norm(dim=1).view(-1,1)
    return dfs/norm

def build_graph_cosine_similarity(dfs, threshold):
    batch_nodes = normalize_features(dfs)

    sim_matrix = batch_nodes @ batch_nodes.T
    adj_matrix = torch.where(sim_matrix > threshold, 1, 0)
        
    row, col = torch.where(adj_matrix==1)

    return dgl.graph((row, col))

def build_graph_pearson_similarity(dfs, threshold):
    corr_matrix = torch.corrcoef(dfs)
    
    adj_matrix = torch.where(corr_matrix > threshold, 1, 0)
    row, col = torch.where(adj_matrix==1)
    
    return dgl.graph((row, col))


class MLPBuilder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x):
        
        # All against all concat
        
        left = x.repeat_interleave(x.shape[0], dim=0)
        right = x.repeat(x.shape[0],1)
                
        cat_tensor = torch.cat([left,right], dim=1)
        
        out = self.mlp(cat_tensor)
        
        out = F.softmax(out, dim=1)
        out = torch.argmax(out, dim=1)
    
        adj_matrix = out.view(x.shape[0], x.shape[0]) #NxN
        adj_matrix.fill_diagonal_(1)
        
        return adj_matrix
        
    def build_graph(self, x):
        adj_matrix = self.forward(x)
        row, col = torch.where(adj_matrix==1)
    
        return dgl.graph((row, col))
    
    
    
class AttentionBuilder(nn.Module):
    """
    https://github.com/pbloem/former/blob/master/former/modules.py
    """
    def __init__(self, in_dims, hidden, num_heads=4):
        super().__init__()
        
        self.hidden = hidden
        self.num_heads = num_heads
        
        self.q = nn.Linear(in_dims, num_heads*hidden, bias=False)
        self.k = nn.Linear(in_dims, num_heads*hidden, bias=False)
        self.v = nn.Linear(in_dims, num_heads*hidden, bias=False)
        
        self.unify_heads = nn.Linear(num_heads*hidden, hidden)
        
    def forward(self, x):
        t = x.shape[0]
        h = self.hidden
        heads= self.num_heads
        
        queries = self.q(x).view(t, heads, h)
        keys = self.k(x).view(t, heads, h)
        values = self.v(x).view(t, heads, h)
        
        # Fold heads
        queries = queries.view(heads,t,h)
        keys = keys.view(heads,t,h)
        values = values.view(heads,t,h)
        
        dot = torch.bmm(queries, keys.transpose(1,2))/torch.sqrt(h) #hxtxt
        dot = F.softmax(dot, dim=2) #self attention probs
        
        out = torch.bmm(dot, values).view(t,heads*h)
        
        return self.unify_heads(out)     
        
    def build_graph(self, x, threshold):
        return build_graph_cosine_similarity(x, threshold)
    


class GATConvGNN(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_hidden, num_heads, threshold, adjacency_builder = 'cos_sim'):
        super().__init__()

        self.adjacency_builder = adjacency_builder
        
        if adjacency_builder == 'mlp':
            self.builder = MLPBuilder(2*2048, 512)
        elif adjacency_builder == 'attention':
            self.builder = AttentionBuilder(2048, 512, 4)
        
        
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

    def forward(self, x):
        
        deep_features = self.cnn(x).view(-1, 2048)
        
        ### ADJACENCY MATRIX CONSTRUCTION
        if self.adjacency_builder == 'cos_sim':
            g = build_graph_cosine_similarity(deep_features, self.THRESHOLD)
        elif self.adjacency_builder == 'pearson_corr':
            g = build_graph_pearson_similarity(deep_features, self.THRESHOLD)
        elif self.adjacency_builder == 'mlp':
            g = self.builder.build_graph(deep_features)
        elif self.adjacency_builder == 'attention':
            g = self.builder.build_graph(deep_features, self.THRESHOLD)
        else:
            raise NotImplementedError("Invalid builder")
    
    
        ### MESSAGE PASSING
        x = F.relu(self.gnn_layer_1(g, deep_features).sum(dim=1))
        
        for gnn_layer in self.gnn_stack[:-1]:
            x = F.relu(gnn_layer(g, x).sum(dim=1))
        
        x = self.gnn_stack[-1](g, x).squeeze(1)

        x = torch.cat([deep_features, x], dim=1)
        x = F.leaky_relu(self.linear_1(x))

        logits_1 = self.class_1(x)
        logits_2 = self.class_2(deep_features)

        return logits_1, logits_2
    