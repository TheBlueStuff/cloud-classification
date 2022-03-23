import numpy as np
import networkx as nx
import dgl
from dgl.nn import GATConv
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import config


class GATClassifier(nn.Module):
    def __init__(self, in_channels, emb_dims, in_dims, out_dims, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.emb_dims = emb_dims
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_classes = num_classes

        self.adj_out_dims = 256

        self.cnn = torch.nn.Sequential(
            *(list(models.resnet50(pretrained=True).children())[:-1])
        )

        self.gatconv1 = GATConv(2048, 1024, num_heads=1)
        self.gatconv2 = GATConv(1024, 512, num_heads=2)

        self.lin1 = nn.Linear(2048 + out_dims, 1024)
        self.class1 = nn.Linear(1024, num_classes)
        self.class2 = nn.Linear(2048, num_classes)

    def forward(self, x, get_embeddings=False):
        embeddings = self.cnn(x)  # bs, deep_features_h
        deep_features = embeddings.reshape(-1, self.emb_dims)
        dfs = deep_features.clone()
        g = self.build_graph(dfs)

        x = F.relu(self.gatconv1(g, dfs).reshape(-1, 1024))
        x = self.gatconv2(g, x).sum(dim=1)

        x = torch.cat((deep_features, x), dim=1)
        x = F.leaky_relu(self.lin1(x))

        if get_embeddings:
            return x

        logits1 = self.class1(x)
        logits2 = self.class2(deep_features)

        return logits1, logits2

    def build_graph(self, deep_features):
        dfs = deep_features.clone().detach()
        z_norm = torch.linalg.norm(dfs, dim=1, keepdim=True)  # Size (n, 1).
        similarity = ((dfs @ dfs.T) / (z_norm @ z_norm.T)).T

        adjacency = torch.where(similarity > config.THRESHOLD, 1, 0)

        gx = nx.from_numpy_array(adjacency.cpu().numpy())
        g = dgl.from_networkx(gx).to(config.DEVICE)
        g = dgl.add_self_loop(g)
        return g

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    gat = GATClassifier(
        in_channels=3,
        emb_dims=2048,  # Deep features
        in_dims=256,
        out_dims=512,  # Graph features
        num_classes=7,
    ).to(config.DEVICE)
    img = torch.randn(5, 3, 256, 256).to(config.DEVICE)
    print(gat(img).shape)
