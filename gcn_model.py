import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, num_classes)

    def forward(self, X, adj):
        # A * X
        h = torch.spmm(adj, X)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)

        h = torch.spmm(adj, h)
        h = self.fc2(h)
        return h
