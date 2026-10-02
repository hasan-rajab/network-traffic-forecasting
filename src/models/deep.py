from __future__ import annotations

import torch
from torch import nn


class TemporalCNN(nn.Module):
    def __init__(self, n_cells: int, channels: int = 12, embedding_dim: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.cell_embedding = nn.Embedding(n_cells, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(channels + embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, cell_idx):
        z = self.conv(x.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([z, self.cell_embedding(cell_idx)], dim=1)).squeeze(1)
