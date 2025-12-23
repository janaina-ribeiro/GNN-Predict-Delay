from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class DelayGNN(nn.Module):
    """
    Class for a Graph Neural Network (GNN) model to predict network delays.
    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int, optional
        Number of hidden units in each GNN layer. Default is 64.
    num_layers : int, optional
        Number of GNN layers. Default is 2.
    num_classes : int, optional
        Number of output classes. Default is 2.
    conv_type : str, optional
        Type of GNN convolution to use: 'gcn', 'gat', or 'sage'. Default is 'gcn'.
    dropout : float, optional
        Dropout rate between layers. Default is 0.2.
    gat_heads : int, optional
        Number of attention heads for GAT convolution. Default is 2.
    Returns
    -------
    torch.nn.Module
        The GNN model.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        conv_type: str = "gcn",
        dropout: float = 0.2,
        gat_heads: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        conv_type = conv_type.lower()
        supported = {"gcn", "gat", "sage"}
        if conv_type not in supported:
            raise ValueError(f"conv_type must be one of {supported}")
        self.conv_type = conv_type
        self.dropout = dropout
        self.gat_heads = gat_heads

        if conv_type == "gat":
            self.actual_hidden = (hidden_channels // gat_heads) * gat_heads
            if self.actual_hidden == 0:
                self.actual_hidden = gat_heads
        else:
            self.actual_hidden = hidden_channels

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            in_c = in_channels if layer == 0 else self.actual_hidden
            out_c = self.actual_hidden
            self.convs.append(self._make_conv(conv_type, in_c, out_c))
        self.classifier = nn.Linear(self.actual_hidden, num_classes)

    def _make_conv(
        self, conv_type: str, in_channels: int, out_channels: int
    ) -> nn.Module:
        if conv_type == "gcn":
            return GCNConv(
                in_channels, out_channels, add_self_loops=True, normalize=True
            )
        if conv_type == "sage":
            return SAGEConv(in_channels, out_channels)

        out_per_head = max(1, out_channels // self.gat_heads)
        return GATConv(
            in_channels,
            out_per_head,
            heads=self.gat_heads,
            concat=True,
            dropout=self.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv in self.convs:
            if isinstance(conv, GCNConv):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        return logits
