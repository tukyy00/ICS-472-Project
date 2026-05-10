"""ResNet-style CNN + BiLSTM + CTC recognizer."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ResNetBiLSTMCTC(nn.Module):
    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.cnn = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.sequence_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``T, N, C`` for CTC loss."""
        features = self.cnn(images)  # N, C, H, W
        sequence = features.mean(dim=2).permute(2, 0, 1)  # W, N, C
        sequence = self.sequence_dropout(sequence)
        sequence, _ = self.lstm(sequence)
        return self.classifier(sequence)

