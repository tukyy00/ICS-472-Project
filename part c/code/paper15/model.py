"""Paper15 CNN-BiLSTM-CTC model only."""

from __future__ import annotations

import torch
from torch import nn


class Paper15CRNN(nn.Module):
    def __init__(self, num_classes: int, lstm_hidden: int = 384, lstm_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_block(1, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv_block(64, 128),
            _conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv_block(128, 256),
            _conv_block(256, 256),
            _conv_block(256, 256),
            _conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv_block(256, 512),
            _conv_block(512, 512),
            _conv_block(512, 512),
            _conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv_block(512, 512),
            _conv_block(512, 512),
            _conv_block(512, 512),
            nn.AdaptiveAvgPool2d((1, 96)),
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images).squeeze(2)
        sequence = self.dropout(features.permute(2, 0, 1))
        sequence, _ = self.lstm(sequence)
        return self.classifier(sequence)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

