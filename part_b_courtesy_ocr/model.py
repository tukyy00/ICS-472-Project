"""CNN + BiLSTM + CTC recognizers for courtesy amount OCR."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def _conv_block(in_channels: int, out_channels: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


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


class Deep16BiLSTMCTC(nn.Module):
    """A 16-convolution CNN followed by BiLSTM + CTC decoding.

    The pooling schedule reduces the crop height aggressively while preserving
    most of the horizontal resolution for CTC time steps.
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        lstm_hidden: int = 512,
        lstm_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        conv_dropout = max(0.0, dropout * 0.5)
        self.cnn = nn.Sequential(
            _conv_block(input_channels, 32),
            _conv_block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x384 -> 32x192
            _conv_block(32, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x192 -> 16x96
            _conv_block(64, 128),
            _conv_block(128, 128),
            _conv_block(128, 128, conv_dropout),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 16x96 -> 8x96
            _conv_block(128, 256),
            _conv_block(256, 256),
            _conv_block(256, 256, conv_dropout),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 8x96 -> 4x96
            _conv_block(256, 384),
            _conv_block(384, 384),
            _conv_block(384, 384, conv_dropout),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 4x96 -> 2x96
            _conv_block(384, 512),
            _conv_block(512, 512),
            _conv_block(512, 512, conv_dropout),
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
        features = self.cnn(images)
        sequence = features.mean(dim=2).permute(2, 0, 1)
        sequence = self.sequence_dropout(sequence)
        sequence, _ = self.lstm(sequence)
        return self.classifier(sequence)


def build_model(
    arch: str = "deep16",
    num_classes: int = 11,
    dropout: float = 0.1,
    lstm_hidden: int = 256,
    lstm_layers: int = 2,
) -> nn.Module:
    normalized = arch.lower()
    if normalized in {"resnet18", "resnet"}:
        return ResNetBiLSTMCTC(
            num_classes=num_classes,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
        )
    if normalized in {"deep16", "cnn16"}:
        return Deep16BiLSTMCTC(
            num_classes=num_classes,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
        )
    raise ValueError(f"Unsupported Part B model architecture: {arch}")

