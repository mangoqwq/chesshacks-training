# https://lczero.org/dev/backend/nn/
from torch import nn
import torch


class SEBlock(nn.Module):
    def __init__(self, channels, se_channels):
        super(SEBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(channels, se_channels),
            nn.ReLU(),
            nn.Linear(se_channels, 2 * channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale, shift = torch.chunk(self.layers(x), 2)
        scale = self.sigmoid(scale)
        return x * scale[:, None, None] + shift[:, None, None]


class LeelaCNN(nn.Module):
    # Input: ((6 + 6) + 1 + 1 + 2 + 2) * 8 * 8
    #        ((self pieces + opponent pieces) + opponent pawn previous + colour + self castling right (king and queen side) + opponent castling right (king and queen side)) * board
    # Output: Policy, Value
    # Policy: (7 * 4 + 8) * 8 * 8
    #         (queen moves + knight moves) * board (starting position)
    # Value : Single value in (-1, 1)

    # Common Block Counts x Filter Counts: 10×128, 20×256, 24×320.

    def __init__(self, block_count, filter_count, se_channels=32):
        super(LeelaCNN, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(
                (6 + 6) + 1 + 1 + 2 + 2,
                filter_count,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(filter_count),
        )

        blocks = []
        activation = []
        for i in range(block_count):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        filter_count, filter_count, kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(filter_count),
                    nn.Conv2d(
                        filter_count, filter_count, kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(filter_count),
                    SEBlock(filter_count, se_channels),
                )
            )
            activation.append(nn.ReLU())

        self.blocks = nn.ModuleList(blocks)
        self.activations = nn.ModuleList(activation)

        self.policy_head = nn.Sequential(
            nn.Conv2d(filter_count, filter_count, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_count),
            nn.Conv2d(filter_count, 80, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(80),
            nn.Flatten(),
            nn.Linear(80 * 8 * 8, (7 * 4 + 8) * 8 * 8),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(filter_count, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.input(x)
        for block, activation in zip(self.blocks, self.activations):
            block_out = block(x)
            x = x + block_out
            x = activation(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value
