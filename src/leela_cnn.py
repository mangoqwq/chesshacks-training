# https://lczero.org/dev/backend/nn/
from torch import nn
import torch
from torch import Tensor
import chess
from chess import Board, Move


def board_to_tensor(board: Board) -> Tensor:
    # assert not board.is_game_over(), "Board is in terminal state"
    player_color = board.turn
    if board.turn == chess.BLACK:
        board = board.mirror()
    tensor = torch.zeros((12 + 1 + 1 + 2 + 2, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 6
        row = chess.square_rank(square)
        col = chess.square_file(square)
        tensor[piece_type + color_offset, row, col] = 1.0
        if piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
            tensor[12, row, col] = 1.0
    if player_color == chess.WHITE:
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[15, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[17, :, :] = 1.0
    return tensor


def move_to_index(move: Move) -> int:
    from_square = move.from_square
    to_square = move.to_square
    from_row = chess.square_rank(from_square)
    from_col = chess.square_file(from_square)
    to_row = chess.square_rank(to_square)
    to_col = chess.square_file(to_square)
    if {abs(from_row - to_row), abs(from_col - to_col)} == {2, 1}:
        # knight moves
        diff_to_dir = {
            (2, 1): 0,
            (2, -1): 1,
            (-2, 1): 2,
            (-2, -1): 3,
            (1, 2): 4,
            (1, -2): 5,
            (-1, 2): 6,
            (-1, -2): 7,
        }
        direction = diff_to_dir[(to_row - from_row, to_col - from_col)]
    else:
        if from_row == to_row:
            # horizontal moves
            direction = 8 + to_col
        elif from_col == to_col:
            # vertical moves
            direction = 16 + to_row
        elif to_row - to_col == from_row - from_col:
            direction = 24 + to_row
        else:
            direction = 32 + to_row
    index = direction * 40 + from_row * 8 + from_col
    return index

def board_to_legal_mask(board: Board) -> Tensor:
    mask = torch.zeros(( (8 * 4 + 8) * 8 * 8,), dtype=torch.bool)
    for move in board.legal_moves:
        if board.turn == chess.BLACK:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
        index = move_to_index(move)
        mask[index] = 1
    return mask

def policy_dict_to_policy(policy_dict: dict[Move, float], board: Board) -> Tensor:
    policy = torch.zeros(( (8 * 4 + 8) * 8 * 8,), dtype=torch.float32)
    for move, prob in policy_dict.items():
        if board.turn == chess.BLACK:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
        index = move_to_index(move)
        policy[index] = float(prob)
    return policy

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
        scale, shift = torch.chunk(self.layers(x), 2, dim=-1)
        scale = self.sigmoid(scale)
        return x * scale[:, :, None, None] + shift[:, :, None, None]


class LeelaCNN(nn.Module):
    # Input: ((6 + 6) + 1 + 1 + 2 + 2) * 8 * 8
    #        ((self pieces + opponent pieces) + opponent pawn previous + colour + self castling right (king and queen side) + opponent castling right (king and queen side)) * board
    # Output: Policy, Value
    # Policy: (8 * 4 + 8) * 8 * 8
    #         (queen moves + knight moves) * board (starting position)
    # Value : Single value in (-1, 1)

    # Common Block Counts x Filter Counts: 10×128, 20×256, 24×320.

    def __init__(self, block_count: int, filter_count: int, se_channels: int = 32):
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
            nn.Linear(80 * 8 * 8, (8 * 4 + 8) * 8 * 8),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(filter_count, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=8, stride=1, padding=0),
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
