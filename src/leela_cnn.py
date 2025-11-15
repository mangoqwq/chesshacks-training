# https://lczero.org/dev/backend/nn/
from torch import nn
import torch
from torch import Tensor
import chess
from chess import Board, Move

# Input: ((6 + 6) + 1 + 1 + 2 + 2) * 8 * 8
#        ((self pieces + opponent pieces) + opponent pawn previous + colour + self castling right (king and queen side) + opponent castling right (king and queen side)) * board
# Output: Policy, Value
# Policy: (8 * 4 + 8) * 8 * 8
#         (queen moves + knight moves) * board (starting position)
# Value : Single value in [-1, 1]

def board_to_tensor(board: Board) -> Tensor:
    assert not board.is_game_over(), "Board is in terminal state"
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
    if chess.square_distance(from_square, to_square) == 1 and (
        from_row == to_row or from_col == to_col
    ):
        direction = 0  # up, down, left, right
    elif chess.square_distance(from_square, to_square) == 2 and (
        abs(from_row - to_row) == 2 or abs(from_col - to_col) == 2
    ):
        direction = 1  # knight moves
    else:
        direction = 2 + (to_row - from_row + 1) * 3 + (to_col - from_col + 1)
    index = direction * 64 + from_row * 8 + from_col
    return index

def index_to_move(index: int) -> Move:
    direction = index // 64
    from_square_index = index % 64
    from_row = from_square_index // 8
    from_col = from_square_index % 8
    if direction == 0:
        to_row, to_col = from_row, from_col + 1  # example: right move
    elif direction == 1:
        to_row, to_col = from_row + 2, from_col + 1  # example: knight move
    else:
        dir_index = direction - 2
        delta_row = (dir_index // 3) - 1
        delta_col = (dir_index % 3) - 1
        to_row = from_row + delta_row
        to_col = from_col + delta_col
    from_square = chess.square(from_col, from_row)
    to_square = chess.square(to_col, to_row)
    return Move(from_square, to_square)

def moves_to_mask(tensor: Tensor) -> Board:
    board = Board.empty()
    piece_map = {}
    for row in range(8):
        for col in range(8):
            for piece_type in range(6):
                if tensor[piece_type, row, col] == 1.0:
                    square = chess.square(col, row)
                    piece_map[square] = chess.Piece(piece_type + 1, chess.WHITE)
                if tensor[piece_type + 6, row, col] == 1.0:
                    square = chess.square(col, row)
                    piece_map[square] = chess.Piece(piece_type + 1, chess.BLACK)
    board.set_piece_map(piece_map)
    if tensor[13, 0, 0] == 1.0:
        player_color = chess.WHITE
    else:
        player_color = chess.BLACK
    if tensor[14, 0, 0] == 1.0:
        board.castling_rights |= chess.BB_H1
    if tensor[15, 0, 0] == 1.0:
        board.castling_rights |= chess.BB_A1
    if tensor[16, 0, 0] == 1.0:
        board.castling_rights |= chess.BB_H8
    if tensor[17, 0, 0] == 1.0:
        board.castling_rights |= chess.BB_A8
    if player_color == chess.BLACK:
        board = board.mirror()
    return board


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
            nn.Linear(80 * 8 * 8, (8 * 4 + 8) * 8 * 8),
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
