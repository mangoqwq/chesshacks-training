from dataclasses import dataclass
from pathlib import Path
import numpy as np
import chess
import torch
from torch.utils.data import IterableDataset
from .policy_index import policy_index as POLICY_INDEX
from .new_data_pipeline import data_generator, multiprocess_generator, file_generator
from .leela_cnn import board_to_tensor, policy_dict_to_policy, board_to_legal_mask

policy_index = np.array(POLICY_INDEX)


def lc0_planes_to_board(
    planes: np.ndarray, castle_rights: np.ndarray, player_plane: np.ndarray
) -> chess.Board:
    board = chess.Board.empty()
    piece_map = board.piece_map()
    piece_planes = planes[:12]
    for i in range(64):
        square = chess.SQUARES[i]
        for piece_type in range(6):
            if piece_planes[piece_type][i]:
                piece_map[square] = chess.Piece(piece_type + 1, chess.WHITE)
            elif piece_planes[piece_type + 6][i]:
                piece_map[square] = chess.Piece(piece_type + 1, chess.BLACK)
    board.set_piece_map(piece_map)
    board.castling_rights = 0
    if castle_rights[0, 0]:
        board.castling_rights |= chess.BB_H1
    if castle_rights[1, 0]:
        board.castling_rights |= chess.BB_A1
    if castle_rights[2, 0]:
        board.castling_rights |= chess.BB_H8
    if castle_rights[3, 0]:
        board.castling_rights |= chess.BB_A8
    if player_plane[0, 0] == 1:
        board = board.mirror()
    return board


def lc0_policy_to_moves(
    policy: np.ndarray, player_plane: np.ndarray
) -> dict[chess.Move, float]:
    mask = policy >= 0
    moves = policy_index[mask]
    probs = policy[mask]
    move_dict = {}
    black = player_plane[0, 0] == 1
    for m, p in zip(moves, probs):
        move = chess.Move.from_uci(m)
        if black:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
        if move.promotion is not None and move.promotion != chess.QUEEN:
            move.promotion = chess.QUEEN
        if move not in move_dict:
            move_dict[move] = 0.0
        move_dict[move] += p
    return move_dict


def lc0_q_to_value(q: np.ndarray, player_plane: np.ndarray) -> float:
    black = player_plane[0, 0] == 1
    if black:
        return q[2] - q[0]
    return q[0] - q[2]


@dataclass
class DataPoint:
    board: chess.Board
    policy: dict[chess.Move, float]
    value: float


def lc0_convert(
    lc0_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> list[DataPoint]:
    inputs, policy, z, q, ply_count = lc0_data
    batch_size = inputs.shape[0]
    data: list[DataPoint] = []
    for i in range(batch_size):
        planes_i = inputs[i, 0:12]
        policy_i = policy[i]
        q_i = q[i]
        castle_rights = inputs[i, -8:-4]
        player_plane = inputs[i, -4:-3]
        data.append(
            DataPoint(
                board=lc0_planes_to_board(planes_i, castle_rights, player_plane),
                policy=lc0_policy_to_moves(policy_i, player_plane),
                value=lc0_q_to_value(q_i, player_plane),
            )
        )
    return data


class Lc0Loader:
    def __init__(
        self,
        chunk_dir: Path,
        batch_size: int = 1024,
        num_workers: int = 16,
        shuffle_buffer_factor: int = 128,
        skip_factor: int = 32,
        validation: bool = False,
    ):
        # Store parameters instead of generator to make it pickleable
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_factor * batch_size
        self.skip_factor = skip_factor
        self.validation = validation

    def __iter__(self):
        # Create generator in __iter__ so each worker process gets its own generator
        gen = multiprocess_generator(
            chunk_dir=self.chunk_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_buffer_size=self.shuffle_buffer_size,
            skip_factor=self.skip_factor,
            validation=self.validation,
        )
        for lc0_data in gen:
            print(lc0_data)
            data_points = lc0_convert(lc0_data) # type: ignore
            print("!", len(data_points))
            for dp in data_points:
                yield dp


class Lc0TeacherDataset(IterableDataset):
    def __init__(self, lc0_loader: Lc0Loader):
        self.gen = lc0_loader

    def __iter__(self):
        for dp in self.gen:
            board, policy, value = dp.board, dp.policy, dp.value
            yield (
                board_to_tensor(board),
                board_to_legal_mask(board),
                policy_dict_to_policy(policy, board),
                torch.Tensor([value]),
            )
