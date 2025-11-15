import torch
import chess
from chess.polyglot import zobrist_hash
from chess import Board, Move
import numpy as np
from numpy import float64, int64, uint64
from typing import Dict, Sequence, Any
from dataclasses import dataclass
import time

from leela_cnn import board_to_tensor, move_to_index


def flip_move(move: Move) -> Move:
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    return Move(from_square, to_square)


def result_to_value(result: str) -> float64:
    if result == "1-0":
        return float64(1.0)
    elif result == "0-1":
        return float64(-1.0)
    else:
        return float64(0.0)


@dataclass
class Edge:
    q: float64
    num_visits: int64
    p_prior: float64


@dataclass
class Node:
    nn_value: float64
    board: Board
    moves: Sequence[Move]
    edges: Sequence[Edge]
    num_visits: int64

    def register_visit(self, edge: Edge, new_value: float64) -> None:
        edge.q = (edge.q * edge.num_visits + new_value) / (edge.num_visits + 1)
        edge.num_visits += int64(1)
        self.num_visits += int64(1)

    def to_explore(self, c_puct: float64) -> int:
        ucb = [
            edge.q
            + c_puct * edge.p_prior * np.sqrt(self.num_visits) / (1 + edge.num_visits)
            for edge in self.edges
        ]
        move_idx = int(np.argmax(ucb))
        return move_idx


class MCTS:
    model: Any
    c_puct: float64
    nodes: Dict[int, Node]

    def __init__(self, model: Any, c_puct: float64 = float64(1.0)):
        self.model = model
        self.c_puct = c_puct
        self.nodes = {}

    def construct_node(self, board: Board) -> Node:
        board_hash = uint64(zobrist_hash(board))
        if board_hash not in self.nodes:
            tensor_board = board_to_tensor(board)
            nn_policy, nn_val = self.model.predict(tensor_board)
            move_indices = [
                move_to_index(move if board.turn == chess.WHITE else flip_move(move))
                for move in board.legal_moves
            ]
            nn_policy = nn_policy[move_indices]
            torch.softmax(nn_policy, dim=0, out=nn_policy)

            edges = [
                Edge(
                    q=float64(0.0),
                    num_visits=int64(0),
                    p_prior=float64(nn_policy[i].item()),
                )
                for i, move in enumerate(board.legal_moves)
            ]
            self.nodes[board_hash] = Node(
                nn_value=float64(nn_val),
                board=board,
                moves=list(board.legal_moves),
                edges=edges,
                num_visits=int64(0),
            )
        return self.nodes[board_hash]

    def to_explore(self, node: Node) -> int:
        ucb = [
            edge.q
            + self.c_puct
            * edge.p_prior
            * np.sqrt(node.num_visits)
            / (1 + edge.num_visits)
            for edge in node.edges
        ]
        move_idx = int(np.argmax(ucb))
        return move_idx

    def get_node(self, board: Board) -> Node:
        board_hash = uint64(zobrist_hash(board))
        if board_hash not in self.nodes:
            self.construct_node(board)

        return self.nodes[board_hash]

    def search(self, board: Board) -> float64:
        if board.is_game_over():
            return -result_to_value(board.result())

        board_hash = uint64(zobrist_hash(board))
        if board_hash not in self.nodes:
            node = self.construct_node(board)
            return -node.nn_value

        node = self.nodes[board_hash]
        move_idx = node.to_explore(self.c_puct)
        move = node.moves[move_idx]
        edge = node.edges[move_idx]

        next_board = board.copy(stack=False)
        next_board.push(move)
        value = self.search(next_board)
        node.register_visit(edge, value)
        return -value

    def get_most_explored_move(self, board: Board) -> Move:
        if board.is_game_over():
            return Move.null()
        node = self.get_node(board)
        visit_counts = [edge.num_visits for edge in node.edges]
        most_visited_idx = int(np.argmax(visit_counts))
        return node.moves[most_visited_idx]

    def ponder(self, board: Board, num_simulations: int) -> Move:
        if board.is_game_over():
            return Move.null()

        for _ in range(num_simulations):
            self.search(board)

        return self.get_most_explored_move(board)

    def ponder_time(
        self, board: Board, time_ns: int, simulation_time_ns: int = int(1e7)
    ) -> Move:
        stop_time = time.time_ns() + time_ns - simulation_time_ns
        while time.time_ns() <= stop_time:
            self.search(board)

        return self.get_most_explored_move(board)
