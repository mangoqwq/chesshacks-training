import chess
from chess.polyglot import zobrist_hash
from chess import Board, Move
import numpy as np
from numpy import float64, int64
from typing import Dict, Sequence, Any
from dataclasses import dataclass

from leela_cnn import board_to_tensor, tensor_to_board


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
    nodes: Dict[int64, Node]

    def __init__(self, model: Any, c_puct: float64 = float64(1.0)):
        self.model = model
        self.c_puct = c_puct
        self.nodes = {}

    def construct_node(self, board: Board) -> Node:
        board_hash = int64(zobrist_hash(board))
        if board_hash not in self.nodes:
            tensor_board = board_to_tensor(board)
            nn_val, nn_policy = self.model.predict(tensor_board)
            edges = [
                Edge(
                    q=float64(0.0),
                    num_visits=int64(0),
                    p_prior=float64(nn_policy[tensor_to_board(move)]),
                )
                for move in board.legal_moves
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

    def search(self, board: Board) -> float64:
        if board.is_game_over():
            return -result_to_value(board.result())

        board_hash = int64(zobrist_hash(board))
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
