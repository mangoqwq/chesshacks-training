import chess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from chess import Board, Move
from .leela_cnn import board_to_tensor


def get_nn_moves(board: Board) -> list[Move]:
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: str(move))
    moves = list(
        filter(lambda m: m.promotion is None or m.promotion == chess.QUEEN, moves)
    )
    return moves


def save_gif(board_states: list[Board], out_path: str) -> None:
    PIECE_UNICODE = {
        chess.PAWN: ("♙", "♟"),
        chess.KNIGHT: ("♘", "♞"),
        chess.BISHOP: ("♗", "♝"),
        chess.ROOK: ("♖", "♜"),
        chess.QUEEN: ("♕", "♛"),
        chess.KING: ("♔", "♚"),
    }

    def draw_board(axis, board):
        axis.clear()
        axis.set_aspect("equal")
        axis.axis("off")
        for rank in range(8):
            for file in range(8):
                facecolor = "#f0d9b5" if (rank + file) % 2 == 0 else "#b58863"
                axis.add_patch(
                    patches.Rectangle(
                        (file, 7 - rank), 1, 1, facecolor=facecolor, edgecolor=facecolor
                    )
                )
        for square, piece in board.piece_map().items():
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            symbol = PIECE_UNICODE[piece.piece_type][0 if piece.color else 1]
            axis.text(
                file + 0.5,
                7 - rank + 0.5,
                symbol,
                ha="center",
                va="center",
                fontsize=28,
                color="#1a1a1a" if piece.color else "#f6f6f6",
                fontweight="bold",
            )
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlim(0, 8)
        axis.set_ylim(0, 8)
        axis.set_title("Board view", fontsize=10)

    initial_tensor = board_to_tensor(board_states[0])
    fig = plt.figure(figsize=(10, 4))
    gridspec = fig.add_gridspec(
        nrows=3, ncols=7, width_ratios=[3, 1, 1, 1, 1, 1, 1], wspace=0.15, hspace=0.2
    )
    board_ax = fig.add_subplot(gridspec[:, 0])
    plane_axes = []
    for row in range(3):
        for col in range(6):
            axis = fig.add_subplot(gridspec[row, col + 1])
            axis.axis("off")
            plane_axes.append(axis)

    plane_count = min(len(plane_axes), initial_tensor.shape[0])
    tensor_images = []
    for plane_idx in range(plane_count):
        img = plane_axes[plane_idx].imshow(
            initial_tensor[plane_idx], vmin=0, vmax=1, animated=True
        )
        plane_axes[plane_idx].set_title(f"Plane {plane_idx}", fontsize=6)
        tensor_images.append(img)
    for extra_axis in plane_axes[plane_count:]:
        extra_axis.axis("off")

    draw_board(board_ax, board_states[0])

    def update_frame(frame_idx: int):
        tensor = board_to_tensor(board_states[frame_idx])
        for plane_idx, img in enumerate(tensor_images):
            img.set_data(tensor[plane_idx])
        draw_board(board_ax, board_states[frame_idx])
        move_label = f"Move {frame_idx} / {len(board_states) - 1}"
        fig.suptitle(f"Random game tensors + board — {move_label}", fontsize=10)
        return tensor_images

    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(board_states),
        interval=400,
        blit=False,
        repeat=False,
    )

    ani.save(out_path, writer="pillow", fps=5)
