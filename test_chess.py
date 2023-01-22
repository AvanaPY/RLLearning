import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tf_agents.specs import BoundedArraySpec
from environments.chess.chess import ChessGame, PyChessGame, Position
from environments.chess.piece import Piece
from environments.chess.chess import MoveGenerator
from model.chess_model.model import create_model

game = PyChessGame()
mg = game._move_generator

moves = mg.generate_moves_for_position(Position(0, 1))
print(moves)
game.print_board()

input_spec = game.observation_spec()

model = create_model(
    input_spec=input_spec
)
model.build(input_shape=input_spec.shape)
model.summary()

state = game._get_state()
# print(state[0])
a = model(state)
print(a)