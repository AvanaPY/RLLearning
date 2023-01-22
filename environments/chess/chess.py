from typing import *

import functools
import operator
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from environments.chess.piece import Piece, Pawn, Rook, Knight, Bishop, King, Queen
from environments.chess.team import Team
from environments.chess.utils import get_pawn_direction

Position = namedtuple('Position', 'x y')
BoardSize = namedtuple('BoardSize', 'width height')

def create_position_range(f : Position, t : Position):
    pos_range = []
    if f.x == t.x:
        x, y = f
        tx, ty = t
        d = 1 if (ty - y) > 0 else -1
        while y != ty:
            pos_range.append(Position(x, y))
            y += d
        pos_range.append(Position(x, y))
    elif f.y == t.y:
        x, y = f
        tx, ty = t
        d = 1 if (tx - x) > 0 else -1
        while x != tx:
            pos_range.append(Position(x, y))
            x += d
        pos_range.append(Position(x, y))
    else:
        raise RuntimeError(f'Cannot generate range of positions if the origin and end aren\'t on the same row or column')
    return pos_range

class ChessGame:
    def __init__(self):
        self._board_size = BoardSize(8, 8)
        self._board = None
        self.reset()

    def reset(self):
        self._board = [None] * functools.reduce(operator.mul, self._board_size)

        for pos in create_position_range(Position(0, 1), Position(7, 1)):
            self.set_piece_at_position(pos, Pawn(Team.WHITE))
        for pos in create_position_range(Position(0, 6), Position(7, 6)):
            self.set_piece_at_position(pos, Pawn(Team.BLACK))

        for pos, piece in ( (Position(0, 0), Rook(Team.WHITE)), 
                            (Position(7, 0), Rook(Team.WHITE)),
                            (Position(0, 7), Rook(Team.BLACK)),
                            (Position(7, 7), Rook(Team.BLACK)),
                            (Position(1, 0), Knight(Team.WHITE)),
                            (Position(6, 0), Knight(Team.WHITE)),
                            (Position(1, 7), Knight(Team.BLACK)),
                            (Position(6, 7), Knight(Team.BLACK)),
                            (Position(2, 0), Bishop(Team.WHITE)),
                            (Position(5, 0), Bishop(Team.WHITE)),
                            (Position(2, 7), Bishop(Team.BLACK)),
                            (Position(5, 7), Bishop(Team.BLACK)),
                            (Position(3, 0), King(Team.WHITE)),
                            (Position(4, 0), Queen(Team.WHITE)),
                            (Position(3, 7), King(Team.BLACK)),
                            (Position(4, 7), Queen(Team.BLACK)),):
            self.set_piece_at_position(pos, piece)

    def to_index(self, pos : Position):
        return self._board_size.width * pos.y + pos.x

    def to_position(self, index : int):
        x = index % self._board_size.width
        y = index // self._board_size.height
        return Position(x, y)

    def find_all_positions_of_piece(self, piece : Union[Pawn, Rook, Knight, Bishop, Queen, King], team : Team):
        positions = []
        for i, p in enumerate(self._board):
            if isinstance(p, piece) and p.team == team:
                positions.append(i)
        return positions
    
    def move_piece_to_position(self, pos_from : Position, pos_to : Position, assume_valid : bool = False) -> Optional[Piece]:
        piece = self.get_piece_at_position(pos_from)
        if piece is None:
            return
        captured = self.get_piece_at_position(pos_to)
        self.set_piece_at_position(pos_from, None)
        self.set_piece_at_position(pos_to, piece)

        piece.on_move()
        return captured

    def set_piece_at_position(self, pos : Position, piece : Piece):
        index = self.to_index(pos)
        self._board[index] = piece

    def get_piece_at_position(self, pos : Position) -> Optional[Piece]:
        index = self.to_index(pos)
        return self._board[index]

    def _piece_to_str(self, piece):
        if piece is None:
            return '    '
        return str(piece).ljust(4)

    def print_board(self):
        print('\u250c' + '\u2508' * (self._board_size.width * 4 + 2) + '\u2510')
        print('\u2502  ' + self._piece_to_str(None) * self._board_size.width + '\u2502')
        for y in range(self._board_size.height):        # y
            print('\u2502  ', end='')
            for x in range(self._board_size.width):    # x
                pos = Position(x, y)
                piece = self.get_piece_at_position(pos)
                piece_str = self._piece_to_str(piece)
                print(piece_str, end='')
            print('\u2502')
            print('\u2502  ' + self._piece_to_str(None) * self._board_size.width + '\u2502')
        print('\u2514' + '\u2508' * (self._board_size.width * 4 + 2) + '\u2518')

class PyChessGame(py_environment.PyEnvironment):
    def __init__(self):
        self._game = ChessGame()
        self._move_generator = MoveGenerator(self._game)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2, 8, 8), dtype=np.float32,
            minimum=0, maximum=1
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6, 8, 8), dtype=np.float32,
            minimum=-1, maximum=1
        )
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _get_state(self):
        state = np.zeros(shape=(6, 8, 8), dtype=np.float32)
        
        MY_TEAM = Team.WHITE
        ENEMY_TEAM = Team.BLACK if MY_TEAM == Team.WHITE else Team.WHITE
        
        # This dictionary maps piece classes to which dimension of the state they should be represented as. i.e the first dimension represents pawns, second rooks etc etc
        piece2dim = {
            Pawn   : 0,
            Rook   : 1,
            Knight : 2,
            Bishop : 3,
            King   : 4,
            Queen  : 5
        }
        # Iterate over all pieces and fill the state. 
        # Calculate the position in the array with the ChessGame.to_position function which takes an index and turns it into a 2D point on the game board
        # Then assign that position to 1 or -1 depending on if it is my piece or an enemy piece
        for i, p in enumerate(self._game._board):
            if p is None:
                continue
            pos = self._game.to_position(i)
            value = 1 if p.team == MY_TEAM else -1
            state[piece2dim[type(p)], pos.y, pos.x] = value
        
        # Second 
        return tf.expand_dims(state, 0)

    def _reset(self):
        self._game.reset()
        self._episode_ended = False
        return ts.restart(self._get_state())

    def print_board(self):
        """
            Wrapper for ChessGame.print_board
        """
        self._game.print_board()

    def _step(self, action):
        return self._reset()
        
class MoveList:
    def __init__(self, piece : Piece, position : Position):
        self._piece = piece
        self._org_position = position
        self._moves = []

    def __str__(self):
        s = f'Moves for {self._piece} at {self._org_position}'
        for move in self._moves:
            s += f'\n\t{move}'
        return s

    def add_move(self, pos : Position):
        self._moves.append(pos)

    def extend_with_list(self, lst : List[Position]):
        self._moves += lst

class MoveGenerator:
    def __init__(self, chess_game : ChessGame):
        self._game = chess_game

    def generate_moves_for_position(self, pos : Position) -> MoveList:
        piece = self._game.get_piece_at_position(pos)
        if piece is None:
            return None
        piece_name = piece.__class__.__name__.lower()
        func_name  = f'_{piece_name}'

        func = getattr(self, func_name)
        moves = func(pos, piece)
        return moves

    def __generate_positions_in_direction(self, start : Position, direction : Tuple[int, int]) -> List[Position]:
        dx, dy = direction
        x, y = start
        x, y = x + dx, y + dy
        lst : List[Position] = []
        while 0 <= x and x < self._game._board_size.width and 0 <= y and y < self._game._board_size.height:
            lst.append(Position(x, y))
            if not (self._game.get_piece_at_position(Position(x, y)) is None):
                break
            x, y = x + dx, y + dy
        return lst

    def _pawn(self, pos : Position, piece : Pawn) -> MoveList:
        direction = get_pawn_direction(piece.team)

        ml = MoveList(piece, pos)
        x, y = pos

        if self._game.get_piece_at_position(Position(x, y + direction)) == None:
            ml.add_move(Position(x, y + direction))
            if not piece.has_moved and self._game.get_piece_at_position(Position(x, y + direction * 2)) is None:
                ml.add_move(Position(x, y + direction * 2))

        p = self._game.get_piece_at_position(Position(x - 1, y + direction))
        if not (p is None) and p.team != piece.team and x - 1 >= 0:
            ml.add_move(Position(x - 1, y + direction))
        p = self._game.get_piece_at_position(Position(x + 1, y + direction))
        if not (p is None) and p.team != piece.team and x + 1 < self._game._board_size.width:
            ml.add_move(Position(x + 1, y + direction)) 

        return ml

    def _rook(self, pos : Position, piece : Rook) -> MoveList:
        ml = MoveList(piece, pos)
        x, y = pos
        dirs = [
            (0, -1), (1, 0), (0, 1), (-1, 0)
        ]
        for d in dirs:
            l = self.__generate_positions_in_direction(pos, d)
            if len(l) > 0:
                piece_at_last : Optional[Piece] = self._game.get_piece_at_position(l[-1])
                if not (piece_at_last is None) and piece_at_last.team == piece.team:
                    l = l[:-1]
                ml.extend_with_list(l)
        return ml

    def _knight(self, pos : Position, piece : Knight) -> MoveList:
        ml = MoveList(piece, pos)
        x, y = pos

        offsets = [
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1),
            (2, 1),
            (1, 2)
        ]
        for ox, oy in offsets:
            nx, ny = x + ox, y + oy
            if nx < 0 or nx >= self._game._board_size.width or ny < 0 or ny >= self._game._board_size.height:
                continue
            p = self._game.get_piece_at_position(Position(nx, ny))
            if not(p is None) and p.team == piece.team:
                continue    
            ml.add_move(Position(nx, ny))

        return ml

    def _bishop(self, pos : Position, piece : Bishop) -> MoveList:
        ml = MoveList(piece, pos)
        x, y = pos
        dirs = [
            (-1, -1), (1, -1), (1, 1), (-1, 1)
        ]
        for d in dirs:
            l = self.__generate_positions_in_direction(pos, d)
            if len(l) > 0:
                piece_at_last : Optional[Piece] = self._game.get_piece_at_position(l[-1])
                if not (piece_at_last is None) and piece_at_last.team == piece.team:
                    l = l[:-1]
                ml.extend_with_list(l)
        return ml

    def _king(self, pos : Position, piece : King) -> MoveList:
        ml = MoveList(piece, pos)
        x, y = pos
        offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= self._game._board_size.width or ny < 0 or ny >= self._game._board_size.height:
                continue
            p = self._game.get_piece_at_position(Position(nx, ny))
            if p is None:
                ml.add_move(Position(nx, ny))
            elif p.team != piece.team:
                ml.add_move(Position(nx, ny))
        return ml

    def _queen(self, pos : Position, piece : Bishop) -> MoveList:
        ml = MoveList(piece, pos)
        x, y = pos
        dirs = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]
        for d in dirs:
            l = self.__generate_positions_in_direction(pos, d)
            if len(l) > 0:
                piece_at_last : Optional[Piece] = self._game.get_piece_at_position(l[-1])
                if not (piece_at_last is None) and piece_at_last.team == piece.team:
                    l = l[:-1]
                ml.extend_with_list(l)
        return ml
    
if __name__ == '__main__':
    cg = ChessGame()
    cg.move_piece_to_position(Position(0, 0), Position(0, 0), True)
    cg.print_board()