from typing import *
import functools
import operator
from collections import namedtuple

try:
    from piece import Piece, Pawn, Rook, Knight, Bishop, King, Queen
    from team import Team
except:
    from environments.chess.piece import Piece, Pawn, Rook, Knight, Bishop, King, Queen
    from environments.chess.team import Team

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

    def _to_index(self, pos : Position):
        return self._board_size.width * pos.y + pos.x

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
        index = self._to_index(pos)
        self._board[index] = piece

    def get_piece_at_position(self, pos : Position) -> Optional[Piece]:
        index = self._to_index(pos)
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


if __name__ == '__main__':
    cg = ChessGame()
    cg.move_piece_to_position(Position(0, 0), Position(0, 0), True)
    cg.print_board()