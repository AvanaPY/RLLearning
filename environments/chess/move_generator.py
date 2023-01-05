from typing import *
try:
    from chess import ChessGame, Position
    from piece import Piece, Pawn, Rook, Knight, Bishop, King, Queen
except:
    from environments.chess.chess import ChessGame, Position
    from environments.chess.piece import Piece, Pawn, Rook, Knight, Bishop, King, Queen
    from environments.chess.utils import get_pawn_direction

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
        lst : List[Positoin] = []
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