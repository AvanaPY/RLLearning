from typing import *
from abc import ABC, abstractmethod

try:
    from team import Team
    from team import as_short_string
except:
    from environments.chess.team import Team
    from environments.chess.team import as_short_string

class Piece(ABC):
    def __init__(self, name : str, team : Team):
        self._name = name
        self._team = team
        self._has_moved = False

    @property
    def team(self):
        return self._team

    @property
    def name(self):
        return self._name

    @property
    def has_moved(self):
        return self._has_moved

    def on_move(self):
        self._has_moved = True

    def __str__(self):
        return (f'{self.name}{as_short_string(self.team)}')

class Pawn(Piece):
    def __init__(self, team : Team):
        super(Pawn, self).__init__('P', team)
        
class Rook(Piece):
    def __init__(self, team : Team):
        super(Rook, self).__init__('R', team)

class Knight(Piece):
    def __init__(self, team : Team):
        super(Knight, self).__init__('Kn', team)

class Bishop(Piece):
    def __init__(self, team : Team):
        super(Bishop, self).__init__('B', team)

class King(Piece):
    def __init__(self, team : Team):
        super(King, self).__init__('K', team)

class Queen(Piece):
    def __init__(self, team : Team):
        super(Queen, self).__init__('Q', team)