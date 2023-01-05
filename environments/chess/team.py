from typing import *
from enum import Enum

class Team(Enum):
    BLACK = 0
    WHITE = 1

def as_string(team : Team):
    if team == Team.WHITE:
        return 'White'
    elif team == Team.BLACK:
        return 'Black'
    
def as_short_string(team : Team):
    return as_string(team)[0]