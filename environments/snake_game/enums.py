from enum import IntEnum, Enum

class SnakeCellState(IntEnum):
    EMPTY   = 0
    HEAD    = 1
    TAIL    = 2
    FOOD    = 3

class Direction(IntEnum):
    RIGHT = 0
    DOWN  = 1
    LEFT  = 2
    UP    = 3

class ActionResult(IntEnum):
    GAME_OVER       = -2
    INVALID_ACTION  = -1
    SELF_COLLISION  = 0
    WALL_COLLISION  = 1
    ATE_FOOD        = 2
    STEPPED_CLOSER  = 3
    STEPPED_FURTHER = 4
    RAN_OUT_OF_TIME = 5