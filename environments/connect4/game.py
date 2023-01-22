from typing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import namedtuple
import numpy as np
from enum import Enum

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs.array_spec import BoundedArraySpec
boardSize = namedtuple('boardSize', 'width height')

class Player(Enum):
    white = 0
    black = 1

class ActionResult(Enum):
    FULL_COLUMN   = 0
    GAME_ENDED = 1
    GAME_END_OLD   = 2
    PLACED     = 100

class ConnectFourGame:
    def __init__(self, 
                 board_size : Union[boardSize, Tuple[int, int]] = boardSize(7 ,7),
                 history_length : int = 7):
        if type(board_size) == boardSize:
            self._board_size : boardSize = board_size
        else:
            self._board_size : boardSize = boardSize(*board_size)
        self._history_length = history_length
        self._winner = None
        self.reset()
    
    @property
    def game_over(self):
        return self._winner is not None
    
    @property
    def winner(self):
        return self._winner
    
    @property
    def history(self):
        return self._history
    
    @property
    def player_turn(self):
        return self._player_turn
    
    def reset(self):
        self._winner = None
        self._state = np.zeros(shape=self._board_size, dtype=np.int32)
        self._player_turn = Player.white
        self._history = [None] * self._history_length
        
    def toggle_turn(self):
        self._player_turn = Player.black if self._player_turn == Player.white else Player.white
    
    def get_player_play_value(self, player : Player):
        if player == Player.white:
            return 1
        elif player == Player.black:
            return 2
    
    def _check_victory(self, played_col : int, played_row : int):
        # Check column
        played_value = self._state[played_row, played_col]
        if played_row >= 3:
            column_win = True
            for i in range(played_row, played_row-4, -1):
                if self._state[i, played_col] != played_value:
                    column_win = False
                    break
            if column_win:
                self._winner = self._player_turn
                return True
        
        # Check row
        for start_col in range(played_col-3, played_col+3):
            row = self._state[played_row, start_col:start_col+4]
            if len(row) == 4:
                bool_row = row == played_value
                if bool_row.all():
                    self._winner = self._player_turn
                    return True
        
        # Check positive diagonal
        for start_col in range(played_col-3, played_col+3):
            diag = self._state.diagonal(start_col)
            diag = diag[np.where(diag == played_value)]
            if len(diag) >= 4:
                self._winner = self._player_turn
                return True
            
        # Check negative diagonal
        rotated_state = np.rot90(self._state)
        for start_col in range(played_col-3, played_col+3):
            diag = rotated_state.diagonal(start_col, axis1=1, axis2=0)
            diag = diag[np.where(diag == played_value)]
            if len(diag) >= 4:
                self._winner = self._player_turn
                return True
            
        return False
    
    def action(self, col_idx : int):
        if self.game_over:
            return ActionResult.GAME_END_OLD
        column = self._state[:,col_idx]
        arg = np.argwhere(column).flatten()
        if arg.size == self._board_size.height:
            return ActionResult.FULL_COLUMN
        elif arg.size > 0:
            row_idx = arg[-1] + 1
        else:
            row_idx = 0
        
        player_value = self.get_player_play_value(self._player_turn)
        self._state[row_idx,col_idx] = player_value
        self._history = self._history[1:] + [self._state.copy()]
        self._check_victory(col_idx, row_idx)
        self.toggle_turn()
        
        if self.game_over:
            return ActionResult.GAME_ENDED
        return ActionResult.PLACED

class PyConnectFourGame(py_environment.PyEnvironment):
    def __init__(self):
        self._game = ConnectFourGame(
            board_size=(7, 7),
            history_length=7
        )
        # 9 states
        #   1 for current
        #   6 for historic states
        #   1 for player turn, 0s for white, 1s for white 
        self._observation_spec = BoundedArraySpec(
            shape=(9, 7, 7),
            minimum=0,
            maximum=1,
            dtype=np.int32,
            name='observation'
        )
        self._action_spec = BoundedArraySpec(
            shape=(),
            minimum=0,
            maximum=6,
            dtype=np.int32,
            name='action'
        )
        self._episode_ended = False
  
    @property
    def game(self):
        return self._game
    
    @property
    def winner(self):
        return self.game.winner
  
    def observation_spec(self):
        return self._observation_spec
    
    def action_spec(self):
        return self._action_spec
    
    def _reset(self):
        print(f'RESET GAME')
        self._game.reset()
        self._episode_ended = False
        return ts.restart(self._get_state())
    
    def _step(self, action : int):
        print(f'PY: Playing action {action}')
        if self._episode_ended:
            print(f'Ended')
            return self._reset()
        
        result : ActionResult = self._game.action(action)

        if result == ActionResult.GAME_ENDED:
            self._episode_ended = True
            return ts.termination(
                self._get_state(),
                reward=1
            )
        elif result == ActionResult.FULL_COLUMN:
            return ts.termination(
                self._get_state(),
                reward=-1
            )
        else:
            return ts.transition(
                self._get_state(), 
                reward=0,
                discount=0.9
            )
    
    def _get_state(self):
        
        curr_value  = self.game.get_player_play_value(self.game.player_turn)
        white_value = self.game.get_player_play_value(Player.white)
        black_value = self.game.get_player_play_value(Player.black)
        state = np.zeros(shape=self._observation_spec.shape, dtype=np.int32)
        state[0,self.game._state != 0] = -1
        state[0,self.game._state == white_value] = 1
        for i in range(1, 8):
            history_item = self.game.history[-i]
            if history_item is not None:
                state[i,history_item != 0] = -1
                state[i,history_item == white_value] = 1
        state[-1,:,:] = curr_value -1
        return state
    
    def fancy_print(self):
        print(self.game._state)
    
if __name__ == '__main__':
    game = PyConnectFourGame()
    for action in [0, 3, 2, 1, 2]:
        game.step(action)
    
    for action in [2, 3, 4, 5]:
        print(f'Turn {game.game.player_turn} | ', end='')
        time_step = game.step(action)
        print(f'Move {action} :: Result = {time_step.reward}')
        print(f'Board state:')
        print(game.game._state)
        print(f'AI State:')
        print(game._get_state())
        
    print(f'Winner: {game.winner}, ({game.game.get_player_play_value(game.winner)})')