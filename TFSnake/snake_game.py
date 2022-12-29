from typing import *
import numpy as np
import random
import time
import sys
import os
import cv2
from enum import IntEnum
from collections import namedtuple
from functools import reduce

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.specs import array_spec
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts

RENDER_STR = ' HTF'

Point = namedtuple('Point', 'y, x')

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

class GameResult(IntEnum):
    GAME_OVER       = -2
    INVALID_ACTION  = -1
    SELF_COLLISION  = 0
    WALL_COLLISION  = 1
    ATE_FOOD        = 2
    STEPPED_CLOSER  = 3
    STEPPED_FURTHER = 4
    RAN_OUT_OF_TIME = 5

def opposite_direction(direction : Direction):
    return [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN][direction]

class SnakeGame:
    def __init__(self, board_shape : Tuple[int] = (16, 10)) -> None:
        self._board_shape = board_shape
        #--------------------------State Related----------------------#
        self._game_over = False
        self._state = None
        self._food  = None
        self._direction = Direction.RIGHT
        self._score = 0
        self._steps_left = 0
        print(f'Initialized SnakeGame with board {self._board_shape} ({self._steps_per_food})')
        #-------------------------------------------------------------#
        self.reset()

    @property
    def _steps_per_food(self):
        return 100

    def reset(self) -> None:
        self._game_over = False
        self._state = [
            Point(self._board_shape[0] // 2, self._board_shape[1] // 2),
            Point(self._board_shape[0] // 2, self._board_shape[1] // 2 - 1),
            Point(self._board_shape[0] // 2, self._board_shape[1] // 2 - 2),
            Point(self._board_shape[0] // 2, self._board_shape[1] // 2 - 3)
        ]
        self._food = Point(self._board_shape[0] // 2, self._board_shape[1] // 2 + 1)
        self._direction = Direction.RIGHT
        self._score = 0
        self._steps_left = 200
        self._random_place_food()

    def step(self, action : Direction):
        if self._game_over:
            return GameResult.GAME_OVER
        

        y, x = self._state[0]
        self._steps_left -= 1

        reward = 0
        if self._direction != opposite_direction(action):
            self._direction = action

        if self._direction == Direction.RIGHT:
            x += 1
        elif self._direction == Direction.DOWN:
            y += 1
        elif self._direction == Direction.LEFT:
            x -= 1
        elif self._direction == Direction.UP:
            y -= 1
        else:
            raise ValueError(f'`action` must be of type Direction')

        game_result =  self._move_snake(Point(y, x))
        if self._steps_left <= 0 and game_result != GameResult.as_integer_ratio:
            self._game_over = True
            return GameResult.RAN_OUT_OF_TIME
        else:
            return game_result

    def _move_snake(self, new_head : Point):

        ny, nx = new_head
        if new_head in self._state:
            self._game_over = True
            return GameResult.SELF_COLLISION
        if ny < 0 or ny >= self._board_shape[0] or nx < 0 or nx >= self._board_shape[1]:
            self._game_over = True
            return GameResult.WALL_COLLISION

        self._head = new_head
        self._state.insert(0, self._head)
        if self._head == self._food:
            self._random_place_food()
            self._steps_left += self._steps_per_food
            self._score += 1
            return GameResult.ATE_FOOD
        else:
            self._state.pop()

        hy, hx = self._head
        fy, fx = self._food
        ly, lx = self._state[1]

        dy1, dx1 = abs(fy - hy), abs(fx - hx)
        dy2, dx2 = abs(fy - ly), abs(fx - lx)
        if dy1 < dy2 or dx1 < dx2:
            return GameResult.STEPPED_CLOSER
        return GameResult.STEPPED_FURTHER

    def _random_place_food(self):
        y, x = np.random.randint(0, self._board_shape[0]), np.random.randint(0, self._board_shape[1])
        if (y, x) in self._state:
            return self._random_place_food()
        self._food = Point(y, x)

    def render(self) -> None:
        state = np.zeros(shape=self._board_shape, dtype=np.int32)
        state[self._food] = SnakeCellState.FOOD
        for t in self._state[1:]:
            state[t] = SnakeCellState.TAIL
        state[self._state[0]] = SnakeCellState.HEAD
        print(f' _Score: {self._score}, {self._game_over}'.ljust(self._board_shape[1] * 2+1, '_'))
        for i in range(self._board_shape[0]):
            print('|', end='')
            for j in range(self._board_shape[1]):
                print(RENDER_STR[state[i, j]], end=' ')
            print('|')
        print(' ' + '\u203e' * self._board_shape[1] * 2 + ' ')

class PySnakeGameEnv(py_environment.PyEnvironment):
    def __init__(self, board_shape : Tuple[int] = (16, 16)):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(24,), dtype=np.float32, minimum=-1, maximum=1, name='observation'
        )
        self._game = SnakeGame(board_shape=board_shape)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _current_time_step(self):
        return self._get_state()

    def _get_state(self):
        states = [
            self._state_to_food(),
            self._state_to_self(),
            self._state_to_wall()
        ]
        return np.concatenate(states).astype(np.float32)

    def _state_to_food(self):
        s = np.ones(shape=(8,)) * -1
        hy, hx = self._game._state[0]
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        for i, (dy, dx) in enumerate(dirs):
            ny, nx = hy + dy, hx + dx
            while ny >= 0 and ny < self._game._board_shape[0] and nx >= 0 and nx < self._game._board_shape[1]:
                if Point(ny, nx) == self._game._food:
                    dy, dx = ny - hy, nx - hx
                    s[i] = 1 / (dy * dy + dx * dx)**0.5
                    break
                ny, nx = ny + dy, nx + dx
            
        return s

    def _state_to_self(self):
        s = np.ones(shape=(8,)) * -1
        hy, hx = self._game._state[0]
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        for i, (dy, dx) in enumerate(dirs):
            ny, nx = hy + dy, hx + dx
            while ny >= 0 and ny < self._game._board_shape[0] and nx >= 0 and nx < self._game._board_shape[1]:
                if Point(ny, nx) in self._game._state:
                    dy, dx = ny - hy, nx - hx
                    s[i] = 1 / (dy * dy + dx * dx)**0.5
                    break
                ny, nx = ny + dy, nx + dx
        return s

    def _state_to_wall(self):
        s = np.zeros(shape=(8,))
        hy, hx = self._game._state[0]
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        for i, (dy, dx) in enumerate(dirs):
            ny, nx = hy + dy, hx + dx
            while ny > 0 and ny < self._game._board_shape[0] - 1 and nx > 0 and nx < self._game._board_shape[1] - 1:
                ny, nx = ny + dy, nx + dx
            
            dy, dx = ny - hy, nx - hx
            s[i] = 1 / (dy * dy + dx * dx)**0.5
        return s

    def _reset(self):
        self._game.reset()
        self._episode_ended = False
        return ts.restart(self._get_state())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()        

        action_direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP][action]
        result = self._game.step(action_direction)

        terminating = [
            GameResult.SELF_COLLISION,
            GameResult.WALL_COLLISION,
            GameResult.RAN_OUT_OF_TIME
        ]
        terminating_rewards = {
            GameResult.SELF_COLLISION :  -5,
            GameResult.WALL_COLLISION :  -5,
            GameResult.RAN_OUT_OF_TIME : -10
        }
        if result in terminating:
            self._episode_ended = True
            reward = terminating_rewards[result]
            return ts.termination(self._get_state(), reward=reward)
        elif result == GameResult.ATE_FOOD:
            return ts.transition(self._get_state(), reward= 10, discount=0.9)
        elif result == GameResult.STEPPED_CLOSER:
            return ts.transition(self._get_state(), reward=  1, discount=0.9)
        elif result == GameResult.STEPPED_FURTHER:
            return ts.transition(self._get_state(), reward= -1, discount=0.9)
        else:
            raise ValueError(f'Unknown result: {result}')

    def render(self, window_size=640):
        BACKGROUND_COLOUR = (50, 50, 50)
        TAIL_COLOUR = (120, 120, 255)
        HEAD_COLOUR = (120, 120, 200)
        FOOD_COLOUR = (255, 100, 100)
        image = np.zeros(shape=(window_size, window_size, 3))
        image[:,:] = BACKGROUND_COLOUR
        block_size = (image.shape[0] // self._game._board_shape[0], image.shape[1] // self._game._board_shape[1])

        fy, fx = self._game._food
        image[fy*block_size[0]:(fy+1)*block_size[0],fx*block_size[1]:(fx+1)*block_size[1]] = FOOD_COLOUR
        for ty, tx in self._game._state:
            image[ty*block_size[0]:(ty+1)*block_size[0],tx*block_size[1]:(tx+1)*block_size[1]] = TAIL_COLOUR


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_colour = (255, 255, 255)
        image = cv2.putText(image, f'Score: {self._game._score}', 
                                (20, 40), 
                                font, font_scale, text_colour, 1, cv2.LINE_AA)

        image = cv2.putText(image, f'Steps: {self._game._steps_left}',
                                (20, 20), 
                                font, font_scale, text_colour, 1, cv2.LINE_AA)
        image = np.rot90(image, k=1, axes=(0,1))
        image = np.flip(image, axis=0)
        return image.astype(np.uint8)

if __name__ == '__main__':
    env = PySnakeGameEnv()
    s = env.reset()
    print(s)
    s = env.step(0)
    print(s)