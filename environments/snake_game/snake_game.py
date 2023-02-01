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
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.specs import array_spec
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts


try:
    from environments.snake_game.enums import SnakeCellState, Direction, ActionResult
    from environments.snake_game.context import GameContext
    from environments.snake_game.life_updater import BaseLifeUpdater
    from environments.snake_game.life_updater import ResetWhenAppleEatenLifeUpdater
    from environments.snake_game.life_updater import AdditiveWhenAppleEatenLifeUpdater
except ModuleNotFoundError:
    from enums import SnakeCellState, Direction, ActionResult
    from context import GameContext
    from life_updater import BaseLifeUpdater
    from life_updater import ResetWhenAppleEatenLifeUpdater
    from life_updater import AdditiveWhenAppleEatenLifeUpdater

RENDER_STR = ' HTF'

Point = namedtuple('Point', 'y, x')

def opposite_direction(direction : Direction):
    return [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN][direction]

def ensure_odd(value : int, offset : int):
    if value % 2 == 0:
        return value + offset
    return value

class SnakeGame:
    def __init__(self,  
            board_shape : Tuple[int] = (16, 10), 
            life_updater : BaseLifeUpdater = None) -> None:
        self._board_shape = board_shape
        self._life_updater = ResetWhenAppleEatenLifeUpdater(200) if life_updater is None else life_updater
        #--------------------------State Related----------------------#
        self._game_over = False
        self._state = None
        self._food  = None
        self._direction = Direction.RIGHT
        self._score = 0
        self._steps_left = 0
        self.reset()
        #-------------------------------------------------------------#
        print(f'Initialized SnakeGame:')
        print(f'\tBoard shape: {self._board_shape}')
        print(f'\tUpdater    : {self._life_updater.__class__.__name__}')

    @property
    def head(self):
        return self._state[0]

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

    def step(self, action : Direction) -> GameContext:
        if self._game_over:
            return GameContext.GameOver(self._Score)
        

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

        action_result =  self._move_snake(Point(y, x))
        self._steps_left = self._life_updater.update_life(
            self._steps_left, 
            GameContext(self._score, action_result))

        if self._steps_left <= 0 and action_result != ActionResult.ATE_FOOD:
            self._game_over = True
            return GameContext(
                self._score,
                ActionResult.RAN_OUT_OF_TIME
            )
        else:
            return GameContext(
                self._score,
                action_result
            )

    def _move_snake(self, new_head : Point):

        ny, nx = new_head
        if new_head in self._state:
            self._game_over = True
            return ActionResult.SELF_COLLISION
        if ny < 0 or ny >= self._board_shape[0] or nx < 0 or nx >= self._board_shape[1]:
            self._game_over = True
            return ActionResult.WALL_COLLISION

        self._head = new_head
        self._state.insert(0, self._head)
        if self._head == self._food:
            self._random_place_food()
            self._score += 1
            return ActionResult.ATE_FOOD
        else:
            self._state.pop()

        hy, hx = self._head
        fy, fx = self._food
        ly, lx = self._state[1]

        dy1, dx1 = abs(fy - hy), abs(fx - hx)
        dy2, dx2 = abs(fy - ly), abs(fx - lx)
        if dy1 < dy2 or dx1 < dx2:
            return ActionResult.STEPPED_CLOSER
        return ActionResult.STEPPED_FURTHER

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
    def __init__(self, 
            board_shape : Tuple[int], 
            life_updater:BaseLifeUpdater,
            discount:float,
            reward_on_death:float,
            reward_on_apple:float,
            reward_on_step_closer:float,
            reward_on_step_further:float):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(28,), dtype=np.float32, minimum=-1, maximum=1, name='observation'
        )
        self._game = SnakeGame(
            board_shape=board_shape,
            life_updater=life_updater
        )
        self._episode_ended = False

        self._board_shape = board_shape
        self._life_updater = life_updater
        self._discount = discount
        self._reward_on_death = reward_on_death
        self._reward_on_apple = reward_on_apple
        self._reward_on_step_closer = reward_on_step_closer
        self._reward_on_step_further = reward_on_step_further
        
    def deep_copy(self):
        raise NotImplementedError(f'Deeop copying is not implemented for class {self.__class__.__name__}')

    def save_config_to_folder(self, folder : str):
        raise NotImplementedError(f'Saving configuration is not implemented for class {self.__class__.__name__}')

    @staticmethod
    def load_config(path : str):
        with open(path, 'r') as f:
            conf = json.load(f)
        return conf

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
            self._state_to_wall(),
            self._relative_food_pos()
        ]
        return np.concatenate(states).astype(np.float32)

    def _relative_food_pos(self):
        fy, fx = self._game._food
        hy, hx = self._game.head
        return np.array([
            fy <= hy,
            hy <= fy,
            fx <= hx,
            hx <= fx
        ])

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
        game_context = self._game.step(action_direction)
        result = game_context.action_result

        terminating = [
            ActionResult.SELF_COLLISION,
            ActionResult.WALL_COLLISION,
            ActionResult.RAN_OUT_OF_TIME
        ]
        terminating_rewards = {
            ActionResult.SELF_COLLISION :  self._reward_on_death,
            ActionResult.WALL_COLLISION :  self._reward_on_death,
            ActionResult.RAN_OUT_OF_TIME : self._reward_on_death
        }
        if result in terminating:
            self._episode_ended = True
            reward = terminating_rewards[result]
            return ts.termination(self._get_state(), reward=reward)
        elif result == ActionResult.ATE_FOOD:
            return ts.transition(self._get_state(), reward=self._reward_on_apple, discount=self._discount)
        elif result == ActionResult.STEPPED_CLOSER:
            return ts.transition(self._get_state(), reward=self._reward_on_step_closer, discount=self._discount)
        elif result == ActionResult.STEPPED_FURTHER:
            return ts.transition(self._get_state(), reward=self._reward_on_step_further, discount=self._discount)
        else:
            raise ValueError(f'Unknown result: {result}')

    def render(self, window_size=640, rotate:bool=False):
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
        if rotate:
            image = np.rot90(image, k=1, axes=(0,1))
            image = np.flip(image, axis=0)
        return image.astype(np.uint8)

class ConvPySnakeGameEnv(PySnakeGameEnv):
    def __init__(self, 
                 board_shape : Tuple[int], 
                 observation_spec_shape : Tuple[int],
                 life_updater:BaseLifeUpdater,
                 discount:float,
                 reward_on_death:float,
                 reward_on_apple:float,
                 reward_on_step_closer:float,
                 reward_on_step_further:float):
        super().__init__(board_shape=board_shape, 
                         life_updater=life_updater,
                         discount=discount,
                         reward_on_death=reward_on_death,
                         reward_on_apple=reward_on_apple,
                         reward_on_step_closer=reward_on_step_closer,
                         reward_on_step_further=reward_on_step_further)
        self.observation_spec_shape = observation_spec_shape
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=observation_spec_shape,
            dtype=np.float32, 
            minimum=0, maximum=1, name='observation'
        )

        self._HEAD_TOKEN = 1
        self._TAIL_TOKEN = 2
        self._FOOD_TOKEN = 3

    def deep_copy(self):
        return ConvPySnakeGameEnv(
            board_shape=self._game._board_shape,
            observation_spec_shape=self.observation_spec_shape,
            life_updater=self._game._life_updater,
            discount=self._discount,
            reward_on_death=self._reward_on_death,
            reward_on_apple=self._reward_on_apple,
            reward_on_step_closer =self._reward_on_step_closer,
            reward_on_step_further=self._reward_on_step_further
        )

    def save_config_to_folder(self, folder : str):
        config_path = os.path.join(folder, 'gameconfig.conf')
        config = {
            "board_shape" : self._board_shape,
            "observation_spec_shape" : self.observation_spec_shape,
            "discount" : self._discount,
            "reward_on_death" : self._reward_on_death,
            "reward_on_apple" : self._reward_on_apple,
            "reward_on_step_closer" : self._reward_on_step_closer,
            "reward_on_step_further" : self._reward_on_step_further
        }
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(config_path, 'w+') as f:
            json.dump(config, f)

    def _get_state(self):
        state = np.zeros(shape=self._game._board_shape, dtype=np.float32)
        
        state[self._game._food] = self._FOOD_TOKEN
        state[self._game.head]  = self._HEAD_TOKEN
        for t in self._game._state[1:]:
            state[t] = self._TAIL_TOKEN
        
        pady, padx, _ = self._observation_spec.shape
        state = np.pad(state, (pady, padx), constant_values=(-1, -1))
        hy, hx = self._game.head
        miny, minx = pady + hy - self._observation_spec.shape[0] // 2    , padx + hx - self._observation_spec.shape[1] // 2
        maxy, maxx = pady + hy + self._observation_spec.shape[0] // 2 + 1, padx + hx + self._observation_spec.shape[1] // 2 + 1
        
        state = state[miny:maxy,minx:maxx]
        state = np.expand_dims(state, -1)
        return state / 3# Divide by 3 to normalize it 

    def render(self, window_size=640, rotate:bool=False):
        BACKGROUND_COLOUR = (50, 50, 50)
        HEAD_COLOUR = (120, 120, 200)
        TAIL_COLOUR = (145, 255, 255)
        FOOD_COLOUR = (255, 100, 100)
        
        fy, fx = self._game._food
        INNER_FACTOR = 0.05

        image = np.zeros(shape=(window_size, window_size, 3))
        block_size = (image.shape[0] // self._game._board_shape[0], image.shape[1] // self._game._board_shape[1])

        image[:,:] = BACKGROUND_COLOUR
        image[fy*block_size[0] : (fy+1)*block_size[0],
              fx*block_size[1] : (fx+1)*block_size[1]] = FOOD_COLOUR
        
        for ty, tx in self._game._state:
            image[ty*block_size[0] : (ty+1)*block_size[0],
                  tx*block_size[1] : (tx+1)*block_size[1]] = TAIL_COLOUR
            
        # Render what he can see
        state = self._get_state()
        h, w, _ = state.shape
        hy, hx = self._game.head
        miny, minx = hy - h // 2    , hx - w // 2
        maxy, maxx = hy + h // 2 + 1, hx + w // 2 + 1
        image = cv2.rectangle(image, (minx * block_size[1], miny * block_size[0]), (maxx * block_size[1], maxy * block_size[0]), (255, 0, 0), 1) 
        
        # Some cool text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_colour = (255, 255, 255)
        image = cv2.putText(image, f'Life:  {self._game._steps_left}',
                                (20, 20), 
                                font, font_scale, text_colour, 1, cv2.LINE_AA)
        image = cv2.putText(image, f'Score: {self._game._score}', 
                                (20, 20 + 20), 
                                font, font_scale, text_colour, 1, cv2.LINE_AA)
        
        # This is fucking bad design man
        if rotate:
            image = np.rot90(image, k=1, axes=(0,1))
            image = np.flip(image, axis=0)
        return image.astype(np.uint8)

if __name__ == '__main__':
    env = ConvPySnakeGameEnv()
    state = env._get_state()
    print(state)