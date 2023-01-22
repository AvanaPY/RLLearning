import numpy as np
from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import TFPolicy
from environments.connect4.game import PyConnectFourGame
from environments.connect4.game import Player
from collections import namedtuple

Action = namedtuple('Action', 'action')

class Actor:
    def __init__(self, name : str):
        self._name = name
    
    def __str__(self):
        return f'Player(name={self._name})'
    def action(self, py_env : PyConnectFourGame, tf_env : TFPyEnvironment):
        raise NotImplementedError
        
class PlayerActor(Actor):
    def __init__(self, name : str):
        super().__init__(name)
        
    def action(self, py_env : PyConnectFourGame, tf_env : TFPyEnvironment):
        receiving = True
        py_env.fancy_print()
        while receiving:
            inp = input('>Input which column to play:')
            try:
                inp = int(inp)
                receiving = False
                return Action(inp)
            except:
                print(f'Only input integers')
        
class PolicyActor(Actor):
    def __init__(self, policy : TFPolicy):
        super().__init__('Fancy pants')
        self._policy = policy
        
    def action(self, py_env : PyConnectFourGame, tf_env : TFPyEnvironment):
        ts = tf_env.current_time_step()
        a = self._policy.action(ts)
        return Action(a.action.numpy().astype(np.int32)[0])
    
class ConnectFourDriver:
    def __init__(self, actor_1 : Actor, actor_2 : Actor):
        self._actor1 = actor_1
        self._actor2 = actor_2
        self._turn = None
        self._py_env = PyConnectFourGame()
        self._tf_env = TFPyEnvironment(self._py_env)
        
        self._player_map = {
            Player.white : self._actor1,
            Player.black : self._actor2
        }
        self.update_turn()
        
    def update_turn(self):
        self._turn = self._player_map[self._py_env.game.player_turn]
        
    def run(self):
        print(f'First turn: {self._turn._name}')
        running = True
        
        ts = self._tf_env.reset()
        while running:
            action = self._turn.action(self._py_env, self._tf_env)
            print(f'\tPlaying action {action} ')
            action_res = self._tf_env.step(action.action)
            self.update_turn()
            print(f'\tUpdating turn to: {self._turn._name}')
            
            game_over = self._py_env.game.game_over
            
            running = not game_over
        print(f'Game over :: Winner = {self._player_map[self._py_env.game.winner]}')
        self._py_env.fancy_print()