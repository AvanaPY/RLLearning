import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from environments.connect4.game import PyConnectFourGame
from environments.connect4.driver import ConnectFourDriver
from environments.connect4.driver import Actor, PlayerActor, PolicyActor
path = os.path.join(
    'connect4_policies',
    'c4_policy_2023_01_13__00_53_17',
    'policy_model'   
)
ckpt_path = os.path.join(
    'connect4_policies',
    'c4_policy_2023_01_13__00_53_17'
)
policy = tf.saved_model.load(path)

player_actor = PlayerActor('Ahwiiiiiiiiii')
policy_actor = PolicyActor(policy)
game_driver = ConnectFourDriver(
    player_actor,
    policy_actor
)
game_driver.run()