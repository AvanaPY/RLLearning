import os
from typing import *
import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks import Network
from tf_agents.environments import PyEnvironment, TFEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from model import get_agent
from model.model_config import ModelConfig
from model.model_config import LinearModelConfig
from model.model_config import ConvModelConfig
from model.res_model import ResidualModelConfig
from utils.my_checkpointer import MyCheckpointLoader
from environments.snake_game import PySnakeGameEnv, ConvPySnakeGameEnv
from environments.snake_game.life_updater import ResetWhenAppleEatenLifeUpdater

def get_pyenvironment_from_model_config(mc : ModelConfig):
    if isinstance(mc, LinearModelConfig):
        mc : LinearModelConfig = mc
        # Create environment
        return PySnakeGameEnv(
            (32, 32),
            ResetWhenAppleEatenLifeUpdater(500)
        )
    elif isinstance(mc, ConvModelConfig):
        mc : ConvModelConfig = mc
        return ConvPySnakeGameEnv(
            board_shape=(32, 32),
            life_updater=ResetWhenAppleEatenLifeUpdater(500)
        )
    elif isinstance(mc, ResidualModelConfig):
        return ConvPySnakeGameEnv(
            board_shape=(32, 32),
            observation_spec_shape=mc.input_shape,
            
            life_updater=ResetWhenAppleEatenLifeUpdater(500)
        )
    else:
        raise RuntimeError(f'Unknown ModelConfig: {mc}')

def load_checkpoint_from_path(ckpt_path : str, 
                              learning_rate : Optional[float]=None) -> Tuple[ModelConfig, PySnakeGameEnv, TFPyEnvironment, DqnAgent, MyCheckpointLoader]:
    if learning_rate is None:
        learning_rate = 1e-10
        print(f'Set `learning_rate` to {learning_rate}!')
    
    config_path = os.path.join(ckpt_path, 'config.json')
    assert os.path.exists(ckpt_path), f'No such checkpoint: `{ckpt_path}`'
    assert os.path.exists(config_path), f'No configuration file exists for checkpoint {ckpt_path}'
    
    mc = ModelConfig.load_config(
        os.path.join(ckpt_path, 'config.json'))

    env = get_pyenvironment_from_model_config(mc)
    tf_env = TFPyEnvironment(env)
    agent, qnet = get_agent(mc, tf_env, learning_rate=learning_rate, return_q_net=True)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    checkpointer = MyCheckpointLoader(
        ckpt_dir=ckpt_path,
        agent=agent,
        global_step=global_step,
        max_to_keep=1,
        model_config=mc
    )
    # qnet.summary()
    return mc, env, tf_env, agent, checkpointer