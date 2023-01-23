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

import json

def get_pyenvironment_from_model_config(mc : ModelConfig, game_config : Dict[str, Any]):
    print(f'HARD CODING `life_updater` to ResetWhenAppleEatenLifeUpdater(500)')
    game_config['life_updater'] = ResetWhenAppleEatenLifeUpdater(500)
    if isinstance(mc, LinearModelConfig):
        mc : LinearModelConfig = mc
        # Create environment
        return PySnakeGameEnv(**game_config)

    elif isinstance(mc, ConvModelConfig):
        mc : ConvModelConfig = mc
        assert game_config['observation_spec_shape'] == mc.input_dims, f'Environment Observation Spec Shape does not match model configuration input dims:\nFound Observation Spec Shape\n\t{tuple(game_config["observation_spec_shape"])}\nReceived input dims\n\t{tuple(mc.input_dims)}'
        return ConvPySnakeGameEnv(**game_config)
    elif isinstance(mc, ResidualModelConfig):
        mc : ResidualModelConfig = mc
        assert game_config['observation_spec_shape'] == mc.input_dims, f'Environment Observation Spec Shape does not match model configuration input dims:\nFound Observation Spec Shape\n\t{tuple(game_config["observation_spec_shape"])}\nReceived input dims\n\t{tuple(mc.input_dims)}'
        return ConvPySnakeGameEnv(**game_config)
    else:
        raise RuntimeError(f'Unknown ModelConfig: {mc}')

def load_checkpoint_from_path(model_path : str,
                              learning_rate : Optional[float]=None) -> Tuple[ModelConfig, PySnakeGameEnv, TFPyEnvironment, DqnAgent, MyCheckpointLoader]:
    if learning_rate is None:
        learning_rate = 1e-10
        print(f'Set `learning_rate` to {learning_rate}!')
    
    # This is bad
    ckpt_path         = os.path.join(model_path, 'ckpts')
    model_config_path = os.path.join(ckpt_path, 'config.json')
    game_config_path  = os.path.join(model_path, 'gameconfig.conf')

    assert os.path.exists(ckpt_path), f'No such checkpoint: `{ckpt_path}`'
    assert os.path.exists(model_config_path), f'No model configuration file: {model_config_path}'
    assert os.path.exists(game_config_path), f'No game configuration file: {game_config_path}'

    mc = ModelConfig.load_config(model_config_path)
    gc = PySnakeGameEnv.load_config(game_config_path)

    env = get_pyenvironment_from_model_config(mc, gc)

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
    return mc, env, tf_env, agent, checkpointer