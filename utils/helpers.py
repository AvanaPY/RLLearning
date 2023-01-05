import os
from typing import *
import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.environments import PyEnvironment, TFEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from model import get_agent
from model.model_config import ModelConfig
from model.model_config import LinearModelConfig
from model.model_config import ConvModelConfig
from utils.my_checkpointer import MyCheckpointLoader
from environments.snake_game import PySnakeGameEnv, ConvPySnakeGameEnv
from environments.snake_game.life_updater import ResetWhenAppleEatenLifeUpdater

def get_policy_from_model_config(model_config : ModelConfig, ckpt_path : str, tf_env : TFPyEnvironment) -> Tuple[DqnAgent, MyCheckpointLoader]:
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent, q_net = get_agent(model_config, tf_env, 1e-3, return_q_net=True)
    checkpointer = MyCheckpointLoader(
        ckpt_dir=ckpt_path,
        agent=agent,
        global_step=global_step,
        max_to_keep=1,
        model_config=model_config
    )
    q_net.summary()
    return agent, checkpointer

def get_pyenvironment_from_model_config(model_config : ModelConfig):
    if isinstance(model_config, LinearModelConfig):
        # Create environment
        return PySnakeGameEnv(
            boardsize,
            ResetWhenAppleEatenLifeUpdater(500)
        )
    elif isinstance(model_config, ConvModelConfig):
        model_config : ConvModelConfig = model_config
        return ConvPySnakeGameEnv(
            board_shape=(32, 32),
            life_updater=ResetWhenAppleEatenLifeUpdater(500)
        )
    else:
        raise RuntimeError(f'Unknown ModelConfig: {model_config}')


def load_checkpoint_from_path(ckpt_path : str) -> Tuple[ModelConfig, PySnakeGameEnv, TFPyEnvironment, DqnAgent, MyCheckpointLoader]:
    mc = ModelConfig.load_config(
        os.path.join(ckpt_path, 'config.json'))

    env = get_pyenvironment_from_model_config(mc)
    tf_env = TFPyEnvironment(env)
    agent, ckptr = get_policy_from_model_config(mc, ckpt_path, tf_env)
    return mc, env, tf_env, agent, ckptr