import os
import argparse

from model.my_checkpointer import MyCheckpointLoader

from environments.snake_game import PySnakeGameEnv, ConvPySnakeGameEnv
from environments.snake_game.life_updater import ResetWhenAppleEatenLifeUpdater 
from environments.snake_game.life_updater import AdditiveWhenAppleEatenLifeUpdater
from environments.snake_game.life_updater import ScoreMultiplyWhenAppleEatenLifeUpdater
from environments.snake_game.life_updater import InfiniteLifeUpdater

from model import load_model
from model import get_agent
from model import create_checkpointer
from model.model_config import ModelConfig
from model.model_config import LinearModelConfig
from model.model_config import ConvModelConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def get_policy_from_model_config(model_config : ModelConfig):
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

    return agent.policy, checkpointer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--framerate', type=int , default=30, dest='framerate')
    parser.add_argument('--board-size', type=int , default=32, dest='boardsize')
    parser.add_argument('--window-size', type=int, default=640, dest='winsize')
    parser.add_argument('--model-folder', type=str, default='saved_policies', dest='modelfolder')
    parser.add_argument('--model-name', type=str , default="Sture2", dest='modelname')
    args = parser.parse_args()

    FRAME_RATE = args.framerate
    boardsize = (args.boardsize, args.boardsize)
    windowsize = (args.winsize, args.winsize)
    model_folder = args.modelfolder
    model_name = args.modelname
    model_path = os.path.join(model_folder, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model \"{model_path}\" does not exist.')
    assert args.winsize % args.boardsize == 0, f'Window Size must be a multiple of board size'
    
    import tensorflow as tf
    import pygame
    from tf_agents.environments import tf_py_environment
    from tf_agents.environments import py_environment
    from tf_agents.environments import tf_environment
    from tf_agents.specs import tensor_spec

    # Init pygame
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('arial', 30)
    display = pygame.display.set_mode(windowsize)
    surf = pygame.surface.Surface(display.get_size())
    clock = pygame.time.Clock()

    # Load the model config
    ckpt_dir = args.modelfolder
    ckpt_name = args.modelname
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model_config : ModelConfig = ModelConfig.load_config(
        os.path.join(ckpt_path, 'config.json'))

    # Load the correct environment for the model
    env = get_pyenvironment_from_model_config(model_config)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    policy, checkpointer = get_policy_from_model_config(model_config)

    # Play the policy
    timestep = tf_env.reset()
    running = True
    while running:
        action = policy.action(timestep)
        timestep = tf_env.step(action.action)
        render = env.render(window_size=args.winsize, rotate=True)

        pygame.surfarray.blit_array(surf, render)

        display.fill((0, 0, 0))
        display.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        clock.tick(FRAME_RATE)

        if timestep.is_last():
            policy = checkpointer.reload_latest()