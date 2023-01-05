import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

from environments.snake_game import PySnakeGameEnv, ConvPySnakeGameEnv
from environments.snake_game.life_updater import ResetWhenAppleEatenLifeUpdater 
from environments.snake_game.life_updater import AdditiveWhenAppleEatenLifeUpdater
from environments.snake_game.life_updater import ScoreMultiplyWhenAppleEatenLifeUpdater
from environments.snake_game.life_updater import InfiniteLifeUpdater

from model.model_config import ModelConfig
from model.model_config import LinearModelConfig
from model.model_config import ConvModelConfig

from utils.my_checkpointer import MyCheckpointLoader
from utils.helpers import load_checkpoint_from_path
from utils.helpers import get_policy_from_model_config, get_pyenvironment_from_model_config
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
    w, h = windowsize
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
    font = pygame.font.SysFont('arial', 15)
    display = pygame.display.set_mode(windowsize)
    surf = pygame.surface.Surface(display.get_size())
    clock = pygame.time.Clock()

    # Load the model config
    ckpt_dir = args.modelfolder
    ckpt_name = args.modelname
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    model_config, env, tf_env, agent, checkpointer = load_checkpoint_from_path(ckpt_path)
    global_step = tf.compat.v1.train.get_global_step()
    step_counter = agent.train_step_counter
    policy = agent.policy

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

        train_steps = step_counter.numpy()
        steps_surface = font.render(f'Total train steps: {train_steps}', True, (255, 255, 255))
        display.blit(steps_surface, (20, h - 20))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        clock.tick(FRAME_RATE)

        if timestep.is_last():
            policy = checkpointer.reload_latest()