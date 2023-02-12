from typing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SDL_AUDIODRIVER'] = 'dsp'
import argparse
import numpy as np
import tensorflow as tf
from environments.snake_game import PySnakeGameEnv, ConvCenteredPySnakeGameEnv
from environments.snake_game.life_updater import ResetWhenAppleEatenLifeUpdater 
from environments.snake_game.life_updater import AdditiveWhenAppleEatenLifeUpdater
from environments.snake_game.life_updater import ScoreMultiplyWhenAppleEatenLifeUpdater
from environments.snake_game.life_updater import InfiniteLifeUpdater

from model.model_config import ModelConfig
from model.model_config import LinearModelConfig
from model.model_config import ConvModelConfig

from utils.my_checkpointer import MyCheckpointLoader
from utils.helpers import load_checkpoint_from_path

STEP_WISE_FRAME_ENABLED = False
SHOW_TEXT_ENABLED       = True

def get_latest_model_name(model_folder : str) -> str:
    if not os.path.exists(model_folder):
        raise FileExistsError(f'Model folder \"{model_folder}\" does not exist')
    models = list(sorted(os.listdir(model_folder)))
    if len(models) == 0:
        raise FileExistsError(f'Folder \"{model_folder}\" contains no potential models.')
    return models[-1]
    
def get_model_name_from_args(args : Dict[str, Any]) -> str:
    if args.load_latest:
        return get_latest_model_name(args.modelfolder)
    return args.modelname

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framerate', type=int , default=30, dest='framerate')
    parser.add_argument('-b', '--board-size', type=int , default=32, dest='boardsize')
    parser.add_argument('-w', '--window-size', type=int, default=640, dest='winsize')
    parser.add_argument('-f', '--model-folder', type=str, default='models', dest='modelfolder')
    parser.add_argument('-m', '--model-name', type=str , default="Sture2", dest='modelname')
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu', help='Turn on GPU acceleration')
    parser.add_argument('-l', '--load-latest', action='store_true', dest='load_latest', help="Turn on to load the latest model in model-folder. Takes priority over model-name")
    parser.add_argument('-v', '--verbose',  action='store_true', dest='verbose', help='Turn on verbose')
    args = parser.parse_args()

    device = 'cpu' if not args.gpu else 'gpu'
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if args.verbose:
            print(f'Playing on CPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            print(f'No GPU support! Exiting')
            exit(0)
        if args.verbose:
            print(f'Playing on GPU, may take a moment to initialise.')
            
    FRAME_RATE = args.framerate
    boardsize = (args.boardsize, args.boardsize)
    windowsize = (args.winsize, args.winsize)
    w, h = windowsize
    
    model_folder = args.modelfolder
    model_name = get_model_name_from_args(args)
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

    # Load model and environment
    model_config, env, tf_env, agent, checkpointer = load_checkpoint_from_path(model_path, None)
    agent._q_network.trainable = False
    env._game._life_updater = ResetWhenAppleEatenLifeUpdater(200)
    global_step = tf.compat.v1.train.get_global_step()
    step_counter = agent.train_step_counter
    policy = agent.policy
    
    if args.verbose:
        print(f'Loaded {model_path}')
        print(f'Loaded {env.__class__}')
        agent._q_network.summary()

    # Play the policy
    timestep = tf_env.reset()
    running = True
    
    # Get initial state and action
    state = timestep.observation
    out = agent._q_network(state)[0][0]
    action = np.argmax(out)
    
    while running:
        # Perform action
        action = tf.constant(action, shape=(1,), dtype=np.int32)
        timestep = tf_env.step(action)

        # Update state and action for the next frame.
        # We do this before rendering the current frame
        #   so we can get information regarding Q values, etc.
        state = timestep.observation
        out = agent._q_network(state)[0][0]
        action = np.argmax(out)
        expected_rewards = out.numpy()

        # Render current frame after action
        render = env.render(window_size=args.winsize, rotate=True)
        pygame.surfarray.blit_array(surf, render)
        display.blit(surf, (0, 0))

        if SHOW_TEXT_ENABLED:
            # Render total training steps done
            train_steps = step_counter.numpy()
            steps_surface  = font.render(f'Total train steps: {train_steps:15,d}', True, (255, 255, 255))
            display.blit(steps_surface, (w - steps_surface.get_width() - 10, 0))
            
            # Render rewards, bad code but works
            x = 20
            y = h - 40
            reward_surface = font.render(f'Expected reward:', True, (255, 255, 255))
            display.blit(reward_surface, (x, y))
            
            x += reward_surface.get_width() + 30
            
            c1 = pygame.Color(255, 0, 0)
            c2 = pygame.Color(0, 255, 0)
            c3 = pygame.Color(255, 255, 0)
            
            max_rew = np.max(expected_rewards)
            min_rew = np.min(expected_rewards)
            rel_poses = [
                (20, 0),
                (0, 20),
                (-20, 0),
                (0, -20)
            ]
            for value, (rx, ry) in zip(expected_rewards, rel_poses):
                # pct =  (value - min_rew) / (max_rew - min_rew)
                # c = c1.lerp(c2, pct)
                
                if value == max_rew:
                    c = c2
                elif value == min_rew:
                    c = c1
                else:
                    c = c3
                
                value_surf = font.render(f'{value:4.1f}', True, c)
                display.blit(value_surf, (x + rx, y + ry))
            
        # This is kinda bad code but it just introduces
        # a mode where you step through the environment
        # by pressing down a key, instead of it doing it on every frame
        # (it actually just waits to go to next frame until you press a key but we don't talk about this)
        if STEP_WISE_FRAME_ENABLED:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    STEP_WISE_FRAME_ENABLED = False
                    
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    STEP_WISE_FRAME_ENABLED = True
                elif event.key == pygame.K_t:
                    SHOW_TEXT_ENABLED = not SHOW_TEXT_ENABLED
                    
        if STEP_WISE_FRAME_ENABLED:
            frame_surface = font.render(f'Key-wise stepping!', True, (255, 255, 255))
            display.blit(frame_surface, (w-frame_surface.get_width() - 25, 20))
        
        pygame.display.flip()
        clock.tick(FRAME_RATE)
        
        # Reload the latest checkpoint if we die
        #   this is so we can see progress, kinda proggers if you know what I mean *wink wink*
        if timestep.is_last():
            npolicy = checkpointer.reload_latest()