import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import numpy as np
import tensorflow as tf
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

STEP_WISE_FRAME_ENABLED = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framerate', type=int , default=30, dest='framerate')
    parser.add_argument('--board-size', type=int , default=32, dest='boardsize')
    parser.add_argument('--window-size', type=int, default=640, dest='winsize')
    parser.add_argument('--model-folder', type=str, default='saved_policies', dest='modelfolder')
    parser.add_argument('--model-name', type=str , default="Sture2", dest='modelname')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='cpu')
    args = parser.parse_args()

    device = args.device
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            print(f'No GPU support! Exiting')
            exit(0)
            
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

    model_config, env, tf_env, agent, checkpointer = load_checkpoint_from_path(ckpt_path, None)
    agent._q_network.trainable = False
    agent._q_network.summary()
    env._game._life_updater = ResetWhenAppleEatenLifeUpdater(200)
    global_step = tf.compat.v1.train.get_global_step()
    step_counter = agent.train_step_counter
    policy = agent.policy

    # Play the policy
    timestep = tf_env.reset()
    running = True
    while running:
        
        state = timestep.observation
        out = agent._q_network(state)[0][0]
        action = np.argmax(out)
        expected_reward = out.numpy()
        action = tf.constant(action, shape=(1,),dtype=np.int32)
        
        # action = policy.action(timestep).action
        # expected_reward = np.nan
        
        timestep = tf_env.step(action)
        render = env.render(window_size=args.winsize, rotate=True)

        new_state = timestep.observation
        new_preds = agent._q_network(new_state)[0][0]
        new_rewards = new_preds.numpy()

        pygame.surfarray.blit_array(surf, render)

        display.fill((0, 0, 0))
        display.blit(surf, (0, 0))

        train_steps = step_counter.numpy()
        steps_surface  = font.render(f'Total train steps: {train_steps}', True, (255, 255, 255))
        display.blit(steps_surface, (20, h - 20))
        
        # Render rewards
        x = 20
        y = h - 80
        reward_surface = font.render(f'Expected reward:', True, (255, 255, 255))
        display.blit(reward_surface, (x, y))
        
        x += reward_surface.get_width() + 30
        
        c1 = pygame.Color(255, 0, 0)
        c2 = pygame.Color(0, 255, 0)
        c = c1.lerp(c2, 0.5)
        
        min_rew = np.min(new_rewards)
        max_rew = np.max(new_rewards)
        rel_poses = [
            (20, 0),
            (0, 20),
            (-20, 0),
            (0, -20)
        ]
        for value, (rx, ry) in zip(new_rewards, rel_poses):
            pct =  (value - min_rew) / (max_rew - min_rew)
            c = c1.lerp(c2, pct)
            value_surf = font.render(f'{value:4.1f}', True, c)
            display.blit(value_surf, (x + rx, y + ry))
            
            
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
                    
        if STEP_WISE_FRAME_ENABLED:
            frame_surface = font.render(f'Key-wise stepping!', True, (255, 255, 255))
            display.blit(frame_surface, (w-frame_surface.get_width() - 25, 20))
        
        pygame.display.flip()
            
        clock.tick(FRAME_RATE)

        if timestep.is_last():
            policy = checkpointer.reload_latest()