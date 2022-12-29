import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--framerate', type=int, default=30, dest='framerate')
    parser.add_argument('--boardsize', type=int, default=32, dest='boardsize')
    parser.add_argument('--windowsize', type=int, default=640, dest='winsize')
    args = parser.parse_args()

    FRAME_RATE = args.framerate
    boardsize = (args.boardsize, args.boardsize)
    windowsize = (args.winsize, args.winsize)

    assert args.winsize % args.boardsize == 0, f'Window Size must be a multiple of board size'
    
    import tensorflow as tf
    import pygame
    from tf_agents.environments import tf_py_environment
    from tf_agents.environments import py_environment
    from tf_agents.environments import tf_environment

    from model import load_model
    from TFSnake import PySnakeGameEnv
    from helpers import create_policy_eval_video

    pygame.init()
    policy_name = os.path.join('saved_policies', 'Sture2')
    policy = load_model(policy_name)
    env = PySnakeGameEnv(boardsize)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    display = pygame.display.set_mode(windowsize)
    surf = pygame.surface.Surface(display.get_size())
    clock = pygame.time.Clock()

    timestep = tf_env.reset()
    running = True
    while running:
        action = policy.action(timestep)
        timestep = tf_env.step(action.action)
        render = env.render(args.winsize)

        pygame.surfarray.blit_array(surf, render)

        display.fill((0, 0, 0))
        display.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        clock.tick(FRAME_RATE)