from __future__ import absolute_import, division, print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import datetime

import reverb
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from environments.snake_game import PySnakeGameEnv
from environments.snake_game import ConvPySnakeGameEnv
from environments.snake_game import AdditiveWhenAppleEatenLifeUpdater
from environments.snake_game import ResetWhenAppleEatenLifeUpdater
from helpers import compute_avg_return
from helpers import create_policy_eval_video
from model import get_agent, get_policy_saver, load_model
from model import create_checkpointer
from model import get_policy_saver
from model.model_config import LinearModelConfig
from model.model_config import ConvModelConfig, ConvLayerParameter

print(f'TF version = {tf.version.VERSION}')

num_iterations = 1_000_000

initial_collect_steps       = 500
collect_steps_per_iteration = 4
replay_buffer_max_length    = 100000

batch_size    = 32
learning_rate = 1e-4
log_interval  = 200

checkpoint_interval = 1000

board_shape = (32, 32)
# train_py_env = PySnakeGameEnv(
#     board_shape,
#     ResetWhenAppleEatenLifeUpdater(400))
# eval_py_env = PySnakeGameEnv(
#     board_shape,
#     ResetWhenAppleEatenLifeUpdater(400))

train_py_env = ConvPySnakeGameEnv(
    board_shape=board_shape,
    life_updater=ResetWhenAppleEatenLifeUpdater(400))
eval_py_env  = ConvPySnakeGameEnv(
    board_shape=board_shape,
    life_updater=ResetWhenAppleEatenLifeUpdater(400))

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# model_config = LinearModelConfig(28, [1024, 256], int(num_actions))
model_config = ConvModelConfig(
    train_py_env.observation_spec().shape,
    [
        ConvLayerParameter('conv2d', 8, (3, 3), (1, 1), (0, 0), 'relu'),
        ConvLayerParameter('conv2d', 16, (5, 5), (1, 1), (0, 0), 'relu'),
        ConvLayerParameter('conv2d', 32, (5, 5), (1, 1), (0, 0), 'relu'),
        ConvLayerParameter('flatten', None, None, None, None, None),
        ConvLayerParameter('dense', 32, None, None, None, 'relu')
    ],
    int(num_actions)
)

global_step = tf.compat.v1.train.get_or_create_global_step()
agent = get_agent(model_config, train_env, learning_rate)

# Reset the environment.
model_name = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# Create the checkpointer
ckpt_dir = os.path.join('ckpts', model_name)
checkpointer = create_checkpointer(
    ckpt_dir=ckpt_dir,
    agent=agent,
    global_step=global_step,
    model_config=model_config,
    max_to_keep=5
)

saver = get_policy_saver(
    agent,
    model_config
)

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

iterator = iter(replay_buffer.as_dataset(
    num_parallel_calls=12,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(5))

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

previous_reward = -np.inf
time_step = train_py_env.reset()
for _ in range(num_iterations):

    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print(f'step = {step:8d}: loss = {train_loss:7.3f}')

    if step % checkpoint_interval == 0:
        episode_model_name = f'policy_iter_{step}'
        episode_model_path = os.path.join('policies', model_name, episode_model_name)
        
        saver.save(episode_model_path)
        checkpointer.save(global_step)
        create_policy_eval_video(
            agent.policy, 
            os.path.join('mp4', model_name, f'step_{step}_'),
            eval_env, 
            eval_py_env, 
            num_episodes=1, 
            fps=60,
            append_score=True)
        print(f'Created checkpoint and evaluation video.')