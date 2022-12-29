from __future__ import absolute_import, division, print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import time
import datetime
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

from TFSnake import PySnakeGameEnv
from helpers import compute_avg_return
from helpers import create_policy_eval_video
from model import get_agent, save_model, load_model
print(f'TF version = {tf.version.VERSION}')
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

num_iterations = 100_000 # @param {type:"integer"}

initial_collect_steps       = 1000      # @param {type:"integer"}
collect_steps_per_iteration = 1        # @param {type:"integer"}
replay_buffer_max_length    = 100000    # @param {type:"integer"}

batch_size    = 64      # @param {type:"integer"}
learning_rate = 1e-3    # @param {type:"number"}
log_interval  = 200     # @param {type:"integer"}

num_eval_episodes = 5     # @param {type:"integer"}
eval_interval     = 2000  # @param {type:"integer"}

board_shape = (32, 32)
train_py_env = PySnakeGameEnv(board_shape=board_shape)
eval_py_env = PySnakeGameEnv(board_shape=board_shape)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (256, 256)
action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

agent = get_agent(fc_layer_params, num_actions, train_env, learning_rate)

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

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

py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

iterator = iter(replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(1))

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
# This has a very good speedup tbh
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
returns = [
    compute_avg_return(eval_env, agent.policy, num_eval_episodes)[0]
]

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

# Reset the environment.
model_name = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
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

    if step % eval_interval == 0:
        episode_model_name = f'policy_iter_{step}'
        episode_model_path = os.path.join('policies', model_name, episode_model_name)

        eval_returns = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        avg_return = np.mean(np.array(eval_returns))
        returns.append(avg_return)

        if avg_return < previous_reward:
            episode_diff_arrow = '▼'
        elif avg_return == previous_reward:
            episode_diff_arrow = '-'
        else:
            episode_diff_arrow = '▲'
        previous_reward = avg_return
        print(f'step = {step}: Returns = {eval_returns} ({avg_return:6.2f} {episode_diff_arrow})')
        
        save_model(agent, episode_model_path)
        create_policy_eval_video(
            agent.policy, 
            os.path.join('mp4', model_name, f'step_{step}_'),
            eval_env, 
            eval_py_env, 
            num_episodes=1, 
            fps=60,
            append_score=True)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
print(f'Safe to exit')
plt.show(block=True)