from __future__ import absolute_import, division, print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import datetime
import time

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
from model.res_model import StureModel, ResidualModelConfig
from utils.helpers import load_checkpoint_from_path

print(f'TF version = {tf.version.VERSION}')

num_iterations = 10_000_000

initial_collect_steps       = 1_000
collect_steps_per_iteration = 1
replay_buffer_max_length    = 100_000

batch_size    = 8
learning_rate = 1e-6
decay_steps   = 10_000
decay_rate    = 0.95
NUM_PAR_CALLS = tf.data.AUTOTUNE
PREFETCH_SIZE = tf.data.AUTOTUNE

log_interval  = 100
checkpoint_interval = 1_000
eval_video_interval = 5_000
board_shape = (32, 32)
conv_observation_spec_shape = (33, 33, 1) 

# StureModel parameters
num_residual_layers = 5    # Our residual layers
num_residual_filters = 128
residual_kernel_size = 5
residual_strides = 1        
num_filters = num_residual_filters # Initial Conv2d layer
kernel_size = 7
strides = 1

# Path constants
MODELS_FOLDER_PATH      = 'models'                  # This will create a folder structure of:  models
CHECKPOINT_FOLDER_NAME  = 'ckpts'                   #                                               |--ckpts
                                                    #                                               |       |--ckpt
POLICIES_FOLDER_NAME    = 'policies.snake'          #                                               |--policies.snake
                                                    #                                               |       |--policy
MP4_FOLDER_NAME         = 'mp4'                     #                                               |--mp4
                                                    #                                               |       |--idkdjlfkdj.mp4

# check for gpu 
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print(f'No GPU support!')

# Build or load in the model

MODEL_LOAD_NAME = '2023_01_24__17_08_52'
MODEL_LOAD_PATH = os.path.join(MODELS_FOLDER_PATH, MODEL_LOAD_NAME)
MODEL_LOAD_CKPT_PATH = os.path.join(MODEL_LOAD_PATH, CHECKPOINT_FOLDER_NAME)
TO_LOAD_MODEL = os.path.exists(MODEL_LOAD_PATH)

if TO_LOAD_MODEL:
    model_name = MODEL_LOAD_NAME
else:
    model_name = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# Path variables
model_path   = os.path.join(MODELS_FOLDER_PATH, model_name)
ckpt_dir     = os.path.join(model_path, CHECKPOINT_FOLDER_NAME)
policies_dir = os.path.join(model_path, POLICIES_FOLDER_NAME)
mp4_dir      = os.path.join(model_path, MP4_FOLDER_NAME)

# Load or create a model
if TO_LOAD_MODEL:
    print(f'Loading model...')
    model_config, train_py_env, train_env, agent, checkpointer \
        = load_checkpoint_from_path(MODEL_LOAD_PATH, learning_rate=learning_rate) 
    global_step = tf.compat.v1.train.get_global_step()
    eval_py_env = train_py_env.deep_copy()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
else:
    print(f'Creating new model.')
    train_py_env = ConvPySnakeGameEnv(
        board_shape=board_shape,
        observation_spec_shape=conv_observation_spec_shape,
        life_updater=ResetWhenAppleEatenLifeUpdater(200),
        discount=0.9,
        reward_on_death=-20,
        reward_on_apple= 20,
        reward_on_step_closer = 1,
        reward_on_step_further=-1    
    )
    eval_py_env = train_py_env.deep_copy()
    train_py_env.save_config_to_folder(model_path)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # model_config = LinearModelConfig(28, [1024, 256], int(num_actions))
    # model_config = ConvModelConfig(
    #     train_py_env.observation_spec().shape,
    #     [
    #         ConvLayerParameter('conv2d', 4, (5, 5), (2, 2), (0, 0), 'relu'),
    #         ConvLayerParameter('conv2d', 16, (3, 3), (2, 2), (0, 0), 'relu'),
    #         ConvLayerParameter('conv2d', 16, (3, 3), (2, 2), (0, 0), 'relu'),
    #         ConvLayerParameter('conv2d', 128, (2, 2), (2, 2), (0, 0), 'relu'),
    #         ConvLayerParameter('flatten', None, None, None, None, None),
    #         ConvLayerParameter('dense', 256, None, None, None, 'relu'),
    #         ConvLayerParameter('dense', 1024, None, None, None, 'relu'),
    #         ConvLayerParameter('dense', 256, None, None, None, 'relu'),
    #     ],
    #     int(num_actions)
    # )
    model_config = ResidualModelConfig(
        num_residual_layers=num_residual_layers,
        num_residual_filters=num_residual_filters,
        residual_kernel_size=residual_kernel_size,
        residual_strides=residual_strides,
        num_filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        input_dims=train_py_env.observation_spec().shape
    )

    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent, qnet = get_agent(model_config, 
                            train_env, 
                            learning_rate, 
                            decay_steps=decay_steps,
                            decay_rate=decay_rate,
                            decay_stair_case=True,
                            return_q_net=True)
    qnet.summary()

    
# 

checkpointer = create_checkpointer(
    ckpt_dir=ckpt_dir,
    agent=agent,
    global_step=global_step,
    model_config=model_config,
    max_to_keep=5
)

saver = get_policy_saver(
    agent,
    model_config,
    policy_folder_name='policy',
    config_file_name='config.json'
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

print(f'Collecting initial data... ({initial_collect_steps} steps)')
py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())
print(f'Done.')

iterator = iter(replay_buffer.as_dataset(
    num_parallel_calls=NUM_PAR_CALLS,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(PREFETCH_SIZE))

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

start_time_str = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
POLICIES_MODEL_NAME = os.path.join(POLICIES_FOLDER_NAME, model_name)
MP4_MODEL_NAME = os.path.join(MP4_FOLDER_NAME, model_name)

print(f'Training agent {model_name} :: Started at {start_time_str}')
print(f'\tCheckpoint:  {ckpt_dir}')
print(f'\tPolicies:    {policies_dir}')
print(f'\tMP4s:        {mp4_dir}')
print(f'\tTotal train steps: {agent.train_step_counter.numpy():11,d}')

previous_reward = -np.inf
time_step = train_py_env.reset()
for iteration in range(1, num_iterations + 1):

    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print(f'[Iteration {iteration:>11,d} / {num_iterations:>11,d}]: Step {step:>11,d}: loss = {train_loss:7.3f}')

    if step % checkpoint_interval == 0:
        episode_model_name = f'policy_iter_{step}'
        episode_model_path = os.path.join(policies_dir, episode_model_name)
        
        # saver.save(episode_model_path)
        checkpointer.save(global_step)
        print(f'Created checkpoint at step {step}...')
    
    if step % eval_video_interval == 0:
        print(f'Generating evaluation video...')
        create_policy_eval_video(
            agent.policy, 
            os.path.join(mp4_dir, f'step_{step}_'),
            eval_env, 
            eval_py_env, 
            num_episodes=1, 
            fps=30,
            append_score=True)