import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import datetime
import time

import tensorflow as tf
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.drivers import py_driver
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from environments.connect4.game import PyConnectFourGame
from model.connect4.model import create_agent
from model.connect4.model import ConnectFourModelConfig

import reverb
## Define constants

num_iterations = 1_000_000

log_interval  = 200
save_interval = 1000

replay_buffer_max_size  = 100_000
initial_collect_steps   = 1_000
collect_steps           = 1

batch_size              = 32
num_parallel_calls      = 4
prefetch_size           = 10
################################################

# Create environments
py_train_env = PyConnectFourGame()
tf_train_env = tf_py_environment.TFPyEnvironment(py_train_env)

# Create agent
config = ConnectFourModelConfig(num_outputs=7)
agent = create_agent(config,
                     tf_train_env.time_step_spec(),
                     tf_train_env.action_spec())

# Create a reverb replay buffer and server
replay_buffer_signature = tensor_spec.add_outer_dim(
    tensor_spec.from_spec(
        agent.collect_data_spec))

reverb_table = reverb.Table(
    'uniform_table',
    max_size=replay_buffer_max_size,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature
)

reverb_server = reverb.Server([reverb_table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name='uniform_table',
    sequence_length=2,
    local_server=reverb_server
)

reverb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    'uniform_table', 
    sequence_length=2
)

# Initialize some steps with a RandomTFPolicy

py_driver.PyDriver(
    py_train_env,
    py_tf_eager_policy.PyTFEagerPolicy(
        random_tf_policy.RandomTFPolicy(tf_train_env.time_step_spec(),
                                        tf_train_env.action_spec()),
        use_tf_function=True
    ),
    [reverb_observer],
    max_steps=initial_collect_steps
).run(py_train_env.reset())

# Define a collect driver and dataset iterator for the Reverb buffer
iterator = iter(replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_parallel_calls=num_parallel_calls,
    num_steps=2
).prefetch(prefetch_size))

collect_driver = py_driver.PyDriver(
    py_train_env,
    py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True
    ),
    [reverb_observer],
    max_steps=collect_steps
)

saver = policy_saver.PolicySaver(
    policy=agent.policy)

# Play
previous_reward = -np.inf
time_step = py_train_env.reset()
start_time = datetime.datetime.now()

# Create a checkpointer
model_save_name = f'c4_policy_{start_time.strftime("%Y_%m_%d__%H_%M_%S")}'
model_path = os.path.join('connect4_policies', model_save_name)
checkpointer = common.Checkpointer(
    model_path
)

global_step = tf.compat.v1.train.get_or_create_global_step()
print(f'Starting training at {start_time.strftime("%Y/%m/%d %H:%M:%S")}')
for iteration in range(1, num_iterations + 1):
    time_step, _ = collect_driver.run(time_step)
    
    exp, _ = next(iterator)
    train_loss = agent.train(exp).loss
    
    step = agent.train_step_counter.numpy()
    
    if step % log_interval == 0:
        print(f'[Iteration #{iteration:>11,d}] :: Train step {step:>11,d}: Loss = {train_loss}')
        
    if step % save_interval == 0:
        checkpointer.save(global_step)
        saver.save(
            os.path.join(model_path, 'policy_model')
        )
        print(f'Saving policy at training step {step}: {model_path}')
