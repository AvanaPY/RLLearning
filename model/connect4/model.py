from model.model_config import ModelConfig

import tensorflow as tf
import tf_agents
from tf_agents.networks import network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import TFEnvironment
from tf_agents.networks import sequential
from tf_agents.utils import common
import json

class ConnectFourModelConfig(ModelConfig):
    def __init__(self, num_outputs : int):
        self.num_outputs = num_outputs
    
    def save_as_json(self, path : str):
        with open(path, 'w') as f:
            json.dump({
                "type": self.__class__.__name__,
                'parameters': {
                    'num_outputs': self.num_outputs
                }
            }, f)
    
def create_agent(
    config : ConnectFourModelConfig, 
    time_step_spec,
    action_spec) -> tf_agents.agents.DqnAgent:
    
    layers = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256)
    ]
    output_layer = tf.keras.layers.Dense(
        config.num_outputs,
        activation=None,
        name='output'
    )
    net = sequential.Sequential(
        layers + [output_layer]
    )
    net.create_variables(
        input_tensor_spec=time_step_spec.observation
    )
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10_000,
        decay_rate=0.99,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule
    )
    td_errors_loss_fn = common.element_wise_squared_loss
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        q_network=net,
        optimizer=optimizer,
        td_errors_loss_fn=td_errors_loss_fn,
        train_step_counter=train_step_counter
    )
    return agent