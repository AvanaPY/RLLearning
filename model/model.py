import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from keras import layers
import keras
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tf_agents.networks import sequential, Network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import PolicySaver
try:
    from utils.my_policy_saver import MyPolicySaver
    from utils.my_checkpointer import MyCheckpointer
    from model_config import ModelConfig
    from model_config import LinearModelConfig
    from model_config import ConvModelConfig, ConvLayerParameter
except ModuleNotFoundError:
    from utils.my_policy_saver import MyPolicySaver
    from utils.my_checkpointer import MyCheckpointer
    from .model_config import ModelConfig, LinearModelConfig, ConvModelConfig, ConvLayerParameter

def build_model(layer_params, num_actions):
    print(f'Building sequential dense model...')
    def dense_layer(num_units):
        activation = None
        return tf.keras.layers.Dense(
            num_units,
            activation=activation)

    dense_layers = []
    for num_units in layer_params:
        dense_layers.append(dense_layer(num_units))
        dense_layers.append(layers.LeakyReLU(alpha=0.1))

    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None)

    qnet = sequential.Sequential(dense_layers + [q_values_layer])
    return qnet

def build_conv_model(config : ConvModelConfig):
    print(f'Building convolutional model...')
    layers = []
    for l in config.layer_parameters:
        if l.type == 'conv2d':
            layers.append(tf.keras.layers.Conv2D(
                l.filters, 
                l.kernel_size, 
                l.strides, activation=l.activation))
        elif l.type == 'flatten':
            layers.append(tf.keras.layers.Flatten())
        elif l.type == 'dense':
            layers.append(tf.keras.layers.Dense(
                l.filters, activation=l.activation))
        else:
            raise RuntimeError(f'Unknown layer: {l.type}')

    layers.append(
        tf.keras.layers.Dense(config.output_dims, activation=None)
    )
    qnet = sequential.Sequential(layers)
    return qnet

def get_agent(model_config : ModelConfig, environment, learning_rate, return_q_net:bool=False):
    assert isinstance(model_config, ModelConfig), 'Parameter model_config must inherit from ModelConfig'
    assert isinstance(environment, tf_py_environment.TFPyEnvironment), 'Environment must be of type TFPyEnvironment'
    assert learning_rate > 0, 'Learning rate must be greater than zero (0)'
    
    if isinstance(model_config, LinearModelConfig):
        num_actions  = model_config.output_dims
        assert num_actions > 0, f'Number of actions must be greater than zero (0)'
        
        layer_params = model_config.layer_dims
        assert len(layer_params) > 0, 'There must be one (1) or more layers'

        qnet = build_model(layer_params, num_actions)
    elif isinstance(model_config, ConvModelConfig):
        qnet = build_conv_model(model_config)
    else:
        raise RuntimeError(f'Unknown config: {model_config}')

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        environment.time_step_spec(),
        environment.action_spec(),
        q_network=qnet,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )
    agent.initialize()
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)
    if return_q_net:
        return agent, qnet
    return agent

def get_policy_saver(agent : dqn_agent.DqnAgent, model_config : ModelConfig):
    return MyPolicySaver(agent, model_config)

def load_model(model_name : str):
    policy = tf.compat.v2.saved_model.load(model_name)
    return policy

def create_checkpointer(ckpt_dir:str, 
                        agent, 
                        global_step:tf.Tensor,
                        model_config:ModelConfig,
                        max_to_keep:int=1) -> MyCheckpointer:
    checkpointer = MyCheckpointer(
        ckpt_dir=ckpt_dir,
        agent=agent,
        global_step=global_step,
        max_to_keep=max_to_keep,
        model_config=model_config
    )
    return checkpointer

if __name__ == '__main__':
    config = ConvModelConfig(
        (32, 32), 
        [
            ConvLayerParameter('conv2d', 3, (5, 5), (1, 1), (0, 0), 'relu'),
            ConvLayerParameter('conv2d', 3, (5, 5), (1, 1), (0, 0), 'relu'),
            ConvLayerParameter('conv2d', 3, (5, 5), (1, 1), (0, 0), 'relu'),
            ConvLayerParameter('conv2d', 3, (5, 5), (1, 1), (0, 0), 'relu'),
            ConvLayerParameter('flatten', None, None, None, None, None),
            ConvLayerParameter('dense'  , 256, None, None, None, 'relu')
        ],
        4)
    config.save_as_json('config.json')
    model = build_conv_model(config)
    a = model(np.random.random(size=(1, 32, 32, 1)))
    print(a[0].shape)


    config = ModelConfig.load_config('config.json')
    m = build_conv_model(config)
    a = m(np.random.random(size=(1, 32, 32, 1)))
    print(a[0].shape)
    m.summary()