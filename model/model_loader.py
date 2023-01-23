import os
import tensorflow as tf
import numpy as np
from model.model_config import ModelConfig, LinearModelConfig, ConvModelConfig
from model.model import build_linear_model, build_conv_model
from model.res_model import StureModel, ResidualModelConfig
from tf_agents.environments import tf_py_environment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

def get_agent(config : ModelConfig, 
              environment, 
              learning_rate, 
              decay_steps : int = None,
              decay_rate  : float = 0.99,
              decay_stair_case : bool = True,
              return_q_net:bool=False):
    assert isinstance(config, ModelConfig), 'Parameter model_config must inherit from ModelConfig'
    assert isinstance(environment, tf_py_environment.TFPyEnvironment), 'Environment must be of type TFPyEnvironment'
    assert learning_rate > 0, 'Learning rate must be greater than zero (0)'
    
    if isinstance(config, LinearModelConfig):
        num_actions  = config.output_dims
        assert num_actions > 0, f'Number of actions must be greater than zero (0)'
        
        layer_params = config.layer_dims
        assert len(layer_params) > 0, 'There must be one (1) or more layers'

        qnet = build_linear_model(layer_params, num_actions)
    elif isinstance(config, ConvModelConfig):
        qnet = build_conv_model(config)
    elif isinstance(config, ResidualModelConfig):
        config : ResidualModelConfig = config
        
        input_tensor_spec = BoundedArraySpec(
            shape=config.input_dims, dtype=np.float32,
            minimum=-1, maximum=1
        )
        qnet = StureModel(
            num_residual_layers=config.num_residual_layers,
            num_residual_filters=config.num_residual_filters,
            residual_kernel_size=config.residual_kernel_size,
            residual_strides=config.residual_strides,
            num_init_filters=config.num_filters,
            init_kernel_size=config.kernel_size,
            init_strides=config.strides,
            input_tensor_spec=input_tensor_spec
        )
    else:
        raise RuntimeError(f'Unknown config: {config.__class__}')

    decay_steps = decay_steps if decay_steps else np.inf
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=decay_stair_case
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
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