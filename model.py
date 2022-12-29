import os
import tensorflow as tf
from keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import PolicySaver
from TFSnake import PySnakeGameEnv

def build_model(layer_params, num_actions):
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

def get_agent(layer_params, num_actions, environment, learning_rate):
    qnet = build_model(layer_params, num_actions)

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
    return agent

def save_model(agent : dqn_agent.DqnAgent, filename : str):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = PolicySaver(agent.policy)
    saver.save(filename)
    del saver

def load_model(model_name : str):
    policy = tf.compat.v2.saved_model.load(model_name)
    return policy

if __name__ == '__main__':
    get_agent((100, 100), 4, tf_py_environment.TFPyEnvironment(PySnakeGameEnv()))