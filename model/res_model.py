from typing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import keras
import tensorflow as tf
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.networks import sequential
from tf_agents.specs.array_spec import BoundedArraySpec
from collections import namedtuple
from model.model_config import ModelConfig
from json import dump

class ResidualModelConfig(ModelConfig):
    def __init__(self, 
                 num_residual_layers : int, 
                 num_residual_filters : int,
                 residual_kernel_size : int,
                 residual_strides : int,
                 num_filters : int, 
                 kernel_size : int,
                 strides : int,
                 input_shape : Tuple[int]):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.num_residual_filters = num_residual_filters
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.residual_kernel_size = residual_kernel_size
        self.residual_strides = residual_strides
        self.input_shape = input_shape        
                
    def __str__(self):
        s = f'ResidualModelConfig'
        s += f'\n\tFilters: {self.num_filters}'
        s += f'\n\tKernel:  {self.kernel_size}'
        s += f'\n\tStrides: {self.strides}'
        s += f'\n\tRes layers: {self.num_residual_layers}'
        s += f'\n\tResidaul Kernel Sizes: {self.residual_kernel_size}'
        s += f'\n\tResidual strides: {self.residual_strides}'
        s += f'\n\t{self.input_shape}'
        
        return s
    
    def save_as_json(self, path : str):
        with open(path, 'w') as f:
            dump({
                "type": self.__class__.__name__,
                'parameters': {
                    "num_residual_layers" : self.num_residual_layers,
                    "num_residual_filters" : self.num_residual_filters,
                    "num_filters" : self.num_filters,
                    "kernel_size" : self.kernel_size,
                    "strides" : self.strides,
                    "residual_kernel_size" : self.residual_kernel_size,
                    "residual_strides" : self.residual_strides,
                    "input_shape" : self.input_shape,
                }
            }, f)
            
class StureResidualLayer(keras.layers.Layer):
    def __init__(self, num_filters : int, kernel_size : int = 3, strides : int = 1):
        super().__init__()
        self._conv2d1 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same'
        )
        self._bn1 = tf.keras.layers.BatchNormalization()
        self._relu1 = tf.keras.layers.ReLU()
        
        self._conv2d2 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same'
        )
        self._bn2 = tf.keras.layers.BatchNormalization()
        self._relu2 = tf.keras.layers.ReLU()
        
    def call(self, x):
        x1 = self._conv2d1(x)
        x1 = self._bn1(x1)
        x1 = self._relu1(x1)
        
        x1 = self._conv2d2(x1)
        x1 = self._bn2(x1)
        
        x1 = x1 + x
        x1 = self._relu2(x1)
        return x1

class StureValueHead(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        self._conv2d1 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same'
        )
        self._bn1 = tf.keras.layers.BatchNormalization()
        self._relu1 = tf.keras.layers.ReLU()
        self._flatten = tf.keras.layers.Flatten()
        
        self._dense1 = tf.keras.layers.Dense(32, activation='relu')
        self._dense2 = tf.keras.layers.Dense(1, activation='tanh')
        
    def call(self, x):
        x = self._conv2d1(x)
        x = self._bn1(x)
        x = self._relu1(x)
        x = self._flatten(x)
        
        x = self._dense1(x)
        x = self._dense2(x)
        return x

class SturePolicyHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._conv2d1 = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid'
        )
        self._bn1 = tf.keras.layers.BatchNormalization()
        self._relu1 = tf.keras.layers.ReLU()
        self._flatten = tf.keras.layers.Flatten()
        self._dense = tf.keras.layers.Dense(
            4, 
            activation=None
        )
    
    def call(self, x):
        x = self._conv2d1(x)
        x = self._bn1(x)
        x = self._relu1(x)
        x = self._flatten(x)
        x = self._dense(x)
        return x

class StureModel(network.Network):
    def __init__(self, 
                 num_residual_layers : int, 
                 num_residual_filters : int,
                 num_init_filters : int,
                 input_tensor_spec : tf_agents.specs.ArraySpec,
                 init_kernel_size : int = 3,
                 init_strides : int = 1,
                 residual_kernel_size : int = 5,
                 residual_strides = 1,
                 name : str = 'SigmaSture'):
        super().__init__(name=name, 
                         input_tensor_spec=input_tensor_spec)
        self._conv1 = tf.keras.layers.Conv2D(
            filters=num_init_filters,
            kernel_size=init_kernel_size,
            strides=init_strides,
            padding='same'
        )
        self._bn1 = tf.keras.layers.BatchNormalization()
        self._relu1 = tf.keras.layers.ReLU()
        self._residual_layers = [
            StureResidualLayer(num_residual_filters, residual_kernel_size, residual_strides) for _ in range(num_residual_layers)
        ]
        # self._vh = StureValueHead()
        self._ph = SturePolicyHead()
                
    def call(self, observations, step_type=(), network_state=()):
        
        x = self._conv1(observations)
        x = self._bn1(x)
        x = self._relu1(x)
        
        for res_layer in self._residual_layers:
            x = res_layer(x)
            
        # value = self._vh(x)
        policy = self._ph(x)
        return policy, network_state

if __name__ == '__main__':
    input_tensor_spec = BoundedArraySpec(
        shape=(32, 32, 1), dtype=np.float32,
        minimum=-1, maximum=1
    )

    q_model = StureModel(
        num_residual_layers=3,
        num_filters=64,
        input_tensor_spec=input_tensor_spec
    )
    q_model.create_variables()
    q_model.summary()

    np.random.seed(0)
    a = np.random.uniform(size=(1, 32, 32, 1), low=-1, high=1)
    print(f'Input shape: {a.shape}')
    print(f'Input std:   {np.std(a)}')

    policy, _ = q_model(a)
    print(f'Policy: {policy}')