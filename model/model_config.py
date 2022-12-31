from typing import *
from abc import ABCMeta, abstractmethod
from json import dump, load
from collections import namedtuple

ConvLayerParameter = namedtuple('ConvolutionalLayerParameter', 'type filters kernel_size strides paddings activation')

class ModelConfig(metaclass=ABCMeta):
    def __init___(self, *args, **kwargs):
        pass

    def __repr__(self):
        return str(self)

    @classmethod
    @abstractmethod
    def save_as_json(self, path : str):
        pass

    @staticmethod
    def load_config(path : str):
        with open(path, 'r') as f:
            json_obj = load(f)
        
        typ = json_obj['type']
        klass = globals()[typ]
        m_config = klass(**(json_obj['parameters']))
        return m_config

class LinearModelConfig(ModelConfig):
    def __init__(self, input_dims : int, layer_dims : List[int], output_dims : int):
        self.input_dims = input_dims
        self.layer_dims = layer_dims
        self.output_dims = output_dims

    def __str__(self):
        return f'LinearModelConfig\n\t{self.input_dims}\n\t{self.layer_dims}\n\t{self.output_dims}'

    def save_as_json(self, path : str):
        with open(path, 'w') as f:
            dump({
                "type": self.__class__.__name__,
                'parameters': {
                    "input_dims": self.input_dims,
                    "layer_dims": self.layer_dims,
                    "output_dims": self.output_dims
                }
            }, f)

class ConvModelConfig(ModelConfig):
    def __init__(self, input_dims : Tuple[int, ...], layer_parameters : List[ConvLayerParameter], output_dims : int):
        self.input_dims = input_dims
        assert input_dims[0] % 2 == 1, f'Input dims must be odd'
        # Explicitly cast all elements to ConvLayerParameter so load_config works smoothly
        self.layer_parameters : List[ConvLayerParameter] = [ConvLayerParameter._make(l) for l in layer_parameters]
        self.output_dims = output_dims

    def __str__(self):
        s = self.__class__.__name__
        s += f'\n\tInput dims: {self.input_dims}'
        s += '\n\tLayers:'
        for l in self.layer_parameters:
            s += f'\n\t\t{l}'
        s += f'\n\tOutput dims: {self.output_dims}'
        return s

    def save_as_json(self, path : str):
        with open(path, 'w') as f:
            dump({
                'type': self.__class__.__name__,
                "parameters": {
                    "input_dims": self.input_dims,
                    "layer_parameters": self.layer_parameters,
                    "output_dims": self.output_dims
                }
            }, f)