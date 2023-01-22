import tf_agents
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.specs.tensor_spec import TensorSpec

def create_model(input_spec : TensorSpec):
    vnet = ValueNetwork(
        input_tensor_spec=input_spec,
        conv_layer_params=[(24, 5, 1)],
        fc_layer_params=(256, 1024, 1024, 1024, 256)
    )
    
    return vnet