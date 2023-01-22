
import os
from tf_agents.policies import PolicySaver
from tf_agents.agents.dqn import dqn_agent

from model.model_config import ModelConfig

class MyPolicySaver:
    def __init__(self, agent : dqn_agent.DqnAgent, 
                 model_config : ModelConfig,
                 policy_name : str,
                 config_name : str):
        self._agent = agent
        self._mc = model_config
        self._policy_name = policy_name
        self._config_name = config_name
        self._policy_saver = PolicySaver(agent.policy)

    def save(self, path : str):
        self._policy_saver.save(os.path.join(path, self._policy_name))
        self._mc.save_as_json(os.path.join(path, self._config_name))