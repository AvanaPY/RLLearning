import os
import tensorflow as tf
from tf_agents.utils import common

try:
    from model_config import ModelConfig
except ModuleNotFoundError:
    from .model_config import ModelConfig

class MyCheckpointer:
    def __init__(self,  ckpt_dir : str, 
                        agent, 
                        global_step, 
                        max_to_keep : int,
                        model_config : ModelConfig):
        self._dir = ckpt_dir
        self._model_config = model_config
        self._checkpointer = common.Checkpointer(
            os.path.join(ckpt_dir, 'ckpt'), 
            agent=agent,
            policy=agent.policy,
            global_step=global_step,
            max_to_keep=max_to_keep)

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self._model_config.save_as_json(os.path.join(self._dir, 'config.json'))
        
    def save(self, global_step):
        """
            Wrappper for tf_agent.utils.common.Checkpointer.save()
        """
        self._checkpointer.save(global_step)

class MyCheckpointLoader:
    def __init__(self,  ckpt_dir : str, 
                        agent, 
                        global_step, 
                        max_to_keep : int,
                        model_config : ModelConfig):
        self._dir = ckpt_dir
        self._agent = agent
        self._policy = agent.policy
        self._model_config = model_config
        self._global_step = global_step
        self._max_to_keep = max_to_keep
        self._checkpointer = common.Checkpointer(
            os.path.join(ckpt_dir, 'ckpt'), 
            agent=self._agent,
            policy=self._policy,
            global_step=global_step,
            max_to_keep=max_to_keep)
        self.initialize_or_restore()

        if not os.path.exists(os.path.join(self._dir, 'config.json')):
            raise RuntimeError(f'config.json file is missing')

        self._ckpt_last_modified = self._get_ckpt_last_modified()

    def _get_ckpt_last_modified(self):
        path = os.path.join(self._dir, 'ckpt', 'ckpt-0.index')
        if not os.path.exists(path):
            return -1
        time = os.path.getmtime(path)
        return time

    def initialize_or_restore(self):
        """
            Wrappper for tf_agent.utils.common.Checkpointer.initialize_or_restore()
        """
        self._checkpointer.initialize_or_restore()
        self._global_step = tf.compat.v1.train.get_global_step()

    def reload_latest(self):
        last_modified = self._get_ckpt_last_modified()
        if last_modified != self._ckpt_last_modified:
            self._ckpt_last_modified = last_modified

            self._checkpointer = common.Checkpointer(
                os.path.join(self._dir, 'ckpt'), 
                agent=self._agent,
                policy=self._policy,
                global_step=self._global_step,
                max_to_keep=self._max_to_keep)
            self.initialize_or_restore()
        return self._policy
        