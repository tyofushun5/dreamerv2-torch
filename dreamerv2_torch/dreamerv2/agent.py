import torch

import common
import expl


class Agent(common.Module):

    def __init__(self, config, obs_space, act_space, step):
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        # self.register_buffer('tfstep', torch.tensor(int(self.step), dtype=torch.long))
        self.wm = WorldModel(config, obs_space, self.step)
        self._task_behavior = ActorCritic(config, self.act_space, self.step)
        if config.expl_behavior == 'greedy':
            self._expl_behavior = self._task_behavior
        else:
            self._expl_behavior = getattr(expl, config.expl_behavior)(
                self.config, self.act_space, self.wm, self.tfstep,
                lambda seq: self.wm.heads['reward'](seq['feat']).mode())

    def policy(self, obs, state=None, mode='train'):
        pass

    def train(self, data, state=None):
        pass

    def report(self, data, state=None):
        pass

class WorldModel(common.Module):

    def __init__(self, config, obs_space, step):
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.step = step
        # self.register_buffer('tfstep', torch.tensor(int(self.step), dtype=torch.long))
        self.encoder = Encoder(config, obs_space)
        self.dyn = RSSM(config, self.encoder.feat_dim, step)
        self.heads = torch.nn.ModuleDict({
            'reward': MLP(config, self.dyn.stoch_dim + self.dyn.deter_dim, 1),
            'continue': MLP(config, self.dyn.stoch_dim + self.dyn.deter_dim, 1),
            'obs': Decoder(config, obs_space, self.encoder.feat_dim,
                           self.dyn.stoch_dim + self.dyn.deter_dim),
        })

    def train(self, data, state=None):
        pass

    def loss(self, data, state=None):
        pass

    def imagine(self, data, state=None):
        pass

    def preprocess(self, obs):
        pass

    def video_pred(self, data, key):
        pass

