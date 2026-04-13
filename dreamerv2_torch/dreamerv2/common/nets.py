import re

import numpy as np
import torch

import commom


class EnsembleRSSM(commom.Module):

    def __init__(self,
                 ensemble=5,
                 stoch=30,
                 deter=200,
                 hidden=200,
                 discrete=False,
                 act='elu',
                 norm='none',
                 std_act='softplus',
                 min_std=0.1):
        super().__init__()
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._cell = GRUCell(self._deter, norm=True)
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = torch.get_default_dtype(torch.float32)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete], dtype),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype))
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], dtype),
                std=torch.ones([batch_size, self._stoch], dtype),
                stoch=torch.zeros([batch_size, self._stoch], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype))
        return state



