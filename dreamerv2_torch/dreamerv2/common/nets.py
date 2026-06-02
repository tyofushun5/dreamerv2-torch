import re

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli, Independent, Normal

import common


class EnsembleRSSM(common.Module):

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

    def observe(self, embed, action, is_first, state=None):
        pass

    def imagine(self, embed, action, is_first, state=None):
        pass

    def get_feat(self, state):
        pass

    def get_dist(self, state):
        pass

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        pass

    def img_step(self, prev_state, prev_action, sample=True):
        pass

    def _suff_stats_ensemble(self, inp):
        pass

    def _suff_stats_layer(self, name, x):
        pass

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        pass


class Encoder(common.Module):

    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        super().__init__()
        self.shapes = shapes
        self.cnn_keys = []
        self.mlp_keys = []
        for k, v in shapes.items():
            if re.match(cnn_keys, k) and len(v) == 3:
                self.cnn_keys.append(k)
        for k, v in shapes.items():
            if re.match(mlp_keys, k) and len(v) == 1:
                self.mlp_keys.append(k)
        print('Encoder CNN inputs:', list(self.cnn_keys))
        print('Encoder MLP inputs:', list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def forward(self, data):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[:-len(shape)]
        data = {
            k: v.reshape((-1,) + tuple(v.shape)[len(batch_dims):])
            for k, v in data.items()}
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = torch.concat(outputs, -1)
        return output.reshape(tuple(batch_dims) + tuple(output.shape[1:]))

    def _cnn(self, data):
        x = torch.concat(list(data.values()), -1)
        x = x.to(dtype=torch.get_default_dtype())
        x = x.permute(0, 3, 1, 2).contiguous()
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** i * self._cnn_depth
            x = self.get(f'conv{i}', nn.Conv2d, x.shape[1], depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        return x.reshape(x.shape[0], -1)

    def _mlp(self, data):
        x = torch.concat(list(data.values()), -1)
        x = x.to(dtype=torch.get_default_dtype())
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f'dense{i}', nn.Linear, x.shape[-1], width)(x)
            x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        return x


class Decoder(common.Module):

    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        super().__init__()
        self._shapes = shapes
        self.cnn_keys = []
        self.mlp_keys = []
        for k, v in shapes.items():
            if re.match(cnn_keys, k) and len(v) == 3:
                self.cnn_keys.append(k)
        for k, v in shapes.items():
            if re.match(mlp_keys, k) and len(v) == 1:
                self.mlp_keys.append(k)
        print('Decoder CNN outputs:', list(self.cnn_keys))
        print('Decoder MLP outputs:', list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def forward(self, features):
        features = features.to(dtype=torch.get_default_dtype())
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        x = self.get('convin', nn.Linear, features.shape[-1], 32 * self._cnn_depth)(features)
        x = x.reshape(-1, 32 * self._cnn_depth, 1, 1)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(channels.values()), lambda x: x, 'none'
            x = self.get(f'conv{i}', nn.ConvTranspose2d, x.shape[1], depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, norm)(x)
            x = act(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(tuple(features.shape[:-1]) + tuple(x.shape[1:]))
        means = torch.split(x, list(channels.values()), dim=-1)
        return {
            key: Independent(Normal(mean, torch.ones_like(mean)), 3)
            for (key, shape), mean in zip(channels.items(), means)}

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f'dense{i}', nn.Linear, x.shape[-1], width)(x)
            x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f'dense_{key}', DistLayer, shape)(x)
        return dists


class GRUCell(common.Module):
    def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.LazyLinear(3 * size, bias=norm is not None, **kwargs)
        if norm:
            self._norm = nn.LayerNorm(3 * size)

    @property
    def state_size(self):
        return self._size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = inputs.shape[0]
        return torch.zeros((batch_size, self._size), dtype=dtype)

    def forward(self, inputs, state):
        state = state[0]
        parts = self._layer(torch.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = parts.to(dtype=torch.float32)
            parts = self._norm(parts)
            parts = parts.to(dtype=dtype)
        reset, cand, update = torch.chunk(parts, 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class MLP(common.Module):

    def __init__(self, shape, layers, units, act='elu', norm='none', **out):
        super().__init__()
        self._shape = (shape, ) if isinstance(shape, int) else tuple(shape)
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def forward(self, features):
        x = features.to(dtype=torch.get_default_dtype())
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f'dense{index}', nn.Linear, x.shape[-1], self._units)(x)
            x = self.get(f'norm{index}', NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape(tuple(features.shape[:-1]) + (x.shape[-1], ))
        return self.get('out', DistLayer, self._shape, **self._out)(x)


class DistLayer(common.Module):

    def __init__(self, shape, dist='mse', min_std=0.1, init_std=0.0):
        super().__init__()
        self._shape = tuple(shape)
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def forward(self, inputs):
        size = int(np.prod(self._shape))
        out = self.get('out', nn.Linear, inputs.shape[-1], size)(inputs)
        out = out.reshape(tuple(inputs.shape[:-1]) + self._shape)
        out = out.to(dtype=torch.float32)
        if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
            std = self.get('std', nn.Linear, inputs.shape[-1], size)(inputs)
            std = std.reshape(tuple(inputs.shape[:-1]) + self._shape)
            std = std.to(dtype=torch.float32)
        if self._dist == 'mse':
            dist = Normal(out, torch.ones_like(out))
            return Independent(dist, len(self._shape))
        if self._dist == 'normal':
            dist = Normal(out, std)
            return Independent(dist, len(self._shape))
        if self._dist == 'binary':
            dist = Bernoulli(logits=out)
            return Independent(dist, len(self._shape))
        if self._dist in ('tanh_normal', 'trunc_normal', 'onehot'):
            raise NotImplementedError(self._dist)
        raise NotImplementedError(self._dist)


class NormLayer(common.Module):

    def __init__(self, name):
        super().__init__()
        if name == 'none':
            self._layer = None
        elif name == 'layer':
            self._layer = 'layer'
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if not self._layer:
            return features
        if features.ndim == 4:
            features = features.permute(0, 2, 3, 1)
            features = self.get('layer', nn.LayerNorm, features.shape[-1])(features)
            return features.permute(0, 3, 1, 2).contiguous()
        return self.get('layer', nn.LayerNorm, features.shape[-1])(features)


def get_act(name):
    if name == 'none':
        return lambda x: x
    if name == 'elu':
        return F.elu
    if name == 'relu':
        return F.relu
    if name == 'tanh':
        return torch.tanh
    raise NotImplementedError(name)
