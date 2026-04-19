import re

import torch
from torch import nn
from torch.nn import functional as F

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

    def kl_loss(self, post, prior, foward, balance, free, free_avq):
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

        if self.cnn_keys:
            in_channels = sum(shapes[k][-1] for k in self.cnn_keys)
            self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=4, stride=2)
            self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2)
            self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2)

        if self.mlp_keys:
            in_features = sum(shapes[k][0] for k in self.mlp_keys)
            self.fc1 = nn.Linear(in_features, 400)
            self.fc2 = nn.Linear(400, 400)
            self.fc3 = nn.Linear(400, 400)
            self.fc4 = nn.Linear(400, 400)

    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        x = x.to(dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self._act(self.conv1(x))
        x = self._act(self.conv2(x))
        x = self._act(self.conv3(x))
        x = self._act(self.conv4(x))
        return x.reshape(x.size(0), -1)

    def _mlp(self, data):
        x = torch.cat(list(data.values()), -1)
        x = x.to(dtype=torch.float32)
        x = self._act(self.fc1(x))
        x = self._act(self.fc2(x))
        x = self._act(self.fc3(x))
        x = self._act(self.fc4(x))
        return x

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
        output = torch.cat(outputs, -1)
        return output.reshape(tuple(batch_dims) + tuple(output.shape[1:]))


class Decoder(common.Module):

    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
        print('Decoder CNN outputs:', list(self.cnn_keys))
        print('Decoder MLP outputs:', list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def forward(self, features):
        pass

    def _cnn(self, features):
        pass

    def _mlp(self, features):
        pass


class MSE(common.Module):

    def __init__(self):
        super().__init__()


class _ChannelLastLayerNorm(common.Module):

    def __init__(self, channels):
        super().__init__()
        self._layer = nn.LayerNorm(channels)

    def forward(self, features):
        if features.ndim != 4:
            return self._layer(features)
        features = features.permute(0, 2, 3, 1)
        features = self._layer(features)
        return features.permute(0, 3, 1, 2).contiguous()


def _make_norm(name, size, dims):
    if name == 'none':
        return nn.Identity()
    if name == 'layer':
        if dims == 4:
            return _ChannelLastLayerNorm(size)
        return nn.LayerNorm(size)
    raise NotImplementedError(name)


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


