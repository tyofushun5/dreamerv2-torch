from torch import nn


class Module(nn.Module):
    def get(self, name, ctor, *args, **kwargs):
        if name not in self._modules:
            self.add_module(name, ctor(*args, **kwargs))
        return self._modules[name]


__all__ = ["Module"]
