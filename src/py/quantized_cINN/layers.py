import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import Module


class DynamicScaling(Module):
    r"""Applies a linear scaling to the incoming data: :math:`y = x*w + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H)` where :math:`*` means any number of
          additional dimensions and :math:`H = \text{features}`.
        - Output: :math:`(N, *, H)`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """
    __constants__ = ['features']
    features: int
    weight: Tensor

    def __init__(self, features: int, bias: bool = True,
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DynamicScaling, self).__init__()
        self.features = features

        self.weight = Parameter(torch.empty(features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(torch.diag(self.weight), a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch.diag(self.weight))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.diag(self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'features={}, bias={}'.format(
            self.features, self.bias is not None
        )


class FixedRangeScaling(Module):
    r"""
    """
    __constants__ = ['features', 'max_out']
    per_feature: bool
    features: int
    max_out: int
    weight: Tensor

    def __init__(self, features: int, max_out: int, per_feature: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FixedRangeScaling, self).__init__()
        self.features = features
        self.max_out = max_out
        self.per_feature = per_feature

        self.weight = Parameter(torch.empty(features, **factory_kwargs),
                                requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            max_abs, _ = torch.max(torch.abs(input), axis=0)
            if not per_feature:
                max_abs = torch.max(max_abs, keepdim=True)
            max_abs[max_abs == 0.] = self.max_out
            self.weight = self.max_out / max_abs
        return F.linear(input, torch.diag(self.weight))

    def extra_repr(self) -> str:
        return 'features={}, max_out={}, per_feature={}'.format(
            self.features, self.max_out, self.per_feature
        )
