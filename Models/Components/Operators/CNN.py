import torch
import torch.nn as nn


class CONV1D(nn.Module):
    r"""Applies a 1D convolution over an input signal composed of several input
        planes.

        .. math::
            \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
            \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
            \star \text{input}(N_i, k)

        where :math:`\star` is the valid `cross-correlation`_ operator,
        :math:`N` is a batch size, :math:`C` denotes a number of channels,
        :math:`L` is a length of signal sequence.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``
        """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(CONV1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.conv1d = nn.Conv1d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups=self.groups,
                                bias=self.bias,
                                padding_mode=self.padding_mode
                                )

    def forward(self, inputs):
        return self.conv1d(inputs)


class CONV2D(nn.Module):
    """
    Applies a 2D convolution over an input signal composed of several input planes.

        In the simplest case, the output value of the layer with input size
        :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
        can be precisely described as:

        .. math::
            \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
            \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


        where :math:`\star` is the valid 2D `cross-correlation`_ operator,
        :math:`N` is a batch size, :math:`C` denotes a number of channels,
        :math:`H` is a height of input planes in pixels, and :math:`W` is
        width in pixels.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

              .. math::
                  H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                            \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              .. math::
                  W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                            \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

        Attributes:
            weight (Tensor): the learnable weights of the module of shape
                :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                The values of these weights are sampled from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
            bias (Tensor):   the learnable bias of the module of shape
                (out_channels). If :attr:`bias` is ``True``,
                then the values of these weights are
                sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

        Examples:

            >>> # With square kernels and equal stride
            >>> m = nn.Conv2d(16, 33, 3, stride=2)
            >>> # non-square kernels and unequal stride and with padding
            >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
            >>> # non-square kernels and unequal stride and with padding and dilation
            >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
            >>> input = torch.randn(20, 16, 50, 100)
            >>> output = m(input)

        .. _cross-correlation:
            https://en.wikipedia.org/wiki/Cross-correlation

        .. _link:
            https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(CONV2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.conv2d = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups=self.groups,
                                bias=self.bias,
                                padding_mode=self.padding_mode
                                )

    def forward(self, inputs):
        return self.conv2d(inputs)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, res_rate, num_layers,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'
                 ):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.res_rate = res_rate
        self.num_layers = num_layers

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.res_block = nn.ModuleList([CONV1D(in_channels=self.in_channels,
                                               out_channels=self.out_channels,
                                               kernel_size=self.kernel_size,
                                               stride=self.stride,
                                               padding=self.padding,
                                               dilation=self.dilation,
                                               groups=self.groups,
                                               bias=self.bias,
                                               padding_mode=self.padding_mode)
                                        for _ in range(num_layers)])

    def forward(self, inputs):
        outputs = inputs
        for layer in self.res_block:
            outputs = layer(outputs)

        return inputs + (self.res_rate * outputs)
