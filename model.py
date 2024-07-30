import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0., scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class LayerNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-6, channels_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.channels_first = channels_first
        self.num_features = (num_features, )
    
    def forward(self, x):
        if not self.channels_first:
            return nn.functional.layer_norm(x, self.num_features, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u).div(torch.sqrt(s + self.eps))
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block3d(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.depthwise = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm3d(dim, eps=1e-6)
        self.pointwise = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.compress = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.compress(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x


class UpSampleLayer3d(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm3d(in_channels, channels_first=True)
        self.compress = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, shape):
        x = nn.functional.interpolate(x, shape[2::], mode='nearest')
        x = self.norm(x)
        x = self.compress(x)
        return x


class ConcatConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        return x

    def __repr__(self):
        return f'ConcatConv3d[in_channels: {self._in_channels}, out_channels: {self._out_channels}]'


class CraNeXt(nn.Module):
    def __init__(self, in_channels=1, num_classes=1,
                 depths=[1,1,1,3,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 256, 128, 64, 32],
                 drop_path_rate=0.1, head_init_scale=1.
                 ):
        super().__init__()
        assert len(depths) == len(dims), f'Invalid depths and dims'
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        dp_rates = [drop_path_rate for _ in range(sum(depths))]
        self.down_stages = nn.ModuleList()
        self.up_stages = nn.ModuleList()
        self.down_norms = nn.ModuleList()
        self.up_norms = nn.ModuleList()
        self.init_stage = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=7, padding=3),
            Block3d(dim=dims[0], drop_path=0.0)
        )
        self.downsample_layers.append(nn.Sequential(
            nn.Conv3d(dims[0], dims[0], kernel_size=2, stride=2),
            LayerNorm3d(dims[0], channels_first=True)
        ))
        for i in range(len(dims) // 2):
            downsample_layer = nn.Sequential(
                LayerNorm3d(dims[i], channels_first=True),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        cur = 0
        for i in range(len(dims) // 2 + 1):
            stage = nn.Sequential(
                *[Block3d(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.down_stages.append(stage)
            cur += depths[i]
        for i_layer in range(len(dims) // 2 + 1):
            layer = LayerNorm3d(dims[i_layer], channels_first=True)
            self.down_norms.append(layer)
        for i_layer in range(len(dims) // 2):
            layer = LayerNorm3d(dims[i_layer + len(dims) // 2 + 1], channels_first=True)
            self.up_norms.append(layer)
        for i_layer in range(len(dims) // 2):
            layer = UpSampleLayer3d(dims[i_layer + len(dims) // 2], dims[i_layer + 1 + len(dims) // 2])
            self.upsample_layers.append(layer)
        cur = 0
        for i in range(len(dims) // 2):
            num_blocks = depths[i + len(dims) // 2 + 1]
            stage = nn.Sequential(
                *[Block3d(dim=dims[i + len(dims) // 2 + 1], drop_path=dp_rates[cur + j]) for j in range(num_blocks)]
            )
            self.up_stages.append(stage)
            cur += depths[i]
        self.concat = nn.ModuleList()
        for i in range(len(dims) // 2):
            self.concat.append(ConcatConv3d(in_channels=dims[i + len(dims) // 2 + 1] + dims[len(dims) // 2 - (i + 1)],
                                          out_channels=dims[i + len(dims) // 2 + 1])
            )
        self.final_concat = ConcatConv3d(in_channels=dims[0] + dims[-1], out_channels=dims[-1])
        self.upscale_out = UpSampleLayer3d(dims[-1], dims[-1])
        self.final_convolution = nn.Conv3d(dims[-1], num_classes, 1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.init_stage(x)
        shapes = [x.shape]
        steps = [x]
        assert len(self.upsample_layers) == len(self.concat)
        assert len(self.upsample_layers) == len(self.up_stages)
        assert len(self.upsample_layers) == len(self.up_norms)
        for i, (down, stage, norm) in enumerate(zip(self.downsample_layers, self.down_stages, self.down_norms)):
            if i != 0:
                steps.append(x)
            x = down(x)
            x = norm(x)
            x = stage(x)
            shapes.append(x.shape)
        shapes.reverse()
        for i, (up, cat, stage, norm) in enumerate(zip(self.upsample_layers, self.concat, self.up_stages, self.up_norms)):
            x = up(x, shapes[i + 1])
            y = steps.pop(-1)
            x = cat(x, y)
            x = stage(x)
            x = norm(x)
        x = self.upscale_out(x, shapes[-1])
        y = steps.pop(-1)
        x = self.final_concat(x, y)
        x = self.final_convolution(x)
        x = self.activation(x)
        return x


class CraNeXt_tiny(CraNeXt):
    def __init__(self):
        super().__init__(depths=[1,3,1,1,1],dims=[32, 64, 128, 64, 32])


class CraNeXt_old_ratio(CraNeXt):
    def __init__(self):
        super().__init__(depths=[2,2,2,2,3,1,1,1,1])
