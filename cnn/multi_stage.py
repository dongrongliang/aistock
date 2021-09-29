from cnn.embedding import stockEmbedding
from cnn.half_res import *
from torch.nn import functional as F


class featHconvResNet(nn.Module):

    def __init__(
            self,
            channel_in: int,
            block: Type[Union[hconvBasicBlock, hconvBottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(featHconvResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = halfConv1d(channel_in, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, hconvBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, hconvBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[hconvBasicBlock, hconvBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                hconv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class msHconvResNet(nn.Module):

    def __init__(self, channel_in, class_num, netrual_layer=[2, 2, 2, 2], var_layer=[2, 2, 2, 2],
                 norm_layer=nn.InstanceNorm1d):
        super(msHconvResNet, self).__init__()
        self.class_num = class_num
        self.var_channel = 128
        self.netural_net = featHconvResNet(channel_in=channel_in, block=hconvBasicBlock,
                                           layers=netrual_layer, norm_layer=norm_layer)
        self.var_net = hconvResBlock(channel_in=128 * hconvBasicBlock.expansion, channel_out=self.var_channel, block=hconvBasicBlock,
                                     layers=var_layer, norm_layer=norm_layer)

        self.netural_avgpool = nn.AdaptiveAvgPool1d(1)
        self.var_avgpool = nn.AdaptiveAvgPool1d(1)
        self.mask_act = nn.Sigmoid()
        self.var_avgpool = nn.AdaptiveAvgPool1d(1)
        self.netural_classify = nn.Linear(128 * hconvBasicBlock.expansion, 2)
        self.var_classify = nn.Linear(self.var_channel, class_num)

    def forward(self, x: Tensor, return_netural: bool = False):
        n, c, l = x.size()
        netural_feat = self.netural_net(x)

        var_feat = self.var_net(netural_feat)

        netural_y = self.netural_classify(torch.flatten(self.netural_avgpool(netural_feat), 1))
        mask_y = self.mask_act(netural_y).view(n, 1, -1)
        # mask_y = netural_y.view(n, 1, -1)
        mask_y = F.pad(mask_y, (0, self.class_num - 2), mode='replicate').view(n, -1)

        var_y = self.var_classify(torch.flatten(self.var_avgpool(var_feat), 1))
        var_y = var_y * mask_y
        # var_y = var_y + mask_y

        if return_netural:
            return netural_y, var_y
        else:
            return var_y


class midHconvResBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            outplanes: int,
            block: Type[Union[hconvBasicBlock, hconvBottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(midHconvResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, outplanes, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, hconvBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, hconvBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[hconvBasicBlock, hconvBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                hconv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class BiMsHconvResNet(nn.Module):

    def __init__(self, channel_in, class_num, netrual_layer=[2, 2, 2, 2], var_layer=[2, 2, 2, 2],
                 norm_layer=nn.InstanceNorm1d):
        super(BiMsHconvResNet, self).__init__()
        self.class_num = class_num
        self.inplanes = 64
        self.midplanes = 128
        self.conv1 = halfConv1d(channel_in, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.netural_net = midHconvResBlock(inplanes=self.inplanes, outplanes=self.midplanes, block=hconvBasicBlock,
                                           layers=netrual_layer, norm_layer=norm_layer)
        self.var_net = midHconvResBlock(inplanes=self.inplanes, outplanes=self.midplanes, block=hconvBasicBlock,
                                     layers=var_layer, norm_layer=norm_layer)


        self.netural_avgpool = nn.AdaptiveAvgPool1d(1)
        self.var_avgpool = nn.AdaptiveAvgPool1d(1)
        self.mask_act = nn.Sigmoid()
        self.var_avgpool = nn.AdaptiveAvgPool1d(1)
        self.netural_classify = nn.Linear(self.midplanes * hconvBasicBlock.expansion, 2)
        self.var_classify = nn.Linear(self.midplanes * hconvBasicBlock.expansion, class_num)

    def forward(self, x: Tensor, return_netural: bool = False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        n, c, l = x.size()

        netural_feat = self.netural_net(x)

        var_feat = self.var_net(x)


        netural_y = self.netural_classify(torch.flatten(self.netural_avgpool(netural_feat), 1))
        mask_y = self.mask_act(netural_y).view(n, 1, -1)
        # mask_y = netural_y.view(n, 1, -1)
        mask_y = F.pad(mask_y, (0, self.class_num - 2), mode='replicate').view(n, -1)

        var_y = self.var_classify(torch.flatten(self.var_avgpool(var_feat), 1))
        var_y = var_y * mask_y
        # var_y = var_y + mask_y

        if return_netural:
            return netural_y, var_y
        else:
            return var_y
