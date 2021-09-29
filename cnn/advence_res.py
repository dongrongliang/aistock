from cnn.base_net import *
from cnn.adtrm import adTransformerBlock, adResTransformerBlock
from cnn.base_block import *
from cnn.regulaztion import *
# todo mix dim convert and res
import functools
class adResNet(nn.Module):

    def __init__(
            self,
            period: int,
            channel_in: int,
            layers: List[int],
            num_classes: int = 3,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(adResNet, self).__init__()

        # self.trend_nn = nn.Sequential(
        #     nn.Linear(period, period*2),
        #     norm_layer(channel_in),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(period * 2, 32),
        #     norm_layer(channel_in),
        #     nn.ReLU(inplace=True)
        # )
        hidden_channel = channel_in
        hidden_period = period
        classify_period = period
        # basic_res_block = functools.partial(BasicBlock, drop_prob=0.1, drop_size=7)
        self.init_norm = nn.Sequential(
            # make_conv(channel_in, hidden_channel, kernel_size=1, padding=0, activation=nn.ReLU(inplace=True),
            #           use_bias=False),
            # norm_layer(hidden_channel),
            # make_conv(hidden_channel, hidden_channel, kernel_size=1, padding=0, activation=nn.ReLU(inplace=True),
            #           use_bias=False),
            # norm_layer(hidden_channel)

        )

        self.dim_convert = nn.Sequential(
            # make_conv(channel_in, hidden_channel, kernel_size=1, padding=0, activation=nn.ReLU(inplace=True),
            #           use_bias=True),
            # make_conv(hidden_channel, hidden_channel, kernel_size=1, padding=0, activation=nn.ReLU(inplace=True),
            #           use_bias=True),
            nn.Linear(period, hidden_period),
            norm_layer(hidden_channel),
            nn.ReLU(inplace=True),
            # nn.FeatureAlphaDropout(0.1),
            nn.Linear(hidden_period, hidden_period),
            norm_layer(hidden_channel),
            nn.ReLU(inplace=True),
            # nn.FeatureAlphaDropout(0.1),
                            )
        self.trend_nn = nn.Sequential(
            # nn.Linear(period, period * 2),
            # norm_layer(channel_in),
            # nn.ReLU(inplace=True),
            # nn.Linear(period * 2, period * 2),
            # norm_layer(channel_in),
            # nn.ReLU(inplace=True),


            # adTransformerBlock(hidden_channel, 4, hidden_channel, channel_in=hidden_channel, norm=nn.LayerNorm),
            # adResTransformerBlock(hidden_channel, 4, hidden_channel, channel_in=hidden_channel, norm=nn.LayerNorm),

            # adResTransformerBlock(period * 2, 4, 32, channel_in=channel_in)

            # adTransformerBlock(period * 2, 4, period, channel_in=channel_in),
            # adTransformerBlock(period, 4, 32, channel_in=channel_in),

            # adResTransformerBlock(hidden_period, 4, hidden_period, channel_in=hidden_channel),
            # adResTransformerBlock(hidden_period, 4, classify_period, channel_in=hidden_channel),

            adResTransformerBlock(hidden_period, 4, hidden_period, channel_in=hidden_period, norm=nn.LayerNorm),
            adResTransformerBlock(hidden_period, 4, classify_period, channel_in=hidden_period, norm=nn.LayerNorm),

            # adResTransformerBlock(period * 2, 4, period * 2, channel_in=channel_in),
            # adResTransformerBlock(period * 2, 4, period * 2, channel_in=channel_in),
            # adResTransformerBlock(period * 2, 4, period, channel_in=channel_in),
            # adResTransformerBlock(period, 4, period, channel_in=channel_in),
            # adResTransformerBlock(period, 4, period, channel_in=channel_in),
            # adResTransformerBlock(period, 4, 32, channel_in=channel_in),
        )

        # self.resnet = mixResBlock(period=classify_period, channel_in=hidden_channel, block=BasicBlock,
        #             layers=layers, num_classes=num_classes, norm_layer=norm_layer,
        #             drop_prob=0.1)
        self.resnet = adResBlock(hidden_channel, BasicBlock, layers, num_classes,
                                 zero_init_residual, groups, width_per_group,
                                 replace_stride_with_dilation, norm_layer, drop_prob=0.1)

    def forward(self, x: Tensor) -> Tensor:
        # x = self.init_norm(x)
        # x = x.transpose(1, 2) # NLC
        x = self.dim_convert(x)
        x = self.trend_nn(x)
        # x = x.transpose(1, 2) # NCL
        x = self.resnet(x)
        return x


class adResBlock(nn.Module):

    def __init__(
            self,
            channel_in: int,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 3,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            drop_prob: float =0,
            drop_size: int =7
    ) -> None:
        super(adResBlock, self).__init__()
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
        self.drop = drop_prob > 0
        # if self.drop:
        #
        #     self.dropblock = LinearScheduler(
        #         DropBlock1D(drop_prob=drop_prob, block_size=drop_size),
        #         start_value=0.,
        #         stop_value=drop_prob,
        #         nr_steps=500
        #     )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(channel_in, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        if self.drop:
            drop_block = functools.partial(block, drop_prob=drop_prob, drop_size=drop_size)
            self.layer3 = self._make_layer(drop_block, 64, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(drop_block, 128, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        else:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 128, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # if self.drop:
        #     self.dropblock.step()  # increment number of iterations

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.dropblock(self.layer3(x)) if self.drop else self.layer3(x)
        # x = self.dropblock(self.layer4(x)) if self.drop else self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class mixResBlock(nn.Module):

    def __init__(
            self,
            period: int,
            channel_in: int,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 3,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            drop_prob: float = 0,
            drop_size: int = 7
    ) -> None:
        super(mixResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.drop = drop_prob > 0
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
        self.conv1 = nn.Conv1d(channel_in, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], dilate=replace_stride_with_dilation[0])

        classify_channel = period // 4
        self.ffn1 = nn.Sequential(
            nn.Linear(classify_channel, int(classify_channel // 2)),
            norm_layer(16),
            nn.ReLU(inplace=True),
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(int(classify_channel // 2), int(classify_channel // 4)),
            norm_layer(32),
            nn.ReLU(inplace=True),
        )

        self.ffn3 = nn.Sequential(
            nn.Linear(int(classify_channel // 4), int(classify_channel // 8)),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )

        self.ffn4 = nn.Sequential(
            nn.Linear(int(classify_channel // 8), 1),
            norm_layer(128),
            nn.ReLU(inplace=True),
        )

        if self.drop:
            drop_block3 = functools.partial(block, drop_prob=drop_prob, drop_size=drop_size)
            drop_block4 = functools.partial(block, drop_prob=drop_prob, drop_size=drop_size // 2)
            self.layer3 = self._make_layer(drop_block3, 64, layers[2], dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(drop_block4, 128, layers[3], dilate=replace_stride_with_dilation[2])
        else:
            self.layer3 = self._make_layer(block, 64, layers[2], dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 128, layers[3], dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.ffn1(x)
        x = self.layer2(x)
        x = self.ffn2(x)
        x = self.layer3(x)
        x = self.ffn3(x)
        x = self.layer4(x)
        x = self.ffn4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
