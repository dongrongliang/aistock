from cnn.base_block import *
import torch.nn as nn
import functools
from torch import Tensor
""""""""""""""""""""""""""""""""" Base Hrnet """""""""""""""""""""""""""""""""

class StockHRNet(nn.Module):

    def __init__(self, input_channel=16, output_channel=3, n1=16, n2=32, n3=64,
                 activation='relu', activation_final='relu',
                 n_block=2, stage_num=-3,
                 mode='train'):
        super(StockHRNet, self).__init__()

        self.mode = mode
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n3 * 2
        self.deep_fuse_flag = stage_num < 0
        self.stage_num = abs(stage_num)
        self.n_block = int(n_block)
        self.use_bias = n_block < 0
        self.g_feats_dim = n1

        if activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        if activation_final == 'tanh':
            self.activation_final = nn.Tanh()
        elif activation_final == 'prelu':
            self.activation_final = nn.PReLU()
        elif activation_final == 'sigmoid':
            self.activation_final = nn.Sigmoid()
        elif activation_final == 'relu':
            self.activation_final = nn.ReLU(inplace=True)
        else:
            self.activation_final = None

        self.conv1 = make_conv(input_channel, self.n1, kernel_size=1, padding=0, activation=self.activation,
                               use_bias=False)

        self.conv2 = make_conv(self.n1, self.n1, kernel_size=3, padding=1, activation=self.activation,
                               use_bias=False)

        #

        base_block = functools.partial(HrBaseBlock, activation=self.activation, use_bias=self.use_bias)

        # stage 1
        # block
        s1_l1_block = [
            make_conv(self.n1, self.n1, kernel_size=3, stride=1, padding=1, activation=self.activation,
                      use_bias=False)]
        s1_l2_block = [make_conv(self.n1, n2, kernel_size=3, stride=2, padding=1,
                                 activation=self.activation,
                                 use_bias=False)]
        for i in range(self.n_block):
            s1_l1_block.append(base_block(n1, n1))
            s1_l2_block.append(base_block(n2, n2))

        self.s1_l1_block = nn.Sequential(*s1_l1_block)
        self.s1_l2_block = nn.Sequential(*s1_l2_block)

        # stage 2

        ## transition
        self.s1_l1_down_l2 = make_conv(n1, n2, kernel_size=3, stride=2, padding=1, activation=self.activation,
                                       use_bias=False)
        self.s1_l1_down_l3 = nn.Sequential(
            *[make_conv(n1, n2, kernel_size=3, stride=2, padding=1, activation=self.activation,
                        use_bias=False),
              make_conv(n2, n3, kernel_size=3, stride=2, padding=1, activation=self.activation,
                        use_bias=False)])
        self.s1_l2_up_l1 = nn.Sequential(
            *[
              upsampling1d(scale_factor=2, mode='bilinear'),
              make_conv(n2, n1, kernel_size=3, stride=1, padding=1, activation=self.activation,
                        use_bias=False)
              ])
        self.s1_l2_down_l3 = make_conv(n2, n3, kernel_size=3, stride=2, padding=1, activation=self.activation,
                                       use_bias=False)

        ## block

        s2_l1_block = []
        s2_l2_block = []
        s2_l3_block = []

        for i in range(self.n_block):
            s2_l1_block.append(base_block(n1, n1))
            s2_l2_block.append(base_block(n2, n2))
            s2_l3_block.append(base_block(n3, n3))

        self.s2_l1_block = nn.Sequential(*s2_l1_block)
        self.s2_l2_block = nn.Sequential(*s2_l2_block)
        self.s2_l3_block = nn.Sequential(*s2_l3_block)

        # stage 3
        if self.stage_num >= 3:
            # transition
            self.s2_l1_down_l2 = make_conv(n1, n2, kernel_size=3, stride=2, padding=1, activation=self.activation,
                                           use_bias=False)
            self.s2_l1_down_l3 = nn.Sequential(
                *[make_conv(n1, n2, kernel_size=3, stride=2, padding=1, activation=self.activation,
                            use_bias=False),
                  make_conv(n2, n3, kernel_size=3, stride=2, padding=1, activation=self.activation,
                            use_bias=False)])

            self.s2_l1_down_l4 = nn.Sequential(
                *[make_conv(n1, n2, kernel_size=3, stride=2, padding=1, activation=self.activation,
                            use_bias=False),
                  make_conv(n2, n3, kernel_size=3, stride=2, padding=1, activation=self.activation,
                            use_bias=False),
                  make_conv(n3, self.n4, kernel_size=3, stride=2, padding=1, activation=self.activation,
                            use_bias=False)])

            self.s2_l2_up_l1 = nn.Sequential(
                *[upsampling1d(scale_factor=2, mode='bilinear'),
                  make_conv(n2, n1, kernel_size=3, stride=1, padding=1, activation=self.activation,
                            use_bias=False)
                  ])

            self.s2_l2_down_l3 = make_conv(n2, n3, kernel_size=3, stride=2, padding=1, activation=self.activation,
                                           use_bias=False)

            self.s2_l2_down_l4 = nn.Sequential(
                *[make_conv(n2, n3, kernel_size=3, stride=2, padding=1, activation=self.activation,
                            use_bias=False),
                  make_conv(n3, self.n4, kernel_size=3, stride=2, padding=1, activation=self.activation,
                            use_bias=False)])

            self.s2_l3_up_l1 = nn.Sequential(
                *[upsampling1d(scale_factor=4, mode='bilinear'),
                  make_conv(n3, n1, kernel_size=3, stride=1, padding=1, activation=self.activation,
                            use_bias=False)
                  ])

            self.s2_l3_up_l2 = nn.Sequential(
                *[upsampling1d(scale_factor=2, mode='bilinear'),
                  make_conv(n3, n2, kernel_size=3, stride=1, padding=1, activation=self.activation,
                            use_bias=False)
                  ])

            self.s2_l3_down_l4 = make_conv(n3, self.n4, kernel_size=3, stride=2, padding=1, activation=self.activation,
                                           use_bias=False)

            # block
            s3_l1_block = []
            s3_l2_block = []
            s3_l3_block = []
            s3_l4_block = []

            for i in range(self.n_block):
                s3_l1_block.append(base_block(n1, n1))
                s3_l2_block.append(base_block(n2, n2))
                s3_l3_block.append(base_block(n3, n3))
                s3_l4_block.append(base_block(self.n4, self.n4))

            self.s3_l1_block = nn.Sequential(*s3_l1_block)
            self.s3_l2_block = nn.Sequential(*s3_l2_block)
            self.s3_l3_block = nn.Sequential(*s3_l3_block)
            self.s3_l4_block = nn.Sequential(*s3_l4_block)

        # cat
        self.c_l2upl1 = upsampling1d(scale_factor=2, mode='bilinear')
        self.c_l3upl1 = upsampling1d(scale_factor=4, mode='bilinear')
        self.c_l4upl1 = upsampling1d(scale_factor=8, mode='bilinear')

        # fuse

        if self.stage_num >= 3:
            if self.deep_fuse_flag:
                deep_fuse_lst = [
                    HrBaseBlock(n1 + n2 + n3 + self.n4, n1, activation=self.activation)
                ]
                self.fuse_layer = nn.Sequential(*deep_fuse_lst)
            else:
                self.fuse_layer = HrBaseBlock(n1 + n2 + n3 + self.n4, n1, activation=self.activation)
        else:
            if self.deep_fuse_flag:
                deep_fuse_lst = [
                    HrBaseBlock(n1 + n2 + n3, n3, activation=self.activation),
                    HrBaseBlock(n3, n2, activation=self.activation),
                    HrBaseBlock(n2, n1, activation=self.activation),
                ]
                self.fuse_layer = nn.Sequential(*deep_fuse_lst)
            else:
                self.fuse_layer = HrBaseBlock(n1 + n2 + n3, n1, activation=self.activation)


        # final layer
        self.final_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n1,
                out_channels=input_channel,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Sequential() if self.activation_final is None else self.activation_final
        )

        self.classify = nn.Linear(input_channel, output_channel)


    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        # stage-1

        l1 = self.s1_l1_block(x)
        l2 = self.s1_l2_block(x)

        # stage-2
        s2_l1_x = self.activation(l1 + self.s1_l2_up_l1(l2))
        s2_l2_x = self.activation(self.s1_l1_down_l2(l1) + l2)
        s2_l3_x = self.activation(self.s1_l1_down_l3(l1) + self.s1_l2_down_l3(l2))

        l1 = self.s2_l1_block(s2_l1_x)
        l2 = self.s2_l2_block(s2_l2_x)
        l3 = self.s2_l3_block(s2_l3_x)

        # stage-3
        if self.stage_num >= 3:
            s3_l1_x = self.activation(l1 + self.s2_l2_up_l1(l2) + self.s2_l3_up_l1(l3))
            s3_l2_x = self.activation(self.s2_l1_down_l2(l1) + l2 + self.s2_l3_up_l2(l3))
            s3_l3_x = self.activation(self.s2_l1_down_l3(l1) + self.s2_l2_down_l3(l2) + l3)
            s3_l4_x = self.activation(self.s2_l1_down_l4(l1) + self.s2_l2_down_l4(l2) + self.s2_l3_down_l4(l3))

            l1 = self.s3_l1_block(s3_l1_x)
            l2 = self.s3_l2_block(s3_l2_x)
            l3 = self.s3_l3_block(s3_l3_x)
            l4 = self.s3_l4_block(s3_l4_x)

            l2 = self.c_l2upl1(l2)
            l3 = self.c_l3upl1(l3)
            l4 = self.c_l4upl1(l4)

            l1 = torch.cat([l1, l2, l3, l4], 1)
        else:
            l2 = self.c_l2upl1(l2)
            l3 = self.c_l3upl1(l3)

            l1 = torch.cat([l1, l2, l3], 1)

        # fuse

        l1 = self.fuse_layer(l1)
        l1 = self.final_layer(l1)
        l1 = l1.mean(dim=2)
        l1 = self.classify(l1)

        return l1