import torch.nn as nn
import torch
import math
from cnn.base_block import halfConv1d
from cnn.half_res import hconvResBlock, hconvBasicBlock


class stockEmbedding(nn.Module):

    def __init__(self, period, feature_num, embed_size, dropout=0, order_emb = True):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.order_emb = order_emb
        # self.token = hconvResBlock(channel_in=feature_num, channel_out=embed_size, block=hconvBasicBlock,
        #                            layers=[2, 2, 2, 2], norm_layer=nn.InstanceNorm1d)

        self.token = nn.Linear(feature_num, embed_size)

        self.position = PositionalEmbedding(d_model=embed_size, max_len=period)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, stock_sequence):
        # stock_sequence = self.token(stock_sequence)
        # stock_sequence = stock_sequence.transpose(1, 2)  # NLC

        stock_sequence = stock_sequence.transpose(1, 2)  # NLC
        stock_sequence = self.token(stock_sequence)


        # b, l, c = stock_sequence.size()
        # cls = torch.zeros(b, 1, c).to(stock_sequence.device)
        # stock_sequence = torch.cat([cls, stock_sequence], dim=1)

        x = stock_sequence + self.position(stock_sequence) if self.order_emb else stock_sequence
        return self.dropout(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 不同位置不同编码
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 不同维度不同缩放
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
