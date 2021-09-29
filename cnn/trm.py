import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from cnn.embedding import stockEmbedding
from cnn.base_block import halfConv1d


# 趋势之间没有相关性，趋势是累积效应，不与其他趋势成近义相关
# todo y label as x
# todo conv embedding to smooth feature without activation
# todo simple trend line without fuzzy operation
# todo unmulti res
# bert model embedding the trend by surrounding trend,so the prob of netural trend will be the max.
# BN rugular the batch while the var trend has low prob, and the varience of var trend will be shrinked   and cant be catch by the model

class stockBERTtoken(nn.Module):
    # todo period+1 for the symbol of classification
    def __init__(self, feature_num, period, class_num, hidden=256, n_layers=12,
                 attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = stockEmbedding(period, feature_num, hidden, dropout=0)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.token_classify = nn.Linear(hidden, class_num)
        self.cls_classify = nn.Linear(hidden, class_num)


    def forward(self, x):
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, None)  # NLC

        token = self.token_classify(x)
        cls = self.cls_classify(x[:, -1])
        return token, cls

class stockBERT(nn.Module):
    # todo period+1 for the symbol of classification
    def __init__(self, feature_num, period, class_num, hidden=256, n_layers=12,
                 attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = stockEmbedding(period, feature_num, hidden, dropout=0)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        # self.token_classify = nn.Linear(hidden, class_num)
        # self.cls_classify = halfConv1d(hidden, class_num, kernel_size=3, stride=1, padding=1,
        #                                bias=True)
        # self.classify = nn.Sequential(
        #     nn.Linear(hidden, hidden * 2),
        #     nn.Tanh(),
        #     nn.Linear(hidden * 2, hidden),
        #     nn.Tanh(),
        #     nn.Linear(hidden, class_num),
        # )

        # self.classify = nn.Sequential(
        #     nn.Conv1d(hidden, class_num, 3, 1, 1, groups=1, bias=False),
        #     nn.AdaptiveAvgPool1d(1)
        # )

        self.classify = nn.Linear(hidden, class_num)

    def forward(self, x):
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, None)  # NLC

        # token = self.token_classify(x)

        # x = x.transpose(1, 2)
        # x = self.classify(x)
        # x = x.squeeze(dim=-1)
        # x = x.mean(axis=-1)
        # classification
        # max only produce grad on the selected elements
        # cls = torch.max(x[:, -3:], 1)
        # cls = self.cls_classify(cls)
        # cls, _ = torch.max(cls, 1)
        # cls = torch.mean(cls, 1)

        # cls = x.transpose(1, 2)
        # cls = self.cls_classify(cls)
        # cls = cls[:, :, -1]

        cls = self.classify(x[:, -1])
        return cls


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
