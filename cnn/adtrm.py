from cnn.trm import *

class adResTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, d_out, dropout=0, channel_in=16, norm=nn.LayerNorm):

        super().__init__()
        self.hidden = hidden
        self.d_out = d_out
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = adPositionwiseFeedForward(d_model=hidden, d_out=d_out, dropout=dropout)
        self.input_sublayer = adSublayerConnection(size=channel_in, dropout=dropout, norm=norm)
        self.output_sublayer = adSublayerConnection(size=channel_in, dropout=dropout, norm=norm)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = norm(channel_in)

    def forward(self, x, mask=None):

        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))

        if self.hidden == self.d_out:
            x = self.output_sublayer(x, self.feed_forward)
        else:
            x = self.norm(x)
            x = self.feed_forward(x)
        return self.dropout(x)

class adSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, norm=nn.LayerNorm):
        super(adSublayerConnection, self).__init__()
        self.norm = norm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class adTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, d_out, dropout=0, channel_in=16, norm=nn.LayerNorm):

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = adPositionwiseFeedForward(d_model=hidden, d_out=d_out, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = norm(channel_in)

    def forward(self, x, mask=None):
        x = self.attention.forward(x, x, x, mask=mask)
        x = self.norm(x)
        x = self.feed_forward(x)
        return self.dropout(x)



class adPositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_out, dropout=0.1):
        super(adPositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model*4)
        self.w_2 = nn.Linear(d_model*4, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
