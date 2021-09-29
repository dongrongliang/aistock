import torch
import torch.nn as nn
from cnn.embedding import stockEmbedding

# simple embedding
# adam
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, feature_num, class_num, ninp, nhid, nlayers, short_period=8, tie_weights=False):
        super(RNNModel, self).__init__()

        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'

        self.embedding = stockEmbedding(0, feature_num, ninp, dropout=0, order_emb=False)
        self.short_period = short_period
        if rnn_type == 'LSTM':
            self.rnns = [
                torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                              1, dropout=0) for l in range(nlayers)]
        else:
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l
                         in range(nlayers)]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, class_num)

        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, return_h=False):

        emb = self.embedding(input)  # input:NCL, embedding:NLC
        emb = emb.transpose(0, 1)  # emd L,N,C
        l, n, c = emb.size()
        period_num = int(l / self.short_period)
        hidden = self.init_hidden(n)
        for i in range(period_num):

            raw_output = emb[i:i+self.short_period, :, :]
            new_hidden = []
            # raw_output, hidden = self.rnn(emb, hidden)
            raw_outputs = []
            for l, rnn in enumerate(self.rnns):
                current_input = raw_output
                raw_output, new_h = rnn(raw_output, hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
            hidden = new_hidden

        output = raw_output.transpose(0, 1)  # N,L,C
        output = self.decoder(output[:, -1])

        if return_h:
            return output, raw_outputs
        return output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                         self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]