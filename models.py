import torch
import torch.nn as nn
from torch.autograd import Variable


class Baseline(nn.Module):
    def __init__(self,
                 vocab,
                 word_embed_size: int,
                 hidden_size: int,
                 use_cuda: bool,
                 inference: bool):

        super(Baseline, self).__init__()
        self.use_cuda = use_cuda
        self.vocab_size = len(vocab)
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.attention = False
        self.bilstm_layers = 3
        self.bilstm_input_size = word_embed_size
        self.bilstm_output_size = 2 * hidden_size
        self.word_emb = nn.Embedding(self.vocab_size,
                                     word_embed_size,
                                     padding_idx=0)
        self.bilstm = nn.LSTM(self.bilstm_input_size,
                              self.hidden_size,
                              num_layers=self.bilstm_layers,
                              batch_first=True,
                              dropout=0.1,
                              bidirectional=True)
        self.dropout = nn.Dropout(p=0.35)
        if self.attention:
            self.attention_size = self.bilstm_output_size * 2
            self.u_a = nn.Linear(self.bilstm_output_size, self.bilstm_output_size)
            self.w_a = nn.Linear(self.bilstm_output_size, self.bilstm_output_size)
            self.v_a_inv = nn.Linear(self.bilstm_output_size, 1, bias=False)
            self.linear_attn = nn.Linear(self.attention_size, self.bilstm_output_size)
        self.linear = nn.Linear(self.bilstm_output_size, self.hidden_size)
        self.pred = nn.Linear(self.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, input_tokens, raw_tokens, labels, phase):
        loss = 0.0
        preds = []
        batch_size, seq_len = input_tokens.size()
        self.init_hidden(batch_size, use_cuda=self.use_cuda)

        x_i = self.word_emb(input_tokens)
        x_i = self.dropout(x_i)

        hidden, (self.h_n, self.c_n) = self.bilstm(x_i, (self.h_n, self.c_n))
        _, _, hidden_size = hidden.size()

        for i in range(seq_len):
            nth_hidden = hidden[:, i, :]
            if self.attention:
                target = nth_hidden.expand(seq_len, batch_size, -1).transpose(0, 1)
                mask = hidden.eq(target)[:, :, 0].unsqueeze(2)
                attn_weight = self.attention(hidden, target, mask)
                context_vector = torch.bmm(attn_weight.transpose(1, 2), hidden).squeeze()

                nth_hidden = torch.tanh(self.linear_attn(torch.cat((nth_hidden, context_vector))))
            output = self.softmax(self.pred(self.linear(nth_hidden)))
            loss += self.criterion(output, labels[:, i])

            _, topi = output.topk(k=1, dim=1)
            pred = topi.squeeze()
            preds.append(pred)

        preds = torch.stack(torch.cat(preds, dim=0).split(batch_size), dim=1)

        return loss, preds

    def attention(self, source, target, mask=None):
        function_g = \
            self.v_a_inv(torch.tanh(self.u_a(source) + self.w_a(target)))
        if mask is not None:
            function_g.masked_fill_(mask, -1e4)
        return nn.functional.softmax(function_g, dim=1)

    def init_hidden(self, batch_size, use_cuda):
        zeros = Variable(torch.zeros(2*self.bilstm_layers, batch_size, self.hidden_size))
        if use_cuda:
            self.h_n = zeros.cuda()
            self.c_n = zeros.cuda()
        else:
            self.h_n = zeros
            self.c_n = zeros
        return self.h_n, self.c_n
