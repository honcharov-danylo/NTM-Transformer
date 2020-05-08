"""Transformer Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.num_inputs = num_inputs
#         self.num_outputs = num_outputs
#         self.num_layers = num_layers
#
#         self.transformer = nn.Transformer()#(input_size=num_inputs,
#                             # hidden_size=num_outputs,
#                             # num_layers=num_layers)
#
#         # The hidden state is a learned parameter
#         self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
#         self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
#
#         self.reset_parameters()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # print(pe.shape)
        # print(position.shape)
        # print(div_term.shape)
        # print((position*div_term).shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerController(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_layers, nhid = 2 , dropout=0.1, nhead=29):
        # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerController, self).__init__()
        # print(num_inputs, num_outputs, num_layers)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        #self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(num_inputs, dropout)
        #print(num_inputs)
        encoder_layers = TransformerEncoderLayer(num_inputs, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        #self.encoder = nn.Embedding(ntoken, num_inputs)

        # print('num inputs',num_inputs)
        # print('num outputs', num_outputs)
        #self.decoder = nn.Linear(num_inputs, num_outputs)
        self.decoder = nn.Linear(num_inputs, num_outputs)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.transformer_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, none_state):
        if none_state is not None:
            # print('src', src.shape)
            # print('state',none_state.shape)
            src = torch.cat((src.unsqueeze(0), none_state),0)
            #print('src shape:',src.shape)
        #print(src.shape)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #print(self.src_mask.shape)
        #print(src.shape)
        #src = self.encoder(src.long()) * math.sqrt(self.num_inputs)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        #print(output.shape)
        #src = torch.cat((src, output.sum(axis=0).unsqueeze(0)), 0)
        output = output[-1]
        output = self.decoder(output)#.unsqueeze(0))
        #output = output.sum(axis=0)
        #print('output shape',output.shape)
        #print(output.shape)
        #exit()
        #print(output.shape, src.shape)
        return output, src # for the consistency

    def size(self):
        return self.num_inputs, self.num_outputs

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        # lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        # lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return None

    def reset_parameters(self):
        for p in self.transformer.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

# class TransformerController(nn.Module):
#     """An NTM controller based on LSTM."""
#     def __init__(self, num_inputs, num_outputs, num_layers):
#         super(TransformerController, self).__init__()
#
#         self.num_inputs = num_inputs
#         self.num_outputs = num_outputs
#         self.num_layers = num_layers
#
#         self.transformer = nn.Transformer()#(input_size=num_inputs,
#                             # hidden_size=num_outputs,
#                             # num_layers=num_layers)
#
#         # The hidden state is a learned parameter
#         self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
#         self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
#
#         self.reset_parameters()
#
#     def create_new_state(self, batch_size):
#         # Dimension: (num_layers * num_directions, batch, hidden_size)
#         lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
#         lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
#         return lstm_h, lstm_c
#
#     def reset_parameters(self):
#         for p in self.transformer.parameters():
#             if p.dim() == 1:
#                 nn.init.constant_(p, 0)
#             else:
#                 stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
#                 nn.init.uniform_(p, -stdev, stdev)
#
#     def size(self):
#         return self.num_inputs, self.num_outputs
#
#     def forward(self, x, prev_state):
#         # x = x.unsqueeze(0)
#         # outp, state = self.transformer(torch.tensor(x))#, prev_state)
#         # return outp.squeeze(0), state
#         x = x.unsqueeze(0)
#
#         outp, state = self.transformer(torch.tensor(x))  # , prev_state)
#         return outp.squeeze(0), state