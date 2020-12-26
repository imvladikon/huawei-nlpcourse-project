from __future__ import unicode_literals, print_function, division

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from processing.transformer_utils.multiheadattn import MultiHeadAttn

MAX_LENGTH = 30
SOS_token = 0
EOS_token = 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.multiheadattn = MultiHeadAttn(8, self.hidden_size, 128, dropout=0.1)
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gru = nn.GRU(self.input_size, self.hidden_size).to(self.device)

    def forward(self, input, batch_size=None, hidden=None):
        if (len(input.shape) == 2):
            input = input.unsqueeze(1)
        if batch_size is None:
            batch_size = 1
        if hidden is None:
            hidden = self.init_hidden(batch_size).to(self.device)
        output = input
        output, hidden = self.gru(output, hidden)
        output = self.multiheadattn(output)
        # output = output[i_s, b_is, :]
        b_is = torch.arange(self.batch_size).reshape((self.batch_size, 1)).squeeze()
        output = output[-1:, b_is, :]
        output = F.relu(output)

        if (len(output.shape) == 1):
            output = output.unsqueeze(0).unsqueeze(1)
        elif (len(output.shape) == 2):
            output = output.unsqueeze(0)
        return output

    def to(self, *args, **kwargs):
        super(EncoderRNN, self).to(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'device' in kwargs:
            self.device = kwargs['device']
        elif args is not None and len(args) > 0:
            self.device = args[0]
        return self

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def init_from_state_dict(self, state_dict):
        td = {k: v for k, v in self.named_parameters() if 'encoder.' + k in state_dict}
        self.load_state_dict(td)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, input_size)
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input, batch_size=None, hidden=None):
        if batch_size is None:
            batch_size = 1
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        if len(input.shape) == 2:
            input = input.unsqueeze(1)
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
            input = input.unsqueeze(1)
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])

        return output, hidden

    def to(self, *args, **kwargs):
        super(DecoderRNN, self).to(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'device' in kwargs:
            self.device = kwargs['device']
        elif args is not None and len(args) > 0:
            self.device = args[0]
        return self

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, **kwargs):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.multiheadattn = MultiHeadAttn(8, self.hidden_size, 128, dropout=0.1)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input, batch_size=None, hidden=None):  # encoder_outputs, hidden,  i_s, b_is):
        if batch_size is None:
            batch_size = 1
        if hidden is None:
            hidden = self.init_hidden(batch_size).to(self.device)
        outputs = torch.zeros((self.max_length, self.batch_size, self.output_size)).to(self.device)
        output = input
        for i in range(input.shape[0]):
            output, hidden = self.gru(output, hidden)
            o = self.out(output[0])
            outputs[i, :, :] = o
        return outputs

    def to(self, *args, **kwargs):
        super(AttnDecoderRNN, self).to(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'device' in kwargs:
            self.device = kwargs['device']
        elif args is not None and len(args) > 0:
            self.device = args[0]
        return self

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class GRUAE(nn.Module):
    def __init__(self, input_size, hidden_size, teacher_forcing_ratio, **kwargs):
        super(GRUAE, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = EncoderRNN(input_size, hidden_size, **kwargs)
        self.decoder = AttnDecoderRNN(hidden_size, input_size, **kwargs)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.logger = logging.getLogger(self.__class__.__name__)

    def to(self, *args, **kwargs):
        super(GRUAE, self).to(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'device' in kwargs:
            self.device = kwargs['device']
        elif args is not None and len(args) > 0:
            self.device = args[0]
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        return self

    def forward(self, tensor: torch.Tensor):
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        output = self.encoder(tensor)
        output = self.decoder(output)
        return output

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters if p is not None)
        return super(GRUAE, self).__str__() + '\nTrainable parameters: {}'.format(params)
