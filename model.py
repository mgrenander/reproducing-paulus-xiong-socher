import torch
from torch.nn import Module, LSTM, Embedding, Linear
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderLSTM(Module):
    def __init__(self, input_size, embed_size, hidden_size, use_gpu=False, gpu_device=0, batch_size=50):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = Embedding(input_size, embed_size)
        self.lstm = LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)

        # GPU settings
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

    def init_hidden(self):
        # Tuple for initialization of LSTM: (hidden state, cell state)
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda(device=self.gpu_device)),
                            Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda(device=self.gpu_device)))
        else:
            self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                            Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def forward(self, batch):
        embeds = self.embedding(batch.text)
        out, hidden = self.lstm(embeds, self.hidden)
        return out, hidden

        # if not self.training:  # Set repeated trigram to false


class DecoderLSTM(Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, use_gpu=False, gpu_device=0, batch_size=50):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = Embedding(input_size, embed_size)
        self.lstm = LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.linear = Linear(hidden_size, output_size)

        # GPU
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

    def init_hidden(self):
        # Tuple for initialization of LSTM: (hidden state, cell state)
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda(device=self.gpu_device)),
                            Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda(device=self.gpu_device)))
        else:
            self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                            Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def forward(self, batch, prev_hidden):
        embeds = self.embedding(batch.text)
        out, hidden = self.lstm(embeds, prev_hidden)
        output = self.linear(out[-1])
        return output, hidden
