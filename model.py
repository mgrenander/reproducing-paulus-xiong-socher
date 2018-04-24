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
        self.hidden = None

        # GPU settings
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

    def init_hidden(self):
        # Tuple for initialization of LSTM: (hidden state, cell state)
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda(device=self.gpu_device)),
                           Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda(device=self.gpu_device)))
        else:
            self.hidden = (Variable(torch.zeros(2, self.batch_size, self.hidden_size)),
                           Variable(torch.zeros(2, self.batch_size, self.hidden_size)))

    def forward(self, art_batch):
        embeds = self.embedding(art_batch)
        out, hidden = self.lstm(embeds, self.hidden)
        self.hidden = hidden  # Update hidden
        return out, hidden


class DecoderLSTM(Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, use_gpu=False, gpu_device=0, batch_size=50):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = Embedding(input_size, embed_size)
        self.lstm = LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.linear = Linear(hidden_size, output_size)
        self.hidden = None

        # GPU
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

    def init_hidden(self, prev_hidden):
        self.hidden = prev_hidden

    def forward(self, batch_summ):
        embeds = self.embedding(batch_summ).unsqueeze(0)
        out, self.hidden = self.lstm(embeds, self.hidden)
        output = F.log_softmax(self.linear(out.squeeze(0)), dim=1)
        return output

        # if not self.training:  # Set repeated trigram to false
