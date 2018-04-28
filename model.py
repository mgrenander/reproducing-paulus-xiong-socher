import torch
from torch.nn import Module, LSTM, Embedding, Linear, Bilinear
from torch.autograd import Variable
import torch.nn.functional as F
from utils import get_dec_context_vector, get_enc_context_vector

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
        embeds = self.embedding(art_batch).unsqueeze(0)
        out, hidden = self.lstm(embeds, self.hidden)
        self.hidden = hidden  # Update hidden
        return out, hidden


class AttnDecoderLSTM(Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, use_gpu=False, gpu_device=0, batch_size=50):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = Embedding(input_size, embed_size)
        self.lstm = LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.w_out = Linear(hidden_size, output_size)
        self.w_u = Linear(hidden_size, 1)
        self.hidden = None

        # For attention
        self.enc_bilinear = Bilinear(hidden_size, hidden_size, 1)
        self.dec_bilinear = Bilinear(hidden_size, hidden_size, 1)

        # GPU
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

    def init_hidden(self, prev_hidden):
        self.hidden = prev_hidden

    def forward(self, batch_summ, encoder_outputs, decoder_outputs):
        embeds = self.embedding(batch_summ).unsqueeze(0)
        out, self.hidden = self.lstm(embeds, self.hidden)

        # Context vectors
        enc_scores = self.enc_bilinear(self.hidden[0].squeeze(0), encoder_outputs)
        dec_scores = self.dec_bilinear(self.hidden[0].squeeze(0), decoder_outputs) if decoder_outputs is not None else None
        enc_context, enc_att_weights = get_enc_context_vector(enc_scores, encoder_outputs)
        dec_context = get_dec_context_vector(dec_scores, decoder_outputs)

        out = torch.cat(out, enc_context, dec_context, dim=1)

        prob = F.log_softmax(self.w_out(out.squeeze(0)), dim=1)
        return prob, self.hidden
