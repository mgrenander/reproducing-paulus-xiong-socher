import torch
from torch import optim
from torch import nn
from torchtext import data
from utils import *
import sys
from model import *
from evaluate import evaluate
from tqdm import tqdm
from datetime import datetime
from random import random
# Constants
DEVICE = int(sys.argv[1])
encoder_vocab_size = 150000
decoder_vocab_size = 50000
embed_size = 100
encoder_hidden_size = 200
decoder_hidden_size = 400
lr = 0.0001
batch_size = 50
teacher_forcing_ratio = 0.25
num_epochs = 1
beam_size = 5


def encode_inputs(encoder, batch_art):
    input_length = batch_art.size(0)  # Length of longest input sequence
    batch_size = batch_art.size(1)
    encoder.init_hidden()
    out, hidden = None, None
    encoder_hidden_states = Variable(torch.zeros(input_length, batch_size, decoder_hidden_size)).cuda(device=DEVICE)
    for w in range(input_length):
        out, hidden = encoder(batch_art[w])
        encoder_hidden_states[w] = hidden[0].view(1, batch_size, decoder_hidden_size).squeeze(0)  # Don't need cell state for attn weights
    # Reshape because we are going from bidirectional to unidirectional LSTM
    hidden = (hidden[0].view(1, batch_size, decoder_hidden_size), hidden[1].view(1, batch_size, decoder_hidden_size))
    return out, hidden, encoder_hidden_states


def decode_outputs(decoder, prev_hidden, encoder_hidden_states, batch_summ, loss_fn, teacher_forcing_ratio):
    target_length = batch_summ.size(0)
    batch_size = batch_summ.size(1)
    curr_tok = batch_summ[0]
    decoder.init_hidden(prev_hidden)
    loss, decoder_out, hidden = 0, None, None
    decoder_hidden_states = Variable(torch.zeros(target_length - 1, batch_size, decoder_hidden_size)).cuda(device=0)
    for w in range(1, target_length):  # Iteration skips the SOS token at w=0
        decoder_hidden_so_far = None if w == 1 else decoder_hidden_states[:w - 1]  # Empty on first pass
        decoder_out, hidden = decoder(curr_tok, encoder_hidden_states, decoder_hidden_so_far)
        decoder_hidden_states[w - 1] = hidden[0]
        loss += loss_fn(decoder_out, batch_summ[w])
        # If not teacher forcing, we take decoder's generated token as next token instead of ground truth
        if random() < teacher_forcing_ratio:
            _, top_ind = decoder_out.data.topk(1)
            curr_tok = Variable(top_ind.squeeze())
        else:
            curr_tok = batch_summ[w]
    return decoder_out, hidden, loss


def train(batch, encoder, decoder, enc_opt, dec_opt, loss_fn, teacher_forcing_ratio):
    # Initialize hidden states and gradients
    encoder.train()
    decoder.train()
    enc_opt.zero_grad()
    dec_opt.zero_grad()
    enc_output, enc_hidden, encoder_hidden_states = encode_inputs(encoder, batch.article)  # Run article through encoder
    dec_output, dec_hidden, loss = decode_outputs(decoder, enc_hidden, encoder_hidden_states, batch.summary, loss_fn, teacher_forcing_ratio)
    loss.backward()
    enc_opt.step()
    dec_opt.step()
    return loss / batch.summary.size(0)


def main():
    ###############################
    # PREPROCESSING
    ###############################
    datasets = ["train", "val", "test"]
    for dataset in datasets:
        if not os.path.exists(os.path.join("data", dataset + ".tsv")):
            print("Creating TSV for " + dataset)
            convert_to_tsv(dataset)

    print("Creating datasets", end='', flush=True)
    curr_time = datetime.now()

    article_field = data.ReversibleField(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_in)
    summary_field = data.ReversibleField(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_out, init_token='<sos>')

    train_set = data.TabularDataset(path='./data/train.tsv', format='tsv', fields=[('article', article_field), ('summary', summary_field)])
    val_set = data.TabularDataset(path='./data/val.tsv', format='tsv', fields=[('article', article_field), ('summary', summary_field)])

    diff_time, curr_time = get_time_diff(curr_time)
    print(", took {} min".format(diff_time))

    print("Building vocabulary and creating batches", end='', flush=True)
    article_field.build_vocab(train_set, vectors="glove.6B.100d", max_size=encoder_vocab_size)
    summary_field.build_vocab(train_set, max_size=decoder_vocab_size)

    train_iter = data.BucketIterator(dataset=train_set, batch_size=batch_size, sort_key=lambda x: len(x.article), repeat=False, device=DEVICE)
    val_iter = data.BucketIterator(dataset=val_set, batch_size=batch_size, sort_key=lambda x: len(x.article), repeat=False, device=DEVICE)

    diff_time, curr_time = get_time_diff(curr_time)
    print(", took {} min".format(diff_time))
    ###############################
    # MODEL CREATION
    ###############################
    print("Creating encoder and decoder models", end='', flush=True)
    encoder = EncoderLSTM(input_size=encoder_vocab_size, embed_size=embed_size, hidden_size=encoder_hidden_size,
                          use_gpu=True, gpu_device=DEVICE, batch_size=batch_size)
    encoder.embedding.weight.data = article_field.vocab.vectors
    encoder.cuda(device=DEVICE)

    decoder = AttnDecoderLSTM(input_size=encoder_vocab_size, embed_size=embed_size, hidden_size=decoder_hidden_size,
                          output_size=decoder_vocab_size, use_gpu=True, gpu_device=DEVICE, batch_size=batch_size)
    decoder.embedding.weight.data = article_field.vocab.vectors
    decoder.cuda(device=DEVICE)
    diff_time, curr_time = get_time_diff(curr_time)
    print(", took {} min".format(diff_time))

    # Loss and SGD optimizers
    loss_func = nn.NLLLoss(ignore_index=1)  # Ignore <pad> token
    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)

    ###############################
    # TRAINING
    ###############################
    print("Beginning training")
    tqdm_epoch = tqdm(range(num_epochs), desc="Epoch")
    for epoch in tqdm_epoch:
        train_iter.init_epoch()
        tqdm_batch = tqdm(train_iter, desc="Batch")
        for b_id, batch in enumerate(tqdm_batch):
            encoder.batch_size = batch.batch_size  # Fixes weird bug where we get batch sizes that are not batch_size
            decoder.batch_size = batch.batch_size
            avg_loss = train(batch, encoder, decoder, encoder_opt, decoder_opt, loss_func, teacher_forcing_ratio)

    ###############################
    # TESTING
    ###############################
    # Load test set
    print("Loading test set")
    test_set = data.TabularDataset(path='./data/test.tsv', format='tsv', fields=[('article', article_field), ('summary', summary_field)])
    test_iter = data.BucketIterator(dataset=test_set, batch_size=batch_size, sort_key=lambda x: len(x.article), repeat=False, device=DEVICE)
    print("Evaluating model")
    evaluate(encoder=encoder, decoder=decoder, dataset=test_iter, rev_field=article_field)

if __name__ == "__main__":
    main()