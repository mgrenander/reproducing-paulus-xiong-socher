from torch import optim
from torch import nn
from torchtext import data
from utils import *
import sys
from model import *
from tqdm import tqdm
from datetime import datetime
import random

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
# ARTICLE = data.Field(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_in, unk_token=None)
# SUMMARY = data.Field(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_out, unk_token=None)
# train, val, test = data.TabularDataset.splits(path='./data/', train='train.tsv', validation='val.tsv', test='test.tsv',
#                                               format='tsv', fields=[('Article', ARTICLE), ('Summary', SUMMARY)])
# ARTICLE.build_vocab(train, vectors="glove.6B.100d", max_size=encoder_vocab_size)
# SUMMARY.build_vocab(train, max_size=decoder_vocab_size)
#
# train_iter, val_iter, test_iter = data.BucketIterator.splits(
#     (train, val, test), sort_key=lambda x: len(x.Text), batch_size=50, repeat=False, device=DEVICE)

# TODO: remove this when ready for training
ARTICLE = data.Field(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_in, unk_token=None)
SUMMARY = data.Field(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_out, unk_token=None)
train, test = data.TabularDataset.splits(path='./data/', train='val.tsv', test='test.tsv', format='tsv',
                                         fields=[('Article', ARTICLE), ('Summary', SUMMARY)])

diff_time, curr_time = get_time_diff(curr_time)
print(", took {} min".format(diff_time))

print("Building vocabulary", end='', flush=True)
ARTICLE.build_vocab(train, vectors="glove.6B.100d", max_size=encoder_vocab_size)
SUMMARY.build_vocab(train, max_size=decoder_vocab_size)

diff_time, curr_time = get_time_diff(curr_time)
print(", took {} min".format(diff_time))

print("Creating batches", end='', flush=True)
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=50, repeat=False, device=DEVICE)
diff_time, curr_time = get_time_diff(curr_time)
print(", took {} min".format(diff_time))
###############################
# MODEL CREATION
###############################
print("Creating encoder and decoder models", end='', flush=True)
encoder = EncoderLSTM(input_size=encoder_vocab_size, embed_size=embed_size, hidden_size=encoder_hidden_size,
                      use_gpu=True, gpu_device=DEVICE, batch_size=batch_size)
encoder.embedding.weight.data = ARTICLE.vocab.vectors
encoder.cuda(device=DEVICE)

decoder = DecoderLSTM(input_size=encoder_vocab_size, embed_size=embed_size, hidden_size=decoder_hidden_size,
                      output_size=decoder_vocab_size, use_gpu=True, gpu_device=DEVICE, batch_size=batch_size)
decoder.embedding.weight.data = ARTICLE.vocab.vectors
decoder.cuda(device=DEVICE)
diff_time, curr_time = get_time_diff(curr_time)
print(", took {} min".format(diff_time))
# TODO: Load previously checkpointed encoder and decoder if they exists

# Loss and SGD optimizers
loss_func = nn.CrossEntropyLoss()
encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
decoder_opt = optim.Adam(decoder.parameters(), lr=lr)

###############################
# TRAINING
###############################
def train(batch, enc, dec, enc_opt, dec_opt, loss_f, teacher_forcing_ratio):
    # Initialize hidden states and gradients
    enc.train()
    dec.train()
    enc_opt.zero_grad()
    dec_opt.zero_grad()
    encoder.init_hidden()

    # TODO: not sure about this
    input_length = batch.text.size()[0]
    target_length = batch.label.size()[0]

    for w_i in range(input_length):
        enc_output, enc_hidden = encoder(batch.text)



print("Beginning training")
val_acc = -1
tqdm_epoch = tqdm(range(num_epochs), desc="Epoch")
for epoch in tqdm_epoch:
    train_iter.init_epoch()
    tqdm_batch = tqdm(train_iter, desc="Batch")
    for b_id, batch in enumerate(tqdm_batch):
        train(batch, encoder, decoder, encoder_opt, decoder_opt, loss_func, teacher_forcing_ratio)
diff_time, curr_time = get_time_diff(curr_time)
print("Evaluating model")