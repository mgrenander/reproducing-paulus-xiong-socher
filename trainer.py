import torch
from torchtext import data
from utils import *

DEVICE = 0

# PREPROCESSING
datasets = ["train", "val", "test"]
for dataset in datasets:
    if not os.path.exists(os.path.join("data", dataset + ".tsv")):
        print("Creating TSV for " + dataset)
        convert_to_tsv(dataset)

ARTICLE = data.Field(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_in, unk_token=None)
SUMMARY = data.Field(tensor_type=torch.cuda.LongTensor, lower=True, tokenize=tokenizer_out, unk_token=None)
train, val, test = data.TabularDataset.splits(path='./data/', train='train.tsv', validation='val.tsv', test='test.tsv',
                                              format='tsv', fields=[('Article', ARTICLE), ('Summary', SUMMARY)])
ARTICLE.build_vocab(train, vectors="glove.6B.100d")
SUMMARY.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), sort_key=lambda x: len(x.Text), batch_size=50, repeat=False, device=DEVICE)

# Need to build model and set initial embeddings to ARTICLE.vocab.vectors