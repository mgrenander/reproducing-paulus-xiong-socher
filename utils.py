import os
from tqdm import tqdm
import spacy
from datetime import datetime

spacy_en = spacy.load('en')
base_path = "data"
base_read_path = "data/finished_files"
max_input_len = 800
max_output_len = 100
def convert_to_tsv(dataset):
    art_path = os.path.join(base_read_path, "article", dataset)
    ref_path = os.path.join(base_read_path, "reference", dataset)

    # Remove previous version
    open(os.path.join(base_path, dataset + ".tsv"), 'w').close()

    f = open(os.path.join(base_path, dataset + ".tsv"), 'a', encoding='utf-8')
    for i in tqdm(range(len(os.listdir(art_path)))):
        article_name = str(i) + "_" + dataset + "_art.txt"
        ref_name = str(i) + "_" + dataset + "_ref.txt"
        article = open(os.path.join(art_path, article_name), encoding='utf-8')
        reference = open(os.path.join(ref_path, ref_name), encoding='utf-8')

        f.write(article.read() + "\t" + reference.read() + "\n")
    f.close()


def tokenizer_in(text):
    """Tokenizer. Note we limit to top 800 tokens, as per Paulus et al."""
    return [tok.text for tok in spacy_en(text)[:max_input_len]]


def tokenizer_out(text):
    """Tokenizer. Note we limit to top 100 tokens"""
    return [tok.text for tok in spacy_en(text)[:max_output_len]]

def get_time_diff(curr_time):
    return (datetime.now() - curr_time).seconds / 60.0, datetime.now()