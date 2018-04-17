from torchtext import data
import pandas as pd
import sys
import os
from tqdm import tqdm

base_path = "data"
base_read_path = "data/finished_files"
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

        f.write(article.read() + "\t" + reference.read())
    f.close()

if __name__ == "__main__":
    datasets = ["train", "val", "test"]
    for dataset in datasets:
        if not os.path.exists(os.path.join("data", dataset + ".tsv")):
            print("Creating TSV for " + dataset)
            convert_to_tsv(dataset)

