from torchtext import data
import pandas as pd
import sys
import os

base_path = "data/finished_files"
def convert_to_tsv(dataset):
    art_path = os.path.join(base_path, "article", dataset)
    ref_path = os.path.join(base_path, "reference", dataset)

    df = pd.DataFrame(columns=['article', 'reference'], dtype=str)
    for i in range(len(os.listdir(art_path))):
        article_name = str(i) + "_" + dataset + "_art.txt"
        ref_name = str(i) + "_" + dataset + "_ref.txt"
        article = open(os.path.join(art_path, article_name), encoding='utf-8')
        reference = open(os.path.join(ref_path, ref_name), encoding='utf-8')

        row = pd.Series(index=['article', 'reference'], data=[article.read(), reference.read()], dtype=str)
        df.append(row)

    df.to_csv(path_or_buf=dataset + ".tsv", sep='\t', columns=['article', 'reference'], header=False, index=False, chunksize=1000)

dataset = sys.argv[1]
convert_to_tsv(dataset)
