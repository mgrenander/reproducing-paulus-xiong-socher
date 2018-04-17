from torchtext import data
import pandas as pd
import sys
import os
from tqdm import tqdm

base_path = "data/finished_files"
def convert_to_tsv(dataset):
    art_path = os.path.join(base_path, "article", dataset)
    ref_path = os.path.join(base_path, "reference", dataset)

    df = pd.DataFrame(columns=['article', 'reference'], dtype=str)
    for i in tqdm(range(len(os.listdir(art_path)))):
        article_name = str(i) + "_" + dataset + "_art.txt"
        ref_name = str(i) + "_" + dataset + "_ref.txt"
        article = open(os.path.join(art_path, article_name), encoding='utf-8')
        reference = open(os.path.join(ref_path, ref_name), encoding='utf-8')

        row = {'article': article.read(), 'reference': reference.read()}
        df = df.append(row, ignore_index=True)

    print(df)

    df.to_csv(path_or_buf=os.path.join("data", dataset + ".tsv"), sep='\t', columns=['article', 'reference'],
              header=False, index=False, chunksize=1000)

if __name__ == "__main__":
    datasets = ["test"]
    for dataset in datasets:
        if not os.path.exists(os.path.join("data", dataset + ".tsv")):
            print("Creating TSV for " + dataset)
            convert_to_tsv(dataset)

