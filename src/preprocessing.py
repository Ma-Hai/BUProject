import tarfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import embeddings

def preprocess() -> None:
    print("Extracting Yelp dataset...")
    with tarfile.open("data/input_data/yelp_dataset.tar") as yelp_compressed:
        yelp_compressed.extractall("data/intermediate_data/")

    df_b = pd.read_json("data/intermediate_data/yelp_dataset/yelp_academic_dataset_business.json", lines=True)
    df_b = df_b[df_b["city"].str.lower() == "philadelphia"]
    philadelphia_businesses = set(df_b["business_id"])
    df_r_chunks = pd.read_json("data/intermediate_data/yelp_dataset/yelp_academic_dataset_review.json", lines=True, chunksize=100000)
    df_r = pd.concat([chunk[np.array([[b_id in philadelphia_businesses] for b_id in chunk["business_id"]], dtype=bool)] for chunk in tqdm(df_r_chunks, total=70)])
    
    df_r["text"] = df_r["text"].str.replace(r"\.{3,}", "...", regex=True)
    
    df_b.to_csv("data/analysis_data/df_b_processed.csv")
    df_r.to_csv("data/analysis_data/df_r_processed.csv")

    print("Yelp dataset extracted and saved.")
    print("Computing embeddings, this may take a while...")

    embeddings.embedding_batch(list(df_r["text"]), "analysis_data/r_emb_large.npy")

    print("Embeddings complete.")

if __name__ == "__main__":
    preprocess()

