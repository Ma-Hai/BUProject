
import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_yelp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_b = pd.read_json("data/yelp_dataset/yelp_academic_dataset_business.json", lines=True)
    df_b = df_b[df_b["city"].str.lower() == "philadelphia"]
    philadelphia_businesses = set(df_b["business_id"])
    df_r_chunks = pd.read_json("data/yelp_dataset/yelp_academic_dataset_review.json", lines=True, chunksize=100000)
    df_r = pd.concat([chunk[np.array([[b_id in philadelphia_businesses] for b_id in chunk["business_id"]], dtype=bool)] for chunk in tqdm(df_r_chunks, total=70)])
    return df_b, df_r

def load_yelp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_b = pd.read_csv("data/df_b_processed.csv")
    df_r = pd.read_csv("data/df_r_processed.csv")
    return df_b, df_r

if __name__ == "__main__":
    df_b, df_r = parse_yelp_data()
    df_b.to_csv("data/df_b_processed.csv")
    df_r.to_csv("data/df_r_processed.csv")