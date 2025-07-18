
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_yelp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_b = pd.read_json("../project/data/yelp_dataset/yelp_academic_dataset_business.json", lines=True)
    df_b = df_b[df_b["city"].str.lower() == "philadelphia"]
    philadelphia_businesses = set(df_b["business_id"])
    df_r_chunks = pd.read_json("../project/data/yelp_dataset/yelp_academic_dataset_review.json", lines=True, chunksize=100000)
    df_r = pd.concat([chunk[np.array([[b_id in philadelphia_businesses] for b_id in chunk["business_id"]], dtype=bool)] for chunk in tqdm(df_r_chunks, total=70)])
    return df_b, df_r