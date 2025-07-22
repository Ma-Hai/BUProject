import time
import os

import numpy as np
from tqdm import tqdm

import pandas as pd
import openai
from dotenv import load_dotenv

load_dotenv()

CLIENT = openai.OpenAI()

def cos_sim(
        query: np.ndarray[np.float32], 
        keys: np.ndarray[np.float32], 
        batch_size: int | None = 10000,
        prog_bar: bool = True,
    ) -> np.ndarray[np.float32]:
    """Compute cosine similarity between a query vector and a batch of key vectors.
    Args:
        query: A 1d numpy array representing the query vector.
        keys: A 2d numpy array where each row is a key vector.
        batch_size: The number of keys to process in each batch. If None, will process the whole thing at once.
        prog_bar: Whether to show a progress bar.
    Returns:
        A 1d numpy array of cosine similarity values, one for each key vector.
    """
    
    if batch_size is None:
        batch_size = len(keys)

    # disable progress bar if only one batch
    if batch_size >= len(keys):
        prog_bar = False

    vals = np.zeros(len(keys))
    for i in tqdm(range(0, len(keys), batch_size), disable=not prog_bar):
        np.matmul(keys[i: i + batch_size], query, out=vals[i: i + batch_size])

    return vals

def embedding(text: str | list[str]) -> np.ndarray[np.float32]:
    """Text -> vector embedding using OpenAI API.
    Args:
        text: A string or list of strings to embed
    Returns:
        A numpy array of embeddings. 2d if a list was passed, else 1d.
    Notes:
        If you get some quota error, ping me - the account must be out of money.
        If you get a rate limit error, try breaking up your input to avoid sending in as much.
        Remember, this function costs money to run.
    """
    if isinstance(text, str):
        return embedding([text])[0]
    response = CLIENT.embeddings.create(
        input=text,
        model="text-embedding-3-large",
    )
    return np.array([entry.embedding for entry in response.data], dtype=np.float32)

def _embedding_long(text: str | list[str]) -> np.ndarray[np.float32]:
    """large-scale embedding function, DO NOT USE"""
    if isinstance(text, str):
        return _embedding_long([text])[0]
    assert not os.path.exists("data/r_emb_large_03.npy"), "ABORTING TO AVOID FILE OVERWRITE"
    out = np.lib.format.open_memmap(
        "data/r_emb_large_03.npy", 
        mode="w+", 
        dtype=np.float32, 
        shape=(len(text), 3072),
    )
    
    for i in tqdm(range(0, len(text), 1000)):
        batch = text[i: i + 1000]
        wait = 5
        done = False
        while not done:
            try:
                response = CLIENT.embeddings.create(
                    input=batch,
                    model="text-embedding-3-large",
                )
                out[i: i + 1000] = np.array([entry.embedding for entry in response.data], dtype=np.float32)
                out.flush()
                done = True
            except openai.RateLimitError as e:
                print(f"hit rate limit, retrying after {wait}s")
                print("Error:", e)
                time.sleep(wait)
                wait *= 2
                
    return out
    

if __name__ == "__main__":
    assert False, "DO NOT RUN THIS! Just download it."
    df_r = pd.read_csv("data/df_r_processed.csv").set_index("review_id")
    df_r["text"] = df_r["text"].str.replace(r"\.{3,}", "...", regex=True)
    out = _embedding_long(list(df_r["text"]))
    breakpoint()
