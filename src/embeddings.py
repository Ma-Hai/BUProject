import time
import os

import numpy as np
from tqdm import tqdm

import pandas as pd
import openai
from dotenv import load_dotenv

load_dotenv()

assert "OPENAI_API_KEY" in os.environ, "Please ensure an OPENAI_API_KEY is set."

CLIENT = openai.OpenAI()

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

def normalize(arr: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    return arr / np.sqrt(np.sum(arr ** 2, axis=-1, keepdims=True))

def embedding_batch(text: str | list[str], path: str) -> np.ndarray[np.float32]:
    """large-scale embedding function, for preprocessing"""
    if os.path.exists(path):
        print("Aborting to avoid file overwrite.")
        return np.lib.format.open_memmap(path)

    out = np.lib.format.open_memmap(
        path, 
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
