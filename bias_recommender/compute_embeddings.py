"""
compute_embeddings.py
─────────────────────
Loads the already-built master_dataset.csv and computes
sentence-transformer embeddings, saved to data/embeddings.npy.

Run once:  python compute_embeddings.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR  = "data"
CSV_PATH  = os.path.join(DATA_DIR, "master_dataset.csv")
EMB_PATH  = os.path.join(DATA_DIR, "embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found. Run dataset_builder.py first.")
    sys.exit(1)

if os.path.exists(EMB_PATH):
    print(f"Embeddings already exist at {EMB_PATH}. Delete it to recompute.")
    sys.exit(0)

print(f"Loading master dataset from {CSV_PATH}...")
master = pd.read_csv(CSV_PATH, index_col="idx")
print(f"Loaded {len(master):,} videos.")

print(f"Loading sentence-transformer model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

texts = (
    master["title"].fillna("").astype(str)
    + " [" + master["genre"] + "]"
    + " [" + master["language"] + "]"
).tolist()

print(f"Computing embeddings for {len(texts):,} videos...")
embeddings = model.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True,
)

np.save(EMB_PATH, embeddings)
print(f"Embeddings saved: {embeddings.shape} -> {EMB_PATH}")
