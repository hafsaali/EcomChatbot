import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path


# Constants
DATA_PATH = Path("data/processed_data.json")
EMBEDDINGS_PATH = Path("data/product_embeddings.npy")
ID_PATH = Path("data/product_ids.json")

# Load model once
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load products
def _load_products():
    products = json.loads(DATA_PATH.read_text())
    product_ids = [p["_id"] for p in products]
    product_texts = [f"{p['title']} {p['description']}" for p in products]

    # Load or compute embeddings
    if EMBEDDINGS_PATH.exists() and ID_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        saved_ids = json.loads(ID_PATH.read_text())
        assert saved_ids == product_ids, "Mismatch in product IDs!"
    else:
        embeddings = encoder.encode(product_texts, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        ID_PATH.write_text(json.dumps(product_ids))

    return product_ids, embeddings, product_texts, products

product_ids, product_embeddings, product_texts, PRODUCTS = _load_products()

# Build FAISS index
dimension = product_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(product_embeddings).astype("float32"))

id_map = {i: product_ids[i] for i in range(len(product_ids))}

# Retrieve products
def get_products_for_query(query: str, top_k: int = 3) -> List[Dict]:
    query_embedding = encoder.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), len(PRODUCTS))
    results = []
    for i in indices[0]:
        product = next(p for p in PRODUCTS if p["_id"] == id_map[i])
        price = float(product["selling_price"].replace(",", ""))
        if price < 1000:
            results.append(product)
        if len(results) >= top_k:
            break
    return results
