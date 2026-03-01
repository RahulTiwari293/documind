import os
import requests
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("ENDEE_URL", "http://localhost:8080")
auth_token = os.getenv("ENDEE_AUTH_TOKEN", "")


def _headers():
    h = {"Content-Type": "application/json"}
    if auth_token:
        h["Authorization"] = auth_token
    return h


def create_index(name, dims=384, metric="cosine", precision="FLOAT32"):
    """
    Create a named vector index.

    precision options (Endee quantization levels):
      BINARY   – smallest, fastest, lowest accuracy
      INT8     – 4x smaller than FLOAT32, great for large corpora
      INT16    – balanced compression
      FLOAT16  – half precision, near-FLOAT32 quality
      FLOAT32  – full precision (default for best accuracy)

    metric options: cosine | L2 | inner_product
    """
    payload = {
        "name": name,
        "dimension": dims,
        "metric": metric,
        "precision": precision
    }
    resp = requests.post(f"{base_url}/api/v1/index/create", json=payload, headers=_headers())
    resp.raise_for_status()
    return resp.json()


def upsert(index_name, vectors):
    """
    Insert or update vectors in an index.
    Each vector dict: { id, vector, metadata }
    Metadata can hold any key-value pairs for filtering.
    """
    payload = {"vectors": vectors}
    resp = requests.post(f"{base_url}/api/v1/index/{index_name}/upsert", json=payload, headers=_headers())
    resp.raise_for_status()
    return resp.json()


def search(index_name, query_vector, top_k=5, filters=None):
    """
    Approximate Nearest Neighbor search.

    filters use Endee's query operators:
      $eq    – exact match        : {"source": {"$eq": "report.pdf"}}
      $in    – value in list      : {"category": {"$in": ["finance", "legal"]}}
      $range – numeric range      : {"chunk_id": {"$range": [0, 10]}}

    Returns ranked results with cosine similarity scores.
    """
    payload = {"vector": query_vector, "top_k": top_k}
    if filters:
        payload["filter"] = filters
    resp = requests.post(f"{base_url}/api/v1/index/{index_name}/search", json=payload, headers=_headers())
    resp.raise_for_status()
    return resp.json()


def list_indexes():
    resp = requests.get(f"{base_url}/api/v1/index/list", headers=_headers())
    resp.raise_for_status()
    return resp.json()


def delete_index(index_name):
    resp = requests.delete(f"{base_url}/api/v1/index/{index_name}", headers=_headers())
    resp.raise_for_status()
    return resp.json()


def index_info(index_name):
    """Fetch metadata and stats for a single index."""
    resp = requests.get(f"{base_url}/api/v1/index/{index_name}", headers=_headers())
    resp.raise_for_status()
    return resp.json()
