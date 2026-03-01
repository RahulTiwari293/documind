import os
import json
import requests
import msgpack
from src.config import get_secret

base_url = get_secret("ENDEE_URL", "ENDEE_URL") or "http://localhost:8080"
auth_token = get_secret("ENDEE_AUTH_TOKEN", "ENDEE_AUTH_TOKEN") or ""

PRECISION_MAP = {
    "FLOAT32": "float32",
    "FLOAT16": "float16",
    "INT16":   "int16",
    "INT8":    "int8",
    "BINARY":  "binary",
}

SPACE_MAP = {
    "cosine":        "cosine",
    "L2":            "l2",
    "inner_product": "ip",
}


def _headers(content_type="application/json"):
    h = {"Content-Type": content_type}
    if auth_token:
        h["Authorization"] = auth_token
    return h


def create_index(name, dims=384, metric="cosine", precision="FLOAT32"):
    payload = {
        "index_name": name,
        "dim": dims,
        "space_type": SPACE_MAP.get(metric, "cosine"),
        "precision": PRECISION_MAP.get(precision, "float32")
    }
    resp = requests.post(f"{base_url}/api/v1/index/create", json=payload, headers=_headers())
    resp.raise_for_status()
    return resp.text


def insert(index_name, vectors):
    """
    vectors: list of dicts with keys:
      id     – unique string ID
      vector – list of floats (must match index dims)
      meta   – dict (will be JSON-serialised to string for Endee)
      filter – optional dict for metadata filtering (serialised to JSON string)
    """
    payload = []
    for v in vectors:
        entry = {
            "id": v["id"],
            "vector": v["vector"],
            "meta": json.dumps(v.get("metadata", {})),
        }
        if v.get("filter"):
            entry["filter"] = json.dumps(v["filter"])
        payload.append(entry)

    resp = requests.post(
        f"{base_url}/api/v1/index/{index_name}/vector/insert",
        json=payload,
        headers=_headers()
    )
    resp.raise_for_status()
    return resp.status_code


def search(index_name, query_vector, top_k=5, filters=None):
    """
    filters: list of filter dicts in Endee array format, e.g.:
      [{"source": {"$eq": "report.pdf"}}]
      [{"chunk_id": {"$range": [0, 10]}}]
      [{"source": {"$in": ["a.pdf", "b.pdf"]}}]

    Returns a decoded list of result dicts from msgpack response.
    """
    payload = {
        "vector": query_vector,
        "k": top_k
    }
    if filters:
        payload["filter"] = json.dumps(filters)

    resp = requests.post(
        f"{base_url}/api/v1/index/{index_name}/search",
        json=payload,
        headers=_headers()
    )
    resp.raise_for_status()

    raw = msgpack.unpackb(resp.content, raw=False)
    return raw


def list_indexes():
    resp = requests.get(f"{base_url}/api/v1/index/list", headers=_headers())
    resp.raise_for_status()
    return resp.json()


def delete_index(index_name):
    resp = requests.delete(f"{base_url}/api/v1/index/{index_name}/delete", headers=_headers())
    resp.raise_for_status()
    return resp.text


def index_info(index_name):
    resp = requests.get(f"{base_url}/api/v1/index/{index_name}/info", headers=_headers())
    resp.raise_for_status()
    return resp.json()
