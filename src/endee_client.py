import os
import requests
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("ENDEE_URL", "http://localhost:8080")
auth_token = os.getenv("ENDEE_AUTH_TOKEN", "")

def headers():
    h = {"Content-Type": "application/json"}
    if auth_token:
        h["Authorization"] = auth_token
    return h

def create_index(name, dims=384, metric="cosine"):
    payload = {
        "name": name,
        "dimension": dims,
        "metric": metric,
        "precision": "FLOAT32"
    }
    resp = requests.post(f"{base_url}/api/v1/index/create", json=payload, headers=headers())
    resp.raise_for_status()
    return resp.json()

def upsert(index_name, vectors):
    payload = {"vectors": vectors}
    resp = requests.post(f"{base_url}/api/v1/index/{index_name}/upsert", json=payload, headers=headers())
    resp.raise_for_status()
    return resp.json()

def search(index_name, query_vector, top_k=5, filters=None):
    payload = {"vector": query_vector, "top_k": top_k}
    if filters:
        payload["filter"] = filters
    resp = requests.post(f"{base_url}/api/v1/index/{index_name}/search", json=payload, headers=headers())
    resp.raise_for_status()
    return resp.json()

def list_indexes():
    resp = requests.get(f"{base_url}/api/v1/index/list", headers=headers())
    resp.raise_for_status()
    return resp.json()

def delete_index(index_name):
    resp = requests.delete(f"{base_url}/api/v1/index/{index_name}", headers=headers())
    resp.raise_for_status()
    return resp.json()
