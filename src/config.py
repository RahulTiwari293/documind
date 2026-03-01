import streamlit as st
import os

def get_secret(key: str, fallback_env: str = None) -> str:
    """Read from Streamlit secrets first, then env vars."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(fallback_env or key, "")
