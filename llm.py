"""
LLM client setup for the Insurance Claim Processing Agent.

Uses GROQ API (e.g., Llama3) via langchain-groq. If no GROQ_API_KEY is
configured, the app still runs using deterministic rule-based fallbacks
inside each node (see nodes.py), so the demo works out-of-the-box.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama3-8b-8192")

def get_llm(temperature: float = 0.0):
    """
    Returns a configured ChatGroq client, or None if no API key is present.
    Nodes should check for None and fall back to rule-based logic.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return ChatGroq(model_name=MODEL_NAME, temperature=temperature, groq_api_key=api_key)
    except Exception:
        return None


def llm_available() -> bool:
    return get_llm() is not None
