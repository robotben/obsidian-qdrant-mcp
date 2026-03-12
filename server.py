#!/usr/bin/env python3
"""
Qdrant Semantic Search MCP Server
Embeds queries via Ollama (nomic-embed-text) and searches the Obsidian Qdrant collection.
Exposes an HTTP MCP endpoint compatible with claude.ai.
"""

import json
import logging
import os
from typing import Any

import requests
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://192.168.245.62:11434")
EMBED_MODEL      = os.getenv("EMBED_MODEL",       "nomic-embed-text")
QDRANT_HOST      = os.getenv("QDRANT_HOST",       "192.168.245.187")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT",   "6333"))
COLLECTION_NAME  = os.getenv("COLLECTION_NAME",   "obsidian")
DEFAULT_TOP_K    = int(os.getenv("DEFAULT_TOP_K", "5"))
MCP_PORT         = int(os.getenv("MCP_PORT",      "3000"))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="obsidian-qdrant",
    instructions=(
        "Semantic search over Ben's Obsidian Second-Brain vault. "
        "Use this to find notes related to a concept, topic, or question. "
        "Returns relevant note excerpts with file paths. "
        "Prefer this over keyword search when looking for conceptually related content."
    ),
)

@mcp.tool()
def search_vault(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Semantically search the Obsidian vault using vector similarity.
    
    Args:
        query: Natural language search query (e.g. 'notes about RAG pipelines', 
               'what have I written about risk analysis?', 'homelab networking setup')
        top_k: Number of results to return (default 5, max 20)
    
    Returns:
        Matching note excerpts with file paths and similarity scores.
    """
    top_k = min(top_k, 20)
    log.info(f"search_vault: query='{query}' top_k={top_k}")

    try:
        vector = embed(query)
    except Exception as e:
        return f"Error embedding query via Ollama: {e}"

    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
    except Exception as e:
        return f"Error querying Qdrant: {e}"

    if not results:
        return "No results found."

    output = []
    for i, hit in enumerate(results, 1):
        payload = hit.payload or {}
        filepath = payload.get("filepath", "unknown")
        text     = payload.get("text", "")
        score    = round(hit.score, 3)
        chunk_i  = payload.get("chunk_index", 0)
        chunk_t  = payload.get("chunk_total", 1)

        output.append(
            f"## Result {i} (score: {score})\n"
            f"**File:** {filepath}"
            + (f" (chunk {chunk_i+1}/{chunk_t})" if chunk_t > 1 else "")
            + f"\n\n{text}\n"
        )

    return "\n---\n".join(output)


@mcp.tool()
def search_vault_by_tag(tag: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Search for vault notes that mention a specific tag or topic keyword.
    
    Args:
        tag: Tag or keyword to search for (e.g. 'homelab', 'rag', 'job-search')
        top_k: Number of results to return (default 5)
    
    Returns:
        Matching note excerpts containing the tag.
    """
    return search_vault(f"notes tagged with {tag} about {tag}", top_k=top_k)


@mcp.tool()
def find_related_notes(note_content: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Given a piece of text or note content, find semantically related notes in the vault.
    Useful for finding connections and related ideas.
    
    Args:
        note_content: Text content to find related notes for
        top_k: Number of related notes to return (default 5)
    
    Returns:
        Most semantically similar note excerpts from the vault.
    """
    return search_vault(note_content, top_k=top_k)


@mcp.tool()
def vault_stats() -> str:
    """
    Return basic statistics about the indexed Obsidian vault in Qdrant.
    Shows total vectors, collection status, and configuration.
    """
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        return (
            f"Collection: {COLLECTION_NAME}\n"
            f"Total vectors: {info.points_count}\n"
            f"Vector size: {info.config.params.vectors.size}\n"
            f"Distance: {info.config.params.vectors.distance}\n"
            f"Ollama model: {EMBED_MODEL}\n"
            f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}"
        )
    except Exception as e:
        return f"Error fetching collection info: {e}"


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting Obsidian Qdrant MCP server on port {MCP_PORT}")
    log.info(f"Ollama: {OLLAMA_BASE_URL} ({EMBED_MODEL})")
    log.info(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT} → {COLLECTION_NAME}")
    mcp.run(transport="sse", port=MCP_PORT, host="0.0.0.0")
