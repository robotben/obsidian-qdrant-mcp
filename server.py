#!/usr/bin/env python3
"""
Qdrant Semantic Search MCP Server
Embeds queries via Ollama (nomic-embed-text) and searches the Obsidian Qdrant collection.
Exposes an HTTP MCP endpoint compatible with claude.ai.

search_vault enforces a relevance floor (RELEVANCE_FLOOR, default 0.45 -- the value
reconciled in wiki-pipeline Phase 0.5, matching n8n's Util - Obsidian RAG Search and
Scripts/vault_search_mcp.py; see System/wiki-schema.md). search_vault_by_tag is an
exact frontmatter-tag payload filter (not semantic similarity on the tag string),
paginated and deduped to distinct notes.
"""

import json
import logging
import os
from typing import Any

import requests
from fastmcp import FastMCP
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://192.168.245.62:11434")
EMBED_MODEL      = os.getenv("EMBED_MODEL",       "nomic-embed-text")
QDRANT_HOST      = os.getenv("QDRANT_HOST",       "192.168.245.187")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT",   "6333"))
COLLECTION_NAME  = os.getenv("COLLECTION_NAME",   "obsidian")
DEFAULT_TOP_K    = int(os.getenv("DEFAULT_TOP_K", "5"))
MCP_PORT         = int(os.getenv("MCP_PORT",      "3000"))
# Shared relevance floor -- documented in System/wiki-schema.md. Matches the value
# already enforced by n8n's Util - Obsidian RAG Search and Scripts/vault_search_mcp.py.
RELEVANCE_FLOOR  = float(os.getenv("RELEVANCE_FLOOR", "0.45"))

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
        Matching note excerpts with file paths and similarity scores
        (results below the relevance floor are filtered out).
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
            score_threshold=RELEVANCE_FLOOR,
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
def search_vault_by_tag(tag: str, top_k: int = 10) -> str:
    """
    Find notes whose frontmatter `tags` field includes the given tag (exact match).

    Args:
        tag: Tag to match against note frontmatter (e.g. 'homelab', 'rag', 'python')
        top_k: Number of distinct notes to return (default 10)

    Returns:
        List of notes carrying the exact tag, with vault-relative paths.
    """
    try:
        # Exact payload filter on the frontmatter tags embedded at ingestion time --
        # NOT semantic similarity on the tag string. Scroll results are per-chunk
        # and must be deduped to distinct notes; paginate until top_k distinct
        # files or scroll exhaustion, since chunk-heavy notes can fill any page.
        seen_files = {}
        offset = None
        while len(seen_files) < top_k:
            hits, offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="tags", match=models.MatchValue(value=tag))]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for hit in hits:
                payload = hit.payload or {}
                filepath = payload.get("filepath", "unknown")
                if filepath not in seen_files:
                    seen_files[filepath] = payload.get("filename", "unknown")
                    if len(seen_files) >= top_k:
                        break
            if not hits or offset is None:
                break
    except Exception as e:
        return f"Error: {e}"

    if not seen_files:
        return "No notes found."

    lines = [f"Notes tagged '{tag}':"]
    for filepath, filename in seen_files.items():
        lines.append(f"  - {filename}\n    Path: {filepath}")
    return "\n".join(lines)


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
    mcp.run(transport="streamable-http", port=MCP_PORT, host="0.0.0.0")
