"""
Codebase Indexer — Phase 4 (RAG layer, part 1 of 2)
======================================================
Builds a searchable semantic index of a codebase at FUNCTION granularity:

  1. Walk a repo's source files (Python + C++)
  2. Extract each function/method as a chunk (AST for Python, regex for C++)
  3. Embed each chunk with a local sentence-transformers model
  4. Store in a FAISS index, persisted to disk alongside chunk metadata

Why function-level chunking instead of fixed-size text windows: a function
is a semantically complete unit. A 512-token window can cut a function in
half, embedding a fragment whose meaning differs from the whole. Function
chunks also map 1:1 onto what the reviewer needs to retrieve ("is there an
existing function similar to this new one?").

Why local embeddings (sentence-transformers) instead of an API: free, no
rate limits or quota — and at this scale (a few thousand functions per
repo) a small local model on CPU indexes a repo in a couple of minutes.

Why FAISS instead of a managed vector DB: a few thousand vectors fit in
memory; a managed service adds latency, cost, and an account dependency
for zero benefit at this scale.

Usage:
    python codebase_indexer.py /path/to/repo my_index
    # writes my_index.faiss + my_index.chunks.jsonl

    # or programmatically:
    from codebase_indexer import build_index, load_index, search
    build_index("/path/to/repo", "my_index")
    index, chunks = load_index("my_index")
    results = search(index, chunks, "def parse_config(path): ...", k=5)
"""

import ast
import json
import os
import re
import sys

# Lazy-imported inside functions so this module can be imported (e.g. by the
# reviewer for type hints) without the heavy deps installed:
#   sentence_transformers, faiss, numpy

# Small, fast, general-purpose embedding model. Runs fine on Colab CPU.
# all-MiniLM-L6-v2: 384-dim, ~80MB download, good quality/speed tradeoff
# for code-similarity retrieval at this scale.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PYTHON_EXTENSIONS = (".py",)
CPP_EXTENSIONS = (".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx")

# Directories that are never worth indexing
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", "docs",
}

# Skip enormous files (generated code, vendored bundles) — they pollute
# the index with chunks nobody would ever want retrieved.
MAX_FILE_BYTES = 500_000

# Chunks shorter than this are too trivial to be useful retrieval targets
# (getters, pass-throughs) and just add noise to the index.
MIN_CHUNK_CHARS = 80


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

def extract_python_chunks(path: str, source: str) -> list[dict]:
    """Extract each function/method from a Python file via AST."""
    chunks = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None)
            if end is None:
                continue
            chunk_text = "\n".join(lines[node.lineno - 1:end])
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "path": path,
                "name": node.name,
                "start_line": node.lineno,
                "end_line": end,
                "language": "python",
                "text": chunk_text,
            })
    return chunks


# C++ function extraction via regex. Deliberately simple: matches a
# return-type + name + params + opening brace, then brace-counts to find
# the end. Won't handle every template edge case — consistent with the
# project's Python-first, C++-approximate approach (same asymmetry as the
# static analyzer). Misses are acceptable: an incomplete index still
# retrieves useful context; a wrong parse just means one function missing.
CPP_FUNC_START_RE = re.compile(
    r"^[\w:\<\>\*&\s~]+?[\w~]+\s*\([^;{]*\)\s*(const)?\s*(noexcept)?\s*\{",
    re.MULTILINE,
)


def extract_cpp_chunks(path: str, source: str) -> list[dict]:
    """Extract functions from a C++ file via regex + brace counting."""
    chunks = []
    for match in CPP_FUNC_START_RE.finditer(source):
        start_pos = match.start()
        # Brace-count from the opening brace to find the function end
        brace_pos = source.index("{", match.start())
        depth = 0
        end_pos = None
        for i in range(brace_pos, min(len(source), brace_pos + 20_000)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break
        if end_pos is None:
            continue

        chunk_text = source[start_pos:end_pos]
        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue

        start_line = source[:start_pos].count("\n") + 1
        end_line = source[:end_pos].count("\n") + 1

        # Best-effort function name: last identifier before the paren
        header = source[start_pos:brace_pos]
        name_match = re.search(r"([\w~]+)\s*\(", header)
        name = name_match.group(1) if name_match else "unknown"

        chunks.append({
            "path": path,
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "language": "cpp",
            "text": chunk_text,
        })
    return chunks


def walk_repo(repo_path: str) -> list[dict]:
    """Walk a repo and extract all function chunks."""
    all_chunks = []
    for root, dirs, files in os.walk(repo_path):
        # Prune skipped directories in-place so os.walk doesn't descend
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_path)

            is_py = fname.endswith(PYTHON_EXTENSIONS)
            is_cpp = fname.endswith(CPP_EXTENSIONS)
            if not (is_py or is_cpp):
                continue

            try:
                if os.path.getsize(fpath) > MAX_FILE_BYTES:
                    continue
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except OSError:
                continue

            if is_py:
                all_chunks.extend(extract_python_chunks(rel_path, source))
            else:
                all_chunks.extend(extract_cpp_chunks(rel_path, source))

    return all_chunks


# ---------------------------------------------------------------------------
# Embedding + FAISS index
# ---------------------------------------------------------------------------

def build_index(repo_path: str, index_name: str, verbose: bool = True) -> int:
    """
    Index a repo: extract chunks, embed, build FAISS index, persist to disk.
    Writes {index_name}.faiss and {index_name}.chunks.jsonl.
    Returns the number of chunks indexed.
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    if verbose:
        print(f"Extracting function chunks from {repo_path}...")
    chunks = walk_repo(repo_path)
    if verbose:
        py = sum(1 for c in chunks if c["language"] == "python")
        cpp = len(chunks) - py
        print(f"  {len(chunks)} chunks ({py} Python, {cpp} C++)")

    if not chunks:
        raise RuntimeError(f"No indexable functions found in {repo_path}")

    if verbose:
        print(f"Embedding with {EMBEDDING_MODEL} (local, CPU)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=verbose, batch_size=64, convert_to_numpy=True
    )
    embeddings = embeddings.astype(np.float32)

    # Normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # FAISS won't create parent directories — do it ourselves
    parent = os.path.dirname(index_name)
    if parent:
        os.makedirs(parent, exist_ok=True)

    faiss.write_index(index, f"{index_name}.faiss")
    with open(f"{index_name}.chunks.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    if verbose:
        print(f"Index written: {index_name}.faiss + {index_name}.chunks.jsonl")
    return len(chunks)


def load_index(index_name: str):
    """Load a persisted index. Returns (faiss_index, chunks_list)."""
    import faiss

    index = faiss.read_index(f"{index_name}.faiss")
    chunks = []
    with open(f"{index_name}.chunks.jsonl") as f:
        for line in f:
            chunks.append(json.loads(line))
    return index, chunks


# Module-level model cache so repeated search() calls don't reload the model
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def search(index, chunks: list, query_text: str, k: int = 5,
           exclude_path: str = None) -> list[dict]:
    """
    Find the k chunks most similar to query_text.

    exclude_path: skip chunks from this file — when reviewing a diff to
    file X, retrieving X's own (pre-change) functions is usually just
    noise; we want context from *elsewhere* in the codebase.

    Returns chunks with an added "similarity" score (cosine, 0-1).
    """
    import numpy as np
    import faiss

    model = _get_model()
    query_emb = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_emb)

    # Over-fetch so we can filter out excluded paths and still return k
    fetch_k = min(k * 3, len(chunks))
    scores, indices = index.search(query_emb, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if exclude_path and chunk["path"] == exclude_path:
            continue
        results.append({**chunk, "similarity": round(float(score), 3)})
        if len(results) >= k:
            break
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python codebase_indexer.py <repo_path> <index_name>")
        sys.exit(1)
    build_index(sys.argv[1], sys.argv[2])
