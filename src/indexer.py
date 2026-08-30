import json
import re
import hashlib
from pathlib import Path
from collections import Counter
import math

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "extracted_chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings.npy"
META_PATH = BASE_DIR / "data" / "index_meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"

# Equipment identifiers that deserve heavy keyword weighting
EQUIPMENT_IDS = [
    "8FAN", "FAN16", "16FAN2", "16FAN2C", "FAN32H", "FAN",
    "8DC30", "8DC30T", "8DC30T2", "8AC7",
    "16DC30", "16DC30T", "16AC16",
    "32DC40", "32AC15",
    "8EC2", "32EC2", "EC",
    "PSILFAN", "PSILPFDC", "PSILPFAC",
    "PSS-8", "PSS-16", "PSS-16II", "PSS-32",
    "PSI-4L", "PSI-8L",
]


def _chunks_hash(chunks: list) -> str:
    """Compute a hash of the chunk texts to detect changes."""
    h = hashlib.md5()
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# IDF computation for improved keyword scoring
# ---------------------------------------------------------------------------
def _build_idf(chunks: list) -> dict:
    """Build an IDF dictionary from chunk texts.
       It also Give important words higher weight."""
    N = len(chunks)
    doc_freq = Counter()
    for chunk in chunks:
        tokens = set(re.findall(r"\b[A-Za-z0-9][\w\-]*\b", chunk["text"].lower()))
        for token in tokens:
            doc_freq[token] += 1
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((N + 1) / (df + 1)) + 1  # smoothed IDF
    return idf

# This is the retrieval enginge for the RAG system
class VectorIndexer:
    """Build, persist, and search a vector index over chunked documents."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[Indexer] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.idf = {}

    # ------------------------------------------------------------------
    # Augmented text generation
    # ------------------------------------------------------------------
    @staticmethod
    def _augmented_text(chunk: dict) -> str:
        """
        Build an embedding-ready string by prepending section hierarchy
        and page metadata to the chunk text. This helps the embedding
        model associate the text with its context.
        """
        parts = []
        if chunk.get("chapter"):
            parts.append(f"Chapter: {chunk['chapter']}")
        section_path = chunk.get("section", "")
        parent = chunk.get("parent_section", "")
        if parent and parent != chunk.get("chapter"):
            section_path = f"{parent} > {section_path}"
        if section_path:
            parts.append(f"Section: {section_path}")

        page_start = chunk.get("page_start", chunk.get("page", "?"))
        page_end = chunk.get("page_end", page_start)
        if page_start == page_end:
            parts.append(f"Page: {page_start}")
        else:
            parts.append(f"Pages: {page_start}-{page_end}")

        parts.append("")  # blank line separator
        parts.append(chunk["text"])
        return "\n".join(parts)

    def build_and_save_index(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
        meta_path: Path = META_PATH,
    ):
        """Embed all chunks and save the index to disk."""
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at '{chunks_path}'.")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        chunk_hash = _chunks_hash(self.chunks)

        # Check if we can skip re-embedding
        if meta_path.exists() and embeddings_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("hash") == chunk_hash and meta.get("model") == MODEL_NAME:
                print("[Indexer] Embeddings are up-to-date, skipping re-build.")
                self.load_index(chunks_path, embeddings_path)
                return

        print(f"[Indexer] Encoding {len(self.chunks)} chunks "
              f"(with section hierarchy context)...")

        augmented_texts = [self._augmented_text(c) for c in self.chunks]

        raw_embeddings = self.model.encode(
            augmented_texts, show_progress_bar=True, convert_to_numpy=True
        )
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        self.embeddings = raw_embeddings / np.maximum(norms, 1e-12)

        # Save embeddings
        np.save(embeddings_path, self.embeddings)
        print(f"[Indexer] Persisted embeddings to '{embeddings_path}'.")

        # Save metadata
        with open(meta_path, "w") as f:
            json.dump({
                "hash": chunk_hash,
                "model": MODEL_NAME,
                "num_chunks": len(self.chunks),
                "embedding_dim": self.embeddings.shape[1],
            }, f, indent=2)

        # Build FAISS index if available
        self._build_faiss_index()

        # Build IDF
        self.idf = _build_idf(self.chunks)

    # ------------------------------------------------------------------
    # Load from disk
    # ------------------------------------------------------------------
    def load_index(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
    ):
        """Load pre-built chunks and embeddings from disk."""
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embeddings = np.load(embeddings_path)
        self._build_faiss_index()
        self.idf = _build_idf(self.chunks)

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------
    def _build_faiss_index(self):
        """Build a FAISS inner-product index for comparison."""
        if not FAISS_AVAILABLE or self.embeddings is None:
            return
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        print(f"[Indexer] Built FAISS IndexFlatIP with {self.faiss_index.ntotal} vectors.")

    # ------------------------------------------------------------------
    # Keyword scoring (TF-IDF-like with equipment identifier boosting)
    # ------------------------------------------------------------------
    def _keyword_score(self, query: str, chunk: dict) -> float:
        """
        Compute a TF-IDF-inspired keyword score between the query and a chunk.
        Equipment identifiers get a heavy boost.
        """
        query_tokens = re.findall(r"\b[A-Za-z0-9][\w\-]*\b", query.lower())
        if not query_tokens:
            return 0.0

        ''' It doesn't only search the chunk's actual text.
            It also includes: -section  -parent section -text
        '''
        chunk_text = f"{chunk['section']} {chunk.get('parent_section', '')} {chunk['text']}".lower()
        chunk_tokens = re.findall(r"\b[A-Za-z0-9][\w\-]*\b", chunk_text)
        chunk_counter = Counter(chunk_tokens)
        chunk_len = max(len(chunk_tokens), 1)

        score = 0.0
        for token in query_tokens:
            tf = chunk_counter.get(token, 0) / chunk_len # tf -> Term Frequency
            idf = self.idf.get(token, 1.0)
            score += tf * idf

        # Equipment identifier boosting
        query_upper = query.upper()
        chunk_upper = chunk_text.upper()
        for equip_id in EQUIPMENT_IDS:
            if equip_id in query_upper and equip_id in chunk_upper:
                score += 2.0  # Heavy boost for exact equipment match
            elif equip_id in query_upper and equip_id not in chunk_upper:
                score -= 0.5  # Mild penalty if query mentions equipment not in chunk

        return score

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.3,
        shelf_filter: str = None,
        section_filter: str = None,
        use_faiss: bool = False,
    ) -> list:
        """
        Hybrid search combining dense cosine similarity with keyword scoring.
        alpha : float
            Weight for keyword score (0 = pure dense, 1 = pure keyword).
            Dense weight = (1 - alpha).
        shelf_filter : str, optional
            If set, only search chunks tagged with this shelf (e.g. "PSS-8").
        section_filter : str, optional
            If set, only search chunks whose section contains this substring.
        use_faiss : bool
            If True and FAISS is available, use FAISS for the dense retrieval step.

        Returns
        -------
        list of dict with keys: score, dense_score, keyword_score, chunk
        """
        if self.embeddings is None or not self.chunks:
            self.load_index()

        # ---- Determine candidate indices ----
        if shelf_filter or section_filter:
            candidate_indices = []
            for idx, chunk in enumerate(self.chunks):
                if shelf_filter:
                    if shelf_filter.upper() not in [t.upper() for t in chunk.get("shelf_tags", [])]:
                        continue
                if section_filter:
                    searchable = f"{chunk['section']} {chunk.get('parent_section', '')} {chunk.get('chapter', '')}"
                    if section_filter.lower() not in searchable.lower():
                        continue
                candidate_indices.append(idx)

            if not candidate_indices:
                return []
        else:
            candidate_indices = list(range(len(self.chunks)))

        # ---- Dense similarity ----
        query_vec = self.model.encode([query], convert_to_numpy=True)[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        if use_faiss and self.faiss_index is not None and not (shelf_filter or section_filter):
            # FAISS search (full index, no filtering)
            scores_faiss, indices_faiss = self.faiss_index.search(
                query_vec.reshape(1, -1).astype(np.float32), min(top_k * 3, len(self.chunks))
            )
            dense_scores = np.zeros(len(self.chunks))
            for s, i in zip(scores_faiss[0], indices_faiss[0]):
                if i >= 0:
                    dense_scores[i] = s
        else:
            dense_scores = np.dot(self.embeddings, query_vec)

        # ---- Keyword scoring ----
        keyword_scores = np.zeros(len(self.chunks))
        for idx in candidate_indices:
            keyword_scores[idx] = self._keyword_score(query, self.chunks[idx])

        # ---- Normalize scores ----
        cand_dense = dense_scores[candidate_indices]
        cand_kw = keyword_scores[candidate_indices]

        # Min-max normalize within candidates
        d_min, d_max = cand_dense.min(), cand_dense.max()
        if d_max > d_min:
            norm_dense = (cand_dense - d_min) / (d_max - d_min)
        else:
            norm_dense = np.zeros_like(cand_dense)

        kw_max = cand_kw.max() if cand_kw.max() > 0 else 1.0
        norm_kw = cand_kw / kw_max

        # Combined score
        combined = (1 - alpha) * norm_dense + alpha * norm_kw

        # ---- Top-K selection ----
        top_local = np.argsort(combined)[::-1][:top_k]

        results = []
        for local_idx in top_local:
            global_idx = candidate_indices[local_idx]
            results.append({
                "score": float(combined[local_idx]),
                "dense_score": float(dense_scores[global_idx]),
                "keyword_score": float(keyword_scores[global_idx]),
                "chunk": self.chunks[global_idx],
            })

        return results

    def search_dense_only(self, query: str, top_k: int = 5) -> list:
        """Pure dense cosine similarity search (no keyword component)."""
        return self.search(query, top_k=top_k, alpha=0.0)

    def search_faiss(self, query: str, top_k: int = 5) -> list:
        """FAISS-based dense search for comparison."""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            print("[Indexer] FAISS not available, falling back to numpy search.")
            return self.search_dense_only(query, top_k=top_k)
        return self.search(query, top_k=top_k, alpha=0.0, use_faiss=True)


# Alias for compatibility
HybridIndexer = VectorIndexer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    indexer = VectorIndexer()
    indexer.build_and_save_index()

    # Test all 8 evaluation questions
    questions = [
        "How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?",
        "What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide?",
        "What are the two software load-lines supported by the 1830 PSS system?",
        "Which fan units are supported on the 1830 PSS-32 shelf?",
        "Which fan unit(s) are used on the 1830 PSS-16II shelf?",
        "Name the power filter cards supported on the 1830 PSS-8 shelf.",
        "What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, and which common aperture size is explicitly NOT supported?",
        "What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf without amplification?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Q{i}: {q}")
        print(f"{'='*80}")
        results = indexer.search(q, top_k=3)
        for j, r in enumerate(results, 1):
            c = r["chunk"]
            print(f"  Top {j} (Combined: {r['score']:.4f} | Dense: {r['dense_score']:.4f} | "
                  f"KW: {r['keyword_score']:.2f})")
            print(f"    Page: {c['page_start']}-{c['page_end']} | "
                  f"Section: {c['section'][:60]}")
            print(f"    Shelf tags: {c.get('shelf_tags', [])}")
            print(f"    Text: {c['text'][:180]}...")