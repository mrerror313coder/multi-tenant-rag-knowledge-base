"""Embedding service with SentenceTransformers, Cohere, Hugging Face, and fast deterministic fallback."""

from typing import List, Optional
import os
import hashlib
import math
import logging
import httpx

logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    _ST_AVAILABLE = False


class EmbeddingService:
    """Provides vector embeddings using SentenceTransformers, Cohere, or Hugging Face."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        provider: str = "sentence_transformers",
        cohere_api_key: Optional[str] = None,
        cohere_model: str = "embed-multilingual-v3.0",
        huggingface_api_key: Optional[str] = None,
        huggingface_model: str = "BAAI/bge-m3",
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.provider = provider.lower()
        self._cohere_api_key = cohere_api_key
        self.cohere_model = cohere_model
        self._huggingface_api_key = huggingface_api_key
        self.huggingface_model = huggingface_model
        self._model = None
        self._use_fallback = not _ST_AVAILABLE

    @property
    def cohere_api_key(self) -> str:
        return (self._cohere_api_key or os.getenv("COHERE_API_KEY", "")).strip().strip('"').strip("'")

    @property
    def huggingface_api_key(self) -> str:
        return (self._huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")).strip().strip('"').strip("'")

    def _load_model(self):
        """Lazy loader for local SentenceTransformer model."""
        if self._model is None and _ST_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Using deterministic fallback vectorizer.")
                self._use_fallback = True

    def _embed_cohere(self, texts: List[str], input_type: str = "search_document") -> Optional[List[List[float]]]:
        """Generates multilingual embeddings via Cohere Embed v3 API."""
        if not self.cohere_api_key:
            return None
        try:
            url = "https://api.cohere.com/v2/embed"
            headers = {
                "Authorization": f"Bearer {self.cohere_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.cohere_model,
                "texts": texts,
                "input_type": input_type,
                "embedding_types": ["float"],
            }
            with httpx.Client(timeout=20.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    return data["embeddings"]["float"]
                logger.warning(f"Cohere API error: {res.status_code} {res.text}")
        except Exception as e:
            logger.warning(f"Cohere embedding exception: {e}")
        return None

    def _embed_huggingface(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generates embeddings via Hugging Face Serverless Inference API."""
        if not self.huggingface_api_key:
            return None
        try:
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.huggingface_model}"
            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"inputs": texts, "options": {"wait_for_model": True}}
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    embs = res.json()
                    # Handle single vs batch response shapes
                    if embs and isinstance(embs[0], list):
                        return embs
                    elif embs and isinstance(embs[0], (int, float)):
                        return [embs]
                logger.warning(f"Hugging Face API error: {res.status_code} {res.text}")
        except Exception as e:
            logger.warning(f"Hugging Face embedding exception: {e}")
        return None

    def _deterministic_hash_embedding(self, text: str) -> List[float]:
        """Fast, deterministic zero-dependency embedding based on subword hashing."""
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * (1.0 + (len(word) / 10.0))

            if len(word) >= 3:
                for i in range(len(word) - 2):
                    ng = word[i : i + 3]
                    h_ng = int(hashlib.sha256(ng.encode("utf-8")).hexdigest(), 16)
                    ng_idx = h_ng % self.dimension
                    vec[ng_idx] += 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for document chunks using configured provider."""
        if not texts:
            return []

        # 1. Cohere Provider
        if self.provider == "cohere" or self.cohere_api_key:
            cohere_embs = self._embed_cohere(texts, input_type="search_document")
            if cohere_embs:
                return cohere_embs

        # 2. Hugging Face Provider
        if self.provider == "huggingface" or self.huggingface_api_key:
            hf_embs = self._embed_huggingface(texts)
            if hf_embs:
                return hf_embs

        # 3. SentenceTransformers Local Provider
        if not self._use_fallback:
            try:
                self._load_model()
                if self._model:
                    embeddings = self._model.encode(texts, normalize_embeddings=True)
                    return embeddings.tolist()
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed: {e}. Falling back.")

        # 4. Deterministic Hash Fallback
        return [self._deterministic_hash_embedding(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Generate normalized vector embedding for a single user query."""
        if self.provider == "cohere" or self.cohere_api_key:
            cohere_embs = self._embed_cohere([query], input_type="search_query")
            if cohere_embs:
                return cohere_embs[0]

        results = self.embed_documents([query])
        return results[0] if results else [0.0] * self.dimension


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two vector embeddings."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    sim = dot / (norm1 * norm2)
    return max(0.0, min(1.0, float(sim)))


def cluster_chunks_by_embedding(
    chunks: List[Dict[str, Any]],
    query_vector: Optional[List[float]] = None,
    similarity_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """Groups retrieved document chunks into semantic topic clusters based on embedding similarity.
    
    Assigns:
      - cluster_id (int): 1-indexed cluster group
      - cluster_label (str): Topic description extracted from common key terms
      - similarity_score (float): Verified similarity against query embedding
    """
    if not chunks:
        return []

    service = get_embedding_service()
    
    # 1. Ensure all chunks have embeddings
    texts = [c.get("text", "") for c in chunks]
    embeddings = service.embed_documents(texts)
    for c, emb in zip(chunks, embeddings):
        c["_embedding"] = emb
        if query_vector:
            c["similarity_score"] = round(compute_cosine_similarity(emb, query_vector), 4)

    # 2. Semantic Clustering (Threshold-based Agglomerative Grouping)
    clusters = []  # List of lists of chunk indices
    for i, c in enumerate(chunks):
        c_emb = c["_embedding"]
        assigned = False
        for cluster in clusters:
            # Check similarity with centroid or representative member
            rep_idx = cluster[0]
            rep_emb = chunks[rep_idx]["_embedding"]
            sim = compute_cosine_similarity(c_emb, rep_emb)
            if sim >= similarity_threshold:
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    # 3. Label Clusters & Enrich Chunks
    enriched_chunks = []
    for cluster_id, chunk_indices in enumerate(clusters, start=1):
        # Extract prominent keywords from cluster text for descriptive label
        cluster_texts = " ".join([chunks[idx].get("text", "") for idx in chunk_indices])
        words = [
            w for w in cluster_texts.replace("\n", " ").split()
            if len(w) >= 4 and w.isalnum() and w.lower() not in {"this", "that", "with", "from", "your", "have", "they"}
        ]
        top_words = list(dict.fromkeys(words))[:3]
        topic_title = " / ".join(top_words).title() if top_words else f"Topic {cluster_id}"
        cluster_label = f"Cluster {cluster_id}: {topic_title}"

        for idx in chunk_indices:
            chunk = chunks[idx].copy()
            chunk.pop("_embedding", None)
            chunk["cluster_id"] = cluster_id
            chunk["cluster_label"] = cluster_label
            enriched_chunks.append(chunk)

    # 4. Sort by query similarity score descending
    enriched_chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    return enriched_chunks


# Global singleton instance
_embedding_service = None


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service

