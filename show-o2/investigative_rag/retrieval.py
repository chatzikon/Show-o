"""Semantic index and retrieval; imported by the isolated worker only for queries."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_rows(root=ROOT):
    path = Path(root) / "data" / "knowledge.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _corpus_digest(root):
    return hashlib.sha256((Path(root) / "data" / "knowledge.jsonl").read_bytes()).hexdigest()


def build_index(root=ROOT, model_name=DEFAULT_MODEL):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    root = Path(root)
    rows = load_rows(root)
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [r["retrieval_text"] for r in rows], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    folder = root / "index"
    folder.mkdir(exist_ok=True)
    faiss.write_index(index, str(folder / "knowledge.faiss"))
    manifest = {
        "model_name": model_name, "ids": [r["id"] for r in rows],
        "dimension": int(vectors.shape[1]), "corpus_sha256": _corpus_digest(root),
        "similarity": "cosine_using_normalized_inner_product",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class Retriever:
    def __init__(self, root=ROOT):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.root = Path(root)
        manifest = json.loads((self.root / "index" / "manifest.json").read_text())
        if manifest["corpus_sha256"] != _corpus_digest(self.root):
            raise ValueError("Knowledge database changed: rebuild FAISS index")
        self.index = faiss.read_index(str(self.root / "index" / "knowledge.faiss"))
        self.rows = load_rows(self.root)
        self.ids = manifest["ids"]
        if len(self.ids) != self.index.ntotal or set(self.ids) != {r["id"] for r in self.rows}:
            raise ValueError("Index-to-record mismatch")
        if self.index.d != manifest["dimension"]:
            raise ValueError("Index dimension differs from manifest")
        self.by_id = {r["id"]: r for r in self.rows}
        self.model = SentenceTransformer(manifest["model_name"], device="cpu")

    def search(self, observation, top_k=5, minimum_score=None, include_review_needed=False):
        import numpy as np
        if not isinstance(observation, str) or not observation.strip():
            return []
        if not 1 <= top_k <= len(self.ids):
            raise ValueError("top_k out of range")
        vector = self.model.encode(
            [observation], normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(np.ascontiguousarray(vector), len(self.ids))
        hits = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            row = self.by_id[self.ids[index]]
            if row["evidence_status"] == "analyst_extrapolation_review_needed" and not include_review_needed:
                continue
            if minimum_score is not None and float(score) < minimum_score:
                continue
            hits.append({"score": float(score), "entry": row})
            if len(hits) == top_k:
                break
        return hits


def format_context(hits, root=ROOT, max_chars=6500):
    sources = {s["source_id"]: s for s in json.loads((Path(root) / "data" / "sources.json").read_text())}
    blocks = []
    for hit in hits:
        record = hit["entry"]
        refs = "; ".join(f"{sid}: {sources[sid]['url']}" for sid in record["source_ids"]) or "No source (negative control)"
        blocks.append(
            f"ID: {record['id']}\nVisible cues: {', '.join(record['observation_cues'])}\n"
            f"Supported background: {record['source_supported_fact']}\n"
            f"Conditional hypothesis: {record['potential_misuse_hypothesis']}\n"
            f"Alternatives: {', '.join(record['alternative_explanations'])}\n"
            f"Verification required: {record['verification_needed']}\n"
            f"Sources: {refs}\nATT&CK: {', '.join(a['technique_id'] for a in record['attack_mappings']) or 'none'}"
        )
    return "\n\n".join(blocks)[:max_chars]