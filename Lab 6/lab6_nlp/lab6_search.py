import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Retriever:
    def __init__(self, records, backend="tfidf"):
        if not records or any(not r["text"].strip() for r in records):
            raise ValueError("Provide nonempty records")
        self.records, self.backend = records, backend
        texts = [r["text"] for r in records]
        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(texts)
        elif backend == "embedding":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            self.check_length(texts)
            self.matrix = self.model.encode(
                texts, normalize_embeddings=True)
        else:
            raise ValueError("Use tfidf or embedding")

    def check_length(self, texts):
        for text in texts:
            n = len(self.model.tokenizer(text, truncation=False)
                    ["input_ids"])
            if n > self.model.max_seq_length:
                raise ValueError("Input exceeds encoder token limit")

    def search(self, query, k=3, filters=None):
        if not query.strip() or k <= 0:
            return []
        if self.backend == "tfidf":
            q = self.vectorizer.transform([query])
            if q.nnz == 0:
                return []
            scores = (q @ self.matrix.T).toarray()[0]
        else:
            self.check_length([query])
            q = self.model.encode([query], normalize_embeddings=True)[0]
            scores = self.matrix @ q
        allowed = [i for i, r in enumerate(self.records)
                   if all(r.get(key) == value
                          for key, value in (filters or {}).items())]
        ranked = sorted(allowed, key=lambda i: (-float(scores[i]), i))
        return [(self.records[i], float(scores[i])) for i in ranked[:k]]


def show_hits(hits):
    if not hits:
        print("No matching records.")
    for record, score in hits:
        print(record["id"], f"score={score:.3f}")
        print(record["text"])


if __name__ == "__main__":
    demo = [{"id": "a", "text": "Reset a forgotten password using the recovery link."},
            {"id": "b", "text": "Submit the assignment as a PDF through the learning portal."}]
    show_hits(Retriever(demo).search("password recovery", k=1))
