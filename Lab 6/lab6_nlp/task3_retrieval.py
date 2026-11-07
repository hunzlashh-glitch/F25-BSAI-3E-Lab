import csv
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from task2_chunks import make_chunks

REPEATS = 20


def load_model():
    try:
        start = time.perf_counter()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return model, time.perf_counter() - start, None
    except Exception as error:
        return None, None, type(error).__name__ + ": " + str(error)


def ranked_top(row, k=3):
    return np.argsort(-row, kind="stable")[:k]


def evaluate(scores, chunks, items):
    hits, reciprocal = [], []
    for row, q in zip(scores, items):
        if q["doc"] is None:
            continue
        top = ranked_top(row)
        good = [rank for rank, i in enumerate(top, 1)
                if chunks[i]["doc"] == q["doc"]
                and q["evidence"] in chunks[i]["text"]]
        hits.append(bool(good))
        reciprocal.append(1 / good[0] if good else 0)
    if not hits:
        return None, None
    return float(np.mean(hits)), float(np.mean(reciprocal))


def mean_time(function, repeats):
    total = 0.0
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        total += time.perf_counter() - start
    return result, total / repeats


def main():
    with open("corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    with open("queries.json", encoding="utf-8") as f:
        items = json.load(f)
    assert docs and items
    for q in items:
        assert q["split"] in {"dev", "test"}
        if q["doc"] is not None:
            assert q["evidence"] and q["evidence"] in docs[q["doc"]]

    model, load_seconds, model_error = load_model()
    if model is None:
        print("Embedding model unavailable:", model_error)
        print("Only the TF-IDF configurations are run; embedding rows are not reported.")
        methods = ["tfidf"]
    else:
        print("Model load seconds", round(load_seconds, 3))
        methods = ["tfidf", "embedding"]

    query_texts = [q["query"] for q in items]
    logs, metrics, per_query, audit = [], [], [], []
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        texts = [c["text"] for c in chunks]
        word_counts = [len(t.split()) for t in texts + query_texts]
        if model is not None:
            lengths = [len(model.tokenizer(t, truncation=False)["input_ids"])
                       for t in texts + query_texts]
            if max(lengths) > model.max_seq_length:
                raise ValueError("Model limit exceeded; rechunk or shorten query")
            audit.append(dict(strategy=strategy, max_words=max(word_counts), max_tokens=max(lengths),
                              limit=model.max_seq_length, checked="model tokenizer"))
        else:
            audit.append(dict(strategy=strategy, max_words=max(word_counts), max_tokens=None,
                              limit=None, checked="not performed (model tokenizer unavailable)"))
        for method in methods:
            if method == "tfidf":
                def encode():
                    vectorizer = TfidfVectorizer()
                    C = vectorizer.fit_transform(texts)
                    Q = vectorizer.transform(query_texts)
                    return C, Q
                (C, Q), encode_seconds = mean_time(encode, REPEATS)
                scores, search_seconds = mean_time(lambda: (Q @ C.T).toarray(), REPEATS)
            else:
                def encode():
                    return (model.encode(texts, normalize_embeddings=True),
                            model.encode(query_texts, normalize_embeddings=True))
                (C, Q), encode_seconds = mean_time(encode, 3)
                scores, search_seconds = mean_time(lambda: Q @ C.T, REPEATS)
            for split in ["dev", "test"]:
                indices = [i for i, q in enumerate(items) if q["split"] == split]
                hit, mrr = evaluate(scores[indices], chunks, [items[i] for i in indices])
                metrics.append(dict(strategy=strategy, method=method, split=split, hit=hit, mrr=mrr,
                                    chunks=len(chunks), encode_seconds=encode_seconds,
                                    search_seconds=search_seconds))
                print(strategy, method, split, "Hit@3/MRR@3", None if hit is None else round(hit, 3),
                      None if mrr is None else round(mrr, 3), "chunks", len(chunks),
                      "encode/search seconds", round(encode_seconds, 5), round(search_seconds, 6))
            for qi, (row, q) in enumerate(zip(scores, items)):
                top = []
                for rank, i in enumerate(ranked_top(row), 1):
                    c = chunks[i]
                    relevant = bool(q["doc"] is not None and c["doc"] == q["doc"]
                                    and q["evidence"] in c["text"])
                    top.append(dict(rank=rank, chunk_id=c["id"], doc=c["doc"], start=c["start"],
                                    end=c["end"], score=float(row[i]), relevant=relevant, text=c["text"]))
                    logs.append(dict(strategy=strategy, method=method, query=q["query"], split=q["split"],
                                     rank=rank, score=float(row[i]), chunk_id=c["id"], doc=c["doc"],
                                     start=c["start"], end=c["end"], text=c["text"],
                                     answerable=q["doc"] is not None, relevant=relevant))
                per_query.append(dict(strategy=strategy, method=method, query=q["query"], split=q["split"],
                                      doc=q["doc"], top=top))

    check = next(p for p in per_query if p["strategy"] == "sentence" and p["method"] == "tfidf"
                 and p["query"] == "How do I reset a forgotten password?")
    flags = [t["relevant"] for t in check["top"]]
    print()
    print("Manual metric check (sentence-aware, tfidf):", check["query"])
    print("  top-3 ids:", [t["chunk_id"] for t in check["top"]], "relevant flags:", flags)
    first = flags.index(True) + 1 if True in flags else None
    print("  first relevant rank:", first, "-> reciprocal rank:", (1 / first) if first else 0)

    with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    with open("task3_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(model_available=model is not None, model_error=model_error, load_seconds=load_seconds,
                       metrics=metrics, per_query=per_query, audit=audit, repeats=REPEATS), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
