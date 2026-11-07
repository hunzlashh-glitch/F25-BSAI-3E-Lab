import json
from task2_chunks import make_chunks

with open("corpus.json", encoding="utf-8") as f:
    docs = json.load(f)
with open("rules.json", encoding="utf-8") as f:
    rules = json.load(f)

original_words = sum(len(t.split()) for t in docs.values())
print("Documents:", len(docs), "| original word units:", original_words)
for d, t in docs.items():
    print(" ", d, len(t.split()), "words,", len(t), "characters")
print()

summary = {}
for strategy in ["0", "10", "sentence"]:
    chunks = make_chunks(docs, strategy)
    sizes = [len(c["text"].split()) for c in chunks]
    emitted = sum(sizes)
    survive = {}
    for d, pair in rules.items():
        survive[d] = any(c["doc"] == d and pair["rule"] in c["text"] and pair["exception"] in c["text"]
                         for c in chunks)
    summary[strategy] = {
        "chunks": len(chunks), "mean": round(emitted / len(chunks), 1), "min": min(sizes), "max": max(sizes),
        "emitted": emitted, "duplicated": emitted - original_words,
        "survive": survive, "survive_count": sum(survive.values()),
    }
    label = {"0": "window 40, overlap 0", "10": "window 40, overlap 10", "sentence": "sentence-aware, budget 40"}[strategy]
    print(label)
    print("  chunks:", len(chunks), "| words per chunk mean/min/max:", round(emitted / len(chunks), 1), min(sizes), max(sizes))
    print("  emitted words:", emitted, "| duplicated words (emitted - original):", emitted - original_words)
    print("  rule and exception in one chunk:", {d: ("yes" if v else "no") for d, v in survive.items()},
          "->", sum(survive.values()), "of", len(rules))
    for c in chunks:
        assert c["text"] == docs[c["doc"]][c["start"]:c["end"]] and c["text"].strip()
print()
print("Source offsets verified: every chunk text equals source[start:end].")

with open("chunk_comparison.json", "w", encoding="utf-8") as f:
    json.dump({"original_words": original_words, "summary": summary,
               "doc_words": {d: len(t.split()) for d, t in docs.items()}}, f, indent=2)
