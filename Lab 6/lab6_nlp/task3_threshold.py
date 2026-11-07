import json

with open("task3_results.json", encoding="utf-8") as f:
    data = json.load(f)

STRATEGY, METHOD = "sentence", "tfidf"
rows = [p for p in data["per_query"] if p["strategy"] == STRATEGY and p["method"] == METHOD]


def top1(p):
    return p["top"][0]["score"]


def accepted(score, t):
    return score >= t


dev = [p for p in rows if p["split"] == "dev"]
test = [p for p in rows if p["split"] == "test"]
print("Configuration:", STRATEGY, METHOD)
print("Development top-1 scores")
for p in dev:
    print("  ", "answerable  " if p["doc"] else "unanswerable", round(top1(p), 3), "|", p["query"])
scores = sorted({round(top1(p), 6) for p in dev})
candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
best_t, best_correct = None, -1
for t in candidates:
    correct = sum((accepted(top1(p), t) == (p["doc"] is not None)) for p in dev)
    if correct > best_correct:
        best_t, best_correct = t, correct
separable = max(top1(p) for p in dev if p["doc"] is None) < min(top1(p) for p in dev if p["doc"] is not None)
print("Scores separable on development data:", separable)
print("Threshold chosen on development data:", round(best_t, 4), "| development decisions correct:", best_correct, "of", len(dev))
print()
print("Held-out test decisions with this fixed threshold")
for p in test:
    ok = (accepted(top1(p), best_t) == (p["doc"] is not None))
    print("  ", "answerable  " if p["doc"] else "unanswerable", round(top1(p), 3),
          "accepted" if accepted(top1(p), best_t) else "rejected", "| correct decision:", ok, "|", p["query"])
print("Test decisions correct:", sum((accepted(top1(p), best_t) == (p["doc"] is not None)) for p in test), "of", len(test))
