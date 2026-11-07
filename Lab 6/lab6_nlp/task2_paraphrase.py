import json
from task2_chunks import make_chunks
from lab6_search import Retriever

with open("corpus.json", encoding="utf-8") as f:
    docs = json.load(f)
records = make_chunks(docs, "sentence")
queries = [
    ("How can I regain access to my account?", "password"),
    ("I was sick so my coursework will be late, what do I need?", "late"),
    ("Is it okay for a freelancer to work from home?", "contractor"),
]
for backend in ["tfidf", "embedding"]:
    print("Backend:", backend)
    try:
        engine = Retriever(records, backend)
    except Exception as error:
        print("Backend unavailable:", type(error).__name__ + ":", error)
        print()
        continue
    for query, target in queries:
        print("Query:", query, "| intended document:", target)
        hits = engine.search(query, k=3)
        if not hits:
            print("  No matching records.")
        for rank, (r, score) in enumerate(hits, 1):
            print(f"  {rank}. {r['id']} score={score:.3f} chars {r['start']}-{r['end']}")
            print("     " + r["text"][:110].replace("\n", " "))
        print()
