import json
from task2_chunks import sentence_cuts, sentence_chunks

with open("corpus.json", encoding="utf-8") as f:
    docs = json.load(f)

text = docs["assignment"]
cuts = sentence_cuts(text)
print("Segments found by the sentence splitter in the assignment document:")
for a, b in zip(cuts, cuts[1:]):
    print(" ", a, b, repr(text[a:b][:90]))
print()
sample = "Ask Dr. Khan for approval. Then submit the form."
cuts = sentence_cuts(sample)
print("Abbreviation test:", [sample[a:b] for a, b in zip(cuts, cuts[1:])])
print("Expected two sentences, splitter returned", len(cuts) - 1, "segments")
