import re

samples = [
    "AI, search!",
    "I can't log in.",
    "I can\u2019t log in.",
    "help@example.com",
    "CS4944",
    "",
    "\u06cc\u06c1 \u0627\u06cc\u06a9 \u0645\u062b\u0627\u0644 \u06c1\u06d2",
    "POST /v2/token/refresh",
]

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    model_note = None
except Exception as error:
    model = None
    model_note = type(error).__name__ + ": " + str(error)

print("Token comparison on", len(samples), "inputs")
print()
for text in samples:
    whitespace = text.split()
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    print("Original:   ", repr(text))
    print("Whitespace: ", whitespace, len(whitespace))
    print("Word-like:  ", words, len(words))
    if model is not None and text in (samples[0], samples[1]):
        pieces = model.tokenizer.tokenize(text)
        print("Model tokens:", pieces, len(pieces))
    print()

assert re.findall(r"\b\w+\b", "") == []
if model is None:
    print("Model tokenizer not available in this run:", model_note)
    print()

stopwords = {"do", "not", "the", "a", "is", "to", "and"}
sentence = "Do not submit the assignment late."
normalized = [w for w in re.findall(r"\b\w+\b", sentence.lower()) if w not in stopwords]
print("Normalization risk")
print("Original:  ", sentence)
print("Normalized:", " ".join(normalized))
