from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
for label, text in [("short", "Reset my password."),
                    ("long", "password " * 300)]:
    ids = model.tokenizer(text, truncation=False)["input_ids"]
    fits = len(ids) <= model.max_seq_length
    print(label, "tokens:", len(ids), "limit:", model.max_seq_length,
          "fits:", fits)
