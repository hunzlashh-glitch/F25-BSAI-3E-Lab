# Lab 6: NLP Fundamentals (tokenization, embeddings and text chunking)

Course: CAI601400 Artificial Intelligence, CLO3 (C3: Apply). Student: Hunzala, BS Artificial Intelligence, Superior University Gold Campus.

All documents, policies and FAQs in this repository are synthetic teaching data written for this lab. They are not real university policies and must not be used for real decisions. No personal data or credentials are included.

## Setup

    python -m venv .venv
    # Windows PowerShell
    .venv\Scripts\Activate.ps1
    # macOS or Linux: source .venv/bin/activate
    python -m pip install numpy scikit-learn sentence-transformers
    python -m pip freeze > requirements.txt

Python 3.10 or newer is recommended (3.12.3 was used). The first embedding run downloads sentence-transformers/all-MiniLM-L6-v2 from Hugging Face (CPU only, no paid API). The requirements.txt in this repository was produced in an environment where the model host was not reachable, so it lists numpy and scikit-learn with exact versions and sentence-transformers without a version. Run the freeze command above after installing to replace it.

## Run order

Create the dataset once (it refuses to overwrite existing files, so delete corpus.json, queries.json and rules.json first if you want to regenerate them):

    python create_dataset.py

Task 1 (beginner):

    python task1_tokens.py
    python guided_chatbot.py
    python task1_chatbot_check.py

Task 2 (intermediate):

    python task2_chunks.py
    python task2_compare.py
    python task2_splitter_failure.py
    python task2_semantic_check.py
    python task2_paraphrase.py
    python guided_embeddings.py
    python guided_suggestions.py

Task 3 (advanced):

    python task3_retrieval.py
    python task3_threshold.py
    python document_chatbot.py
    python document_chatbot.py --backend embedding --strategy sentence

Guided examples and scenarios: guided_tokens.py, guided_chunks.py, guided_metrics.py, guided_limits.py, scenario1_handbook.py to scenario5_equipment.py. Scenario 1 (university handbook search) was chosen as the dataset theme.

To record a scripted chatbot session with the typed text shown, use: python run_transcript.py document_chatbot.py "question|1|exit". Captured outputs of the runs used in the report are in the outputs folder.

## Embedding limitation

The Hugging Face host could not be reached from the environment where this work was produced, so sentence-transformers/all-MiniLM-L6-v2 was never loaded. All reported results are TF-IDF results. task3_retrieval.py detects this, runs the three TF-IDF configurations and says so. When the model is available it runs all six configurations (three chunking strategies with TF-IDF and with embeddings), audits token lengths, measures model load time and rewrites retrieval_log.csv and task3_results.json. guided_embeddings.py, guided_limits.py, the embedding parts of task2_paraphrase.py and task2_semantic_check.py, and the embedding mode of document_chatbot.py have not been executed.

## Code origin

Adapted from the lab manual reference solution: lab6_search.py, task2_chunks.py (the sentence_cuts helper is a small refactor of the same splitter), task3_retrieval.py (the evaluation logic and CSV log; the model fallback, repeated timing, token-audit record and JSON export were added), document_chatbot.py, the guided examples and the five scenario scripts. Written for this submission: the expanded corpus and rules in create_dataset.py, the 12 labeled queries, task1_tokens.py samples and normalization demonstration, the two added FAQs, task2_compare.py, task2_splitter_failure.py, task2_paraphrase.py, task3_threshold.py and run_transcript.py. External references: scikit-learn TfidfVectorizer, Sentence Transformers semantic search documentation (https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) and the all-MiniLM-L6-v2 model card (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Submitting to GitHub

    git init
    git add .
    git commit -m "Lab 6: NLP fundamentals"
    git branch -M main
    git remote add origin https://github.com/<your-username>/lab6_nlp.git
    git push -u origin main
    git rev-parse HEAD

Submit the repository URL and the commit identifier printed by the last command through the course channel, and make sure the instructor has access.
