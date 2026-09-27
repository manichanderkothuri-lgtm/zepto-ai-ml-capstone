# Zepto Data & AI Platform — IIT Patna AI/ML Capstone

This repository contains the three required modules for the Certificate Program in Artificial Intelligence and Machine Learning capstone.

## Structure
- `data_pipeline/` — Books to Scrape scraping, cleaning, GBP→INR conversion, SQLite schema, SQL and pandas validation.
- `analytics/` — Titanic EDA, cleaning, visual data story, classification, imbalance comparison, Random Forest tuning, fare regression, and saved end-to-end classifier pipeline.
- `support_assistant/` — eight Zepto policy documents, local MiniLM embeddings, ChromaDB retrieval, LangGraph routing, Pydantic output, FastAPI and Docker.

## Setup
A single consolidated `requirements.txt` is used.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run Module 1
```bash
cd data_pipeline
python pipeline.py
```
Creates `books_clean.csv`, `books.db`, `sql_outputs.txt`, and `pandas_join_output.txt`.

Required currency baseline: **1 GBP = 105.50 INR**. This is the project-defined constant and is not a live exchange rate.

## Run Module 2
```bash
cd analytics
python analysis.py
```
The script loads `sns.load_dataset('titanic')` exactly once, immediately creates the committed `titanic.csv` fallback, then continues from the same cleaned dataset. Results and plots are generated under `analytics/` and `analytics/plots/`.

## Run Module 3
```bash
cd support_assistant
pip install -r requirements.txt
# MOCK_LLM is intentionally left at its default/offline state
uvicorn main:app --reload --port 7860
```
Then POST JSON `{"query":"What is the delivery fee below INR 149?"}` to `/ask`. A general question such as `{"query":"What is the capital of India?"}` follows the direct-answer branch.

## Module 3 architecture
Ingestion reads the eight exact corpus files. `app.py` embeds each document with `all-MiniLM-L6-v2` and stores the vectors in the ChromaDB `zepto_policies` collection. The LangGraph `classify_intent` node uses the required keyword heuristic in mock mode and conditionally routes to either `retrieve_and_answer` or `direct_answer`. Policy queries retrieve the top three chunks by cosine similarity; mock generation uses the top chunk snippet. The Pydantic `Answer` schema validates `answer`, `sources`, and `confidence`, and FastAPI exposes the graph at `/ask`. The `MOCK_LLM` toggle affects generation/classification behavior; retrieval remains local and real in both modes.

## Git workflow required by the assignment
Create a feature branch, make at least two commits on it, then merge it back into `main`. Example:
```bash
git checkout -b feature/capstone
# make change
git add . && git commit -m "Build capstone modules"
# make another change
git add . && git commit -m "Add validation and documentation"
git checkout main
git merge --no-ff feature/capstone -m "Merge capstone feature"
```

## Academic note
Review and understand generated code, outputs and interpretations before submission and adapt them to your own work. The assignment requires the submitted reasoning and implementation to be authored by the student.
