# Module 3 — Zepto Support Assistant

## Run
```bash
pip install -r requirements.txt
set MOCK_LLM=1
uvicorn main:app --reload --port 7860
```
The graded baseline is offline mock mode. `MOCK_LLM` is unset or `1` by default; no LLM provider or API key is required.

## RAG architecture
`docs/*.txt` → `app.resources()` loads and chunks one document per file → `SentenceTransformer(all-MiniLM-L6-v2)` creates local embeddings → ChromaDB collection `zepto_policies` stores vectors → LangGraph `classify_intent` routes policy questions → `retrieve_and_answer` performs top-3 cosine retrieval and mock generation → Pydantic `Answer` validates `answer/sources/confidence` → FastAPI `/ask` exposes the result.

The `classify_intent` mock branch uses the required policy keywords. Retrieval always runs with local embeddings/ChromaDB. In the graded mock state, generation is deterministic and makes no network call. A real-provider branch can be layered onto the marked `MOCK_LLM=0` sections; the assignment treats that as optional.

## Example calls
Policy example:
```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery fee below INR 149?\"}"
```
General example:
```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of India?\"}"
```
Expected general response shape: `{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}`.
