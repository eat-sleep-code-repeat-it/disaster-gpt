
#  DisasterGPT 

Ask Anything About U.S. Disaster Declarations.

## How to run 
```bash
# Python 3.13.7

python -m venv venv
venv\Scripts\activate 

pip install gradio
pip install openai
pip install faiss-cpu
pip install python-dotenv
pip freeze > requirements.txt


# set OPENAI_API_KEY in .env file
pip install -r requirements.txt
python -m app.main
python -m app.main --no-verify-ssl


pip install pytest
pytest tests/
pytest tests/test_answer_eval.py
pytest tests/test_embedding_utils.py
pytest tests/test_rag_pipeline.py
```

## Sample Prompts
```js
is there an active disaster in Washington County, Oregon? YES & Evaluation
is there an disaster in Riverside, California?  YES & Evaluation
is there an active disaster in Riverside, California? NO & Evaluation
active fire disasters? YES,  Evaluation
give all fire disasters? NO,  Evaluation & Guardrail
```

## What we Get:
- A clean chat UI (like ChatGPT)
- Each message sends a user query
- RAG pipeline is run
    - create index and save it to local vector file
    - or load index from the saved vector file
- Assistant returns a structured answer using openai
- Built with Guardrail and evaluation
- Previous messages are preserved in context

## Project Structure
- Keeps Gradio app logic clean and isolated
- Separates data/model logic from UI
- Makes it easier to maintain, extend, or even deploy later (e.g., with FastAPI or Streamlit)

```bash
disaster-gpt/
├── app/
│   ├── main.py               # 🔹 Main Gradio app (entry point)
│   ├── rag_pipeline.py       # 🔹 RAG logic (retrieval + answer generation) 
│   ├── models.py             # 🔹 Pydantic models (e.g., DisasterDeclaration)
│   ├── embedding_utils.py    # 🔹 Embedding + FAISS index handling
│   ├── answer_eval.py        # 🔹 Guardrails & GPT-based evaluation
│   ├── data_loader.py        # 🔹 Load/parse CSV data
│   └── constants.py          # 🔹 Paths, constants, config keys
│
├── data/
│   └── disaster_declarations.csv  # 🔹 Source dataset
│
├── saved_index/              # 🔹 Store FAISS index & metadata
│   ├── disaster_faiss.index
│   └── disaster_metadata.pkl
│
├── assets/                   # 🔹 (Optional) Images, logos, docs
│   └── README.md
│
├── tests/
│   └── test_rag_pipeline.py  # 🔹 Unit tests (pytest)
│
├── .env                      # 🔹 OpenAI keys, etc.
├── .gitignore
├── README.md
├── requirements.txt

app/main.py	Launches the Gradio UI (e.g., gr.ChatInterface)
app/rag_pipeline.py	Contains rag_pipeline() and chat_rag_fn() logic
app/models.py	Pydantic DisasterDeclaration and other data models
app/embedding_utils.py	Embedding + FAISS build/save/load
app/answer_eval.py	GPT-based evaluation and keyword-based guardrails
data/	Static data source (e.g., CSVs)
saved_index/	Stores generated FAISS index and metadata
tests/	Optional test suite using pytest or unittest
```

## tests
```bash
Disaster retrieval	search_similar_declarations()
Active disaster filtering	is_active_disaster()
Answer generation	generate_openai_answer()
Guardrails	validate_answer()
Full RAG pipeline test	chat_rag_fn()
OpenAI mocking	@patch(...)
```

## Future

Adapt retrieval + context + prompt formatting pipeline to conform to the MCP standard (or parts of it), you can plug into an MCP-compatible server or client.

### Scenario 1: You want to expose RAG pipeline as an MCP-compatible server
```js
Wrap retrieval and response code into an MCP-compatible handler (e.g., using LangChain's MCP tools)
Expose a REST or HTTP API using FastAPI or Flask
Follow the MCP schema for request/response (context, query, history, etc.)
🧱 Example: Using LangChain’s langchain-mcp
```

### Scenario 2: You want to consume an existing MCP server
```js
Let’s say you want to delegate RAG to an MCP server, not implement one.
Format queries using the MCP request schema (user_input, retrieval_context, history, etc.)
Send that to the MCP server endpoint (e.g., via requests.post())
Receive the MCP-formatted answer
Display in Gradio app
```

### To integrate MCP into the current Gradio RAG app
1. Define MCP request/response format
2. Modify chat_rag_fn() to use MCP
3. (Optional) Expose an MCP server
```bash
Use MCP server for answers	Send query, history, and context to MCP API
Make my own MCP server	    Wrap my RAG logic in a FastAPI MCP route
Match MCP schema	        Use user_input, context, history fields
```


