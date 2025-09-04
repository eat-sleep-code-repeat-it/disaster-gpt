
#  DisasterGPT – Ask Anything About U.S. Disaster Declarations

 
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
python disaster_gpt.py
```

## Sample questions
```js
is there an active disaster in Washington County, Oregon?
is there an disaster in Riverside, California?
is there an active disaster in Riverside, California? 
active fire disasters?
``

## ✅ What we Get:
- A clean chat UI (like ChatGPT)
- Each message sends a user query
- RAG pipeline is run
    - create index and save it to local vector file
    - or load index from the saved vector file
- Assistant returns a structured answer using openai
- Built with Guardrail and evaluation
- Previous messages are preserved in context

## Suggested Project Structure
```bash
disaster-gpt/
├── app/
│   ├── __init__.py
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
│   └── logo.png
│
├── tests/
│   └── test_rag_pipeline.py  # 🔹 Unit tests (pytest)
│
├── .env                      # 🔹 OpenAI keys, etc.
├── .gitignore
├── README.md
├── requirements.txt
└── run.sh                    # 🔹 Simple launcher (optional)


app/main.py	Launches the Gradio UI (e.g., gr.ChatInterface)
app/rag_pipeline.py	Contains rag_pipeline() and chat_rag_fn() logic
app/models.py	Pydantic DisasterDeclaration and other data models
app/embedding_utils.py	Embedding + FAISS build/save/load
app/answer_eval.py	GPT-based evaluation and keyword-based guardrails
data/	Static data source (e.g., CSVs)
saved_index/	Stores generated FAISS index and metadata
tests/	Optional test suite using pytest or unittest
This structure:

Keeps Gradio app logic clean and isolated
Separates data/model logic from UI
Makes it easier to maintain, extend, or even deploy later (e.g., with FastAPI or Streamlit)
```

