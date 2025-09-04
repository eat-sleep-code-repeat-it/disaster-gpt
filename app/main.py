from app.constants import DISASTER_CSV_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH
from app.rag_pipeline import chat_rag_fn, setup_rag_pipeline
from app.embedding_utils import load_index_and_metadata, build_faiss_index, save_index_and_metadata
from app.data_loader import read_disaster_declarations_from_csv
import gradio as gr
import os
import openai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    # Load/build index and declarations
    disaster_declarations = read_disaster_declarations_from_csv(DISASTER_CSV_PATH)
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        index, indexed_declarations = load_index_and_metadata(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    else:
        index, indexed_declarations = build_faiss_index(disaster_declarations)
        save_index_and_metadata(index, indexed_declarations, FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    
    # Initialize rag_pipeline with data
    setup_rag_pipeline(index, indexed_declarations)

    # 🚀 Launch Gradio Chat Interface with history
    gr.ChatInterface(
        fn=chat_rag_fn,
        title="🌀 Disaster AI Assistant",
        description="Ask about FEMA disasters by state/county or general questions. Data powered by FEMA + FAISS + OpenAI."
    ).launch()

if __name__ == "__main__":
    main()