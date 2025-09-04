from app.ai_client import OpenAIClient
from app.constants import DISASTER_CSV_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH
from app.rag_pipeline import chat_rag_fn, setup_rag_pipeline
from app.embedding_utils import load_index_and_metadata, build_faiss_index, save_index_and_metadata
from app.data_loader import read_disaster_declarations_from_csv
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
verify_ssl = False
ai_client = OpenAIClient(api_key, verify_ssl)

def main():
    # Load/build index and declarations
    disaster_declarations = read_disaster_declarations_from_csv(DISASTER_CSV_PATH)
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        index, indexed_declarations = load_index_and_metadata(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    else:
        index, indexed_declarations = build_faiss_index(disaster_declarations, ai_client)
        save_index_and_metadata(index, indexed_declarations, FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    
    # Set AI client for anwser evaluation
    from app.answer_eval import set_ai_client
    set_ai_client(ai_client)

    # Initialize rag_pipeline with data
    setup_rag_pipeline(index, indexed_declarations, ai_client)

    # Launch Gradio Chat Interface with history
    gr.ChatInterface(
        fn=chat_rag_fn,
        title="🌀 Disaster AI Assistant",
        description="Ask about FEMA disasters by state/county or general questions. Data powered by FEMA + FAISS + OpenAI."
    ).launch()

if __name__ == "__main__":
    main()