import os
import csv
from typing import List, Optional, Tuple
from pydantic import ValidationError
from datetime import date
import openai
import numpy as np
import faiss
import pickle
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

from models import DisasterDeclaration
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")




def is_active_disaster(declaration: DisasterDeclaration, today: Optional[date] = None) -> bool:
    if today is None:
        today = datetime.today().date()
    # If no end date, assume it's still active if today >= begin date
    if declaration.incidentEndDate is None:
        return declaration.incidentBeginDate <= today
    return declaration.incidentBeginDate <= today <= declaration.incidentEndDate

def evaluate_with_gpt_judge(query: str, context: str, answer: str) -> str:
    eval_prompt = f"""
    You are evaluating an AI assistant's disaster response.

    User Query: {query}

    Retrieved Context:
    {context}

    AI Answer:
    {answer}

    Evaluate on:
    - Relevance (1-5)
    - Accuracy (1-5)
    - Completeness (1-5)

    Return your scores and one comment as JSON like:
    {{"relevance": 4, "accuracy": 4, "completeness": 5, "comments": "Accurate and complete."}}
    """
    judge_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0
    )
    return judge_response.choices[0].message.content.strip()

def validate_answer(answer: str, required_keywords: List[str]) -> bool:
    """
    Guardrail:    Blocks short or irrelevant answers that don't mention keywords from the query
    """
    if len(answer.strip()) < 10:
        return False
    for keyword in required_keywords:
        if keyword.lower() not in answer.lower():
            return False
    return True

def save_index_and_metadata(index, metadata, index_path, metadata_path):
    """
    Storing FAISS index and embeddings to disk 
    FAISS stores only the vectors
    Need to keep track of the original data (e.g., disaster declarations) corresponding to each vector.
    and reloading them later to avoid recomputing embeddings and rebuilding the index every time you run the script.
    This saves a ton of time and API usage.
    """
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

def load_index_and_metadata(index_path, metadata_path):
    """
    Reload  FAISS index and embeddings to avoid recomputing embeddings and rebuilding the index every time you run the script.
    This saves a ton of time and API usage.
    """    
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
    
def read_disaster_declarations_from_csv(file_path: str) -> List[DisasterDeclaration]:
    """
    load declarations as structured date model from csv
    """
    declarations = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Convert disasterNumber to int explicitly
                if 'disasterNumber' in row and row['disasterNumber']:
                    row['disasterNumber'] = int(row['disasterNumber'])
                decl = DisasterDeclaration(**row)
                declarations.append(decl)
            except ValidationError as e:
                print(f"Skipping invalid row: {e}")
    print(f"Loaded {len(declarations)} disaster declarations from CSV")
    return declarations

def create_text_for_embedding(declaration: DisasterDeclaration, score: Optional[float] = None) -> str:
    parts = [
        f"DisasterNumber: {declaration.disasterNumber}",
        f"State: {declaration.state}",
        f"County: {declaration.designatedArea or 'N/A'}",
        f"DeclarationType: {declaration.declarationType or 'N/A'}",
        f"DeclarationDate: {declaration.declarationDate or 'N/A'}",
        f"declarationTitle: {declaration.declarationTitle or 'N/A'}",
        f"IncidentType: {declaration.incidentType or 'N/A'}",
        f"IncidentBeginDate: {declaration.incidentBeginDate or 'N/A'}",
        f"IncidentEndDate: {declaration.incidentEndDate or 'N/A'}"
    ]
    if score is not None:
        parts.append(f"FAISS Score: {round(score, 3)}")
    return ". ".join(parts)

def get_embedding(text: str) -> List[float]:
    """
    Embeds the structured disaster description using OpenAI Embeddings API.
    """
    response = openai.embeddings.create(
        model="text-embedding-3-small", #model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def build_faiss_index(declarations: List[DisasterDeclaration]) -> Tuple[faiss.IndexFlatL2, List[DisasterDeclaration]]:    
    """
    Creates and populates a FAISS vector index from disaster embeddings.
    """
    print("Creating embeddings and building FAISS index...")
    embeddings = []
    for decl in declarations:
        print(f"create_text_for_embedding {decl.disasterNumber}...")
        text = create_text_for_embedding(decl)
        emb = get_embedding(text)
        embeddings.append(emb)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index, declarations

def search_similar_declarations(
    query: str,
    index: faiss.IndexFlatL2,
    declarations: List[DisasterDeclaration],
    top_k=5,
    only_active: bool = False
) -> List[Tuple[DisasterDeclaration, float]]:
    query_emb = np.array([get_embedding(query)]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    results = []
    today = datetime.today().date()
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            decl = declarations[idx]
            if only_active and not is_active_disaster(decl, today):
                continue
            results.append((decl, dist))
    return results

def generate_openai_answer(query: str, results: List[Tuple[DisasterDeclaration, float]]) -> str:
    if not results:
        return f"Sorry, I found no disaster declarations related to your query: '{query}'."

    context_text = "\n".join(create_text_for_embedding(decl, score) for decl, score in results) 

    prompt = (
        f"You are a helpful assistant. Using the disaster declarations below, answer the question:\n"
        f"Query: {query}\n\n"
        f"Disaster Declarations:\n{context_text}\n\n"
        f"Answer:"
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant. Only answer questions using the disaster declaration data provided. "
                "If the data does not contain an answer, say 'No relevant disaster declarations found.' "
                "Do not guess or fabricate information. Format the answer clearly."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def chat_rag_fn(user_message: str, chat_history: list) -> Tuple[str, list]:
    only_active = "active" in user_message.lower()
    results = search_similar_declarations(user_message, index, indexed_declarations, top_k=5, only_active=only_active)
    answer = generate_openai_answer(user_message, results)

    if not validate_answer(answer, [user_message.split()[0]]):
        response = "⚠️ Guardrail Triggered: Answer may be incomplete or irrelevant."
        return response

    context = "\n".join(create_text_for_embedding(d) for d, _ in results)
    evaluation = evaluate_with_gpt_judge(user_message, context, answer)

    match_summary = "\n".join(
        f"- Disaster {decl.disasterNumber} ({decl.state}, {decl.designatedArea}, {decl.incidentBeginDate}, {decl.incidentEndDate}): Score {round(score, 4)}"
        for decl, score in results
    )

    response = (
        f"**Top Matches:**\n{match_summary}\n\n"
        f"🧠 **Answer:** {answer}\n\n"
        f"📊 **Evaluation:**\n{evaluation}"
    )
    return response

def main():
    global index, indexed_declarations

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "data", "disaster_declarations.csv")
    disaster_declarations = read_disaster_declarations_from_csv(csv_file)

    index_file = os.path.join(script_dir, "saved_index","disaster_faiss.index")
    index_metadata_file = os.path.join(script_dir, "saved_index", "disaster_metadata.pkl")
    if os.path.exists(index_file) and os.path.exists(index_metadata_file):
        index, indexed_declarations = load_index_and_metadata(index_file, index_metadata_file)
    else:
        index, indexed_declarations = build_faiss_index(disaster_declarations)
        save_index_and_metadata(index, indexed_declarations, index_file, index_metadata_file)
    
    # 🚀 Chat Interface with history
    gr.ChatInterface(
        fn=chat_rag_fn,
        title="🌀 Disaster AI Assistant",
        description="Ask about FEMA disasters by state/county or general questions. Data powered by FEMA + FAISS + OpenAI."
    ).launch()

if __name__ == "__main__":
    main()