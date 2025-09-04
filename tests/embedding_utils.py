import os
import pickle
import numpy as np
import faiss
import openai
from typing import List, Tuple, Optional
from app.models import DisasterDeclaration

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate embedding vector for the given text using OpenAI Embeddings API.
    """
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def create_text_for_embedding(declaration: DisasterDeclaration, score: Optional[float] = None) -> str:
    """
    Serialize disaster declaration data to a text string for embedding.
    """
    parts = [
        f"DisasterNumber: {declaration.disasterNumber}",
        f"State: {declaration.state}",
        f"County: {declaration.designatedArea or 'N/A'}",
        f"DeclarationType: {declaration.declarationType or 'N/A'}",
        f"DeclarationDate: {declaration.declarationDate or 'N/A'}",
        f"DeclarationTitle: {declaration.declarationTitle or 'N/A'}",
        f"IncidentType: {declaration.incidentType or 'N/A'}",
        f"IncidentBeginDate: {declaration.incidentBeginDate or 'N/A'}",
        f"IncidentEndDate: {declaration.incidentEndDate or 'N/A'}"
    ]
    if score is not None:
        parts.append(f"FAISS Score: {round(score, 3)}")
    return ". ".join(parts)


def build_faiss_index(declarations: List[DisasterDeclaration]) -> Tuple[faiss.IndexFlatL2, List[DisasterDeclaration]]:
    """
    Create a FAISS index from a list of disaster declarations by generating embeddings.
    Returns the FAISS index and the list of declarations (metadata).
    """
    print("Creating embeddings and building FAISS index...")
    embeddings = []
    for decl in declarations:
        print(f"Embedding disasterNumber {decl.disasterNumber}...")
        text = create_text_for_embedding(decl)
        emb = get_embedding(text)
        embeddings.append(emb)

    if not embeddings:
        raise ValueError("No embeddings generated; empty declarations list?")

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index, declarations


def save_index_and_metadata(index: faiss.IndexFlatL2, metadata: List[DisasterDeclaration], index_path: str, metadata_path: str):
    """
    Save the FAISS index and metadata (declarations) to disk.
    """
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def load_index_and_metadata(index_path: str, metadata_path: str) -> Tuple[faiss.IndexFlatL2, List[DisasterDeclaration]]:
    """
    Load the FAISS index and metadata from disk.
    """
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
