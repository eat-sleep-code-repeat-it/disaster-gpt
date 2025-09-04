from typing import List, Optional, Tuple
import openai
import numpy as np
import faiss
import pickle

from app.constants import EMBEDDING_MODEL
from app.models import DisasterDeclaration


def get_embedding(text: str) -> List[float]:
    """
    Embeds the structured disaster description using OpenAI Embeddings API.
    """
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding
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
    