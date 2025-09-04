# tests/conftest.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import rag_pipeline, embedding_utils, constants
import pytest

@pytest.fixture(scope="session")
def rag_pipeline_fixture():
    # Load from disk or re-init
    index, declarations = embedding_utils.load_index_and_metadata(
        constants.FAISS_INDEX_PATH, constants.FAISS_METADATA_PATH
    )
    rag_pipeline.index = index
    rag_pipeline.indexed_declarations = declarations
    return rag_pipeline