import os
import tempfile
import numpy as np
import faiss
import pickle
from unittest.mock import patch

from app.embedding_utils import (
    get_embedding,
    build_faiss_index,
    save_index_and_metadata,
    load_index_and_metadata,
)
from app.models import DisasterDeclaration


@patch("app.embedding_utils.openai.embeddings.create")
def test_get_embedding_returns_vector(mock_create):
    mock_create.return_value = type("MockResponse", (), {
        "data": [type("Embed", (), {"embedding": [0.1] * 1536})()]
    })()

    vector = get_embedding("Test disaster in Texas")
    assert isinstance(vector, list)
    assert len(vector) == 1536
    assert all(isinstance(v, float) for v in vector)


@patch("app.embedding_utils.get_embedding")
def test_build_faiss_index(mock_embed):
    mock_embed.return_value = [0.1] * 10  # Simulated embedding vector

    decls = [
        DisasterDeclaration(
            disasterNumber=1,
            declarationTitle="Hurricane",
            state="TX",
            designatedArea="Harris",
            declarationType="DR",
            declarationDate="2021-08-01",
            incidentBeginDate="2021-07-29",
            incidentEndDate="2021-08-05",
            fipsStateCode="48",
            fipsCountyCode="201",
            ihProgramDeclared=True,
            iaProgramDeclared=True,
            paProgramDeclared=True,
            incidentType="Hurricane"
        )
    ]

    index, metadata = build_faiss_index(decls)
    assert isinstance(index, faiss.IndexFlatL2)
    assert index.ntotal == 1
    assert metadata == decls


def test_save_and_load_index_and_metadata():
    # Build a temporary FAISS index
    dim = 10
    index = faiss.IndexFlatL2(dim)
    vector = np.random.rand(1, dim).astype("float32")
    index.add(vector)

    metadata = ["fake_metadata"]

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "index_file.index")
        meta_path = os.path.join(tmpdir, "metadata.pkl")

        save_index_and_metadata(index, metadata, index_path, meta_path)
        assert os.path.exists(index_path)
        assert os.path.exists(meta_path)

        loaded_index, loaded_metadata = load_index_and_metadata(index_path, meta_path)

        assert isinstance(loaded_index, faiss.IndexFlatL2)
        assert loaded_index.ntotal == 1
        assert loaded_metadata == metadata
