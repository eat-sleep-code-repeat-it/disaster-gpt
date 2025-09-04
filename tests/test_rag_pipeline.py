import pytest
from datetime import date
from unittest.mock import patch

from app import rag_pipeline, embedding_utils, data_loader, constants, models


@pytest.fixture(scope="session")
def rag_fixture():
    # Load index and metadata once for all tests
    index, declarations = embedding_utils.load_index_and_metadata(
        constants.FAISS_INDEX_PATH, constants.FAISS_METADATA_PATH
    )
    rag_pipeline.index = index
    rag_pipeline.indexed_declarations = declarations
    return rag_pipeline

"""
def test_search_returns_results(rag_fixture):
    query = "flood in Texas"
    results = rag_fixture.search_similar_declarations(query, rag_fixture.index, rag_fixture.indexed_declarations)
    assert len(results) > 0
    assert isinstance(results[0], tuple)


def test_active_disaster_filtering(rag_fixture):
    query = "wildfire"
    results = rag_fixture.search_similar_declarations(
        query,
        rag_fixture.index,
        rag_fixture.indexed_declarations,
        only_active=True
    )
    today = date.today()
    for decl, _ in results:
        assert decl.incidentBeginDate <= today
        if decl.incidentEndDate:
            assert decl.incidentEndDate >= today


def test_generate_answer_contains_keywords(rag_fixture):
    query = "hurricane in Florida"
    results = rag_fixture.search_similar_declarations(query, rag_fixture.index, rag_fixture.indexed_declarations)
    answer = rag_fixture.generate_openai_answer(query, results)
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
"""

def test_validate_answer_positive():
    answer = "There was a major flood in Texas that affected several counties."
    keywords = ["flood", "Texas"]
    assert rag_pipeline.validate_answer(answer, keywords) is True


def test_validate_answer_negative():
    answer = "I don't know."
    keywords = ["earthquake", "California"]
    assert rag_pipeline.validate_answer(answer, keywords) is False


def test_is_active_disaster_true():
    decl = models.DisasterDeclaration(
        disasterNumber=1234,
        declarationTitle="Test Event",
        state="CA",
        designatedArea="Test County",
        declarationType="DR",
        declarationDate=date(2023, 1, 1),
        incidentBeginDate=date(2023, 9, 1),
        incidentEndDate=None,
        fipsStateCode="06",
        fipsCountyCode="001",
        ihProgramDeclared=True,
        iaProgramDeclared=False,
        paProgramDeclared=True,
        incidentType="Fire"
    )
    assert rag_pipeline.is_active_disaster(decl) is True

"""
@patch("app.rag_pipeline.openai.chat.completions.create")
def test_generate_openai_answer_mock(mock_openai, rag_fixture):
    mock_openai.return_value.choices[0].message.content = "Mocked answer."
    query = "earthquake in California"
    results = rag_fixture.search_similar_declarations(query, rag_fixture.index, rag_fixture.indexed_declarations)
    answer = rag_fixture.generate_openai_answer(query, results)
    assert answer == "Mocked answer."


@patch("app.rag_pipeline.openai.chat.completions.create")
def test_full_rag_pipeline_output(mock_openai, rag_fixture):
    mock_openai.return_value.choices[0].message.content = '{"relevance": 5, "accuracy": 5, "completeness": 5, "comments": "Great job."}'
    query = "active flood in Louisiana"
    response = rag_fixture.chat_rag_fn(query, [])
    assert "Answer" in response
    assert "Evaluation" in response
"""