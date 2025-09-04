
# Test Disaster Retrieval Works
def test_search_returns_results(rag_pipeline_fixture):
    results = rag_pipeline_fixture.search_similar_declarations("flood in Texas", top_k=5)
    assert len(results) > 0
    assert isinstance(results[0], tuple) 

# Test Active Disaster Filtering
def test_active_disaster_filtering(rag_pipeline_fixture):
    results = rag_pipeline_fixture.search_similar_declarations(
        "wildfire", top_k=5, only_active=True
    )
    for decl, _ in results:
        assert decl.incidentBeginDate <= date.today()
        if decl.incidentEndDate:
            assert decl.incidentEndDate >= date.today()

# Test Answer Generation
def test_generate_answer_contains_keywords(rag_pipeline_fixture):
    query = "hurricane in Florida"
    results = rag_pipeline_fixture.search_similar_declarations(query, top_k=5)
    answer = rag_pipeline_fixture.generate_openai_answer(query, results)
    assert "hurricane" in answer.lower() or "florida" in answer.lower()
    assert len(answer.strip()) > 0

# Test Guardrail Validation
def test_validate_answer_positive():
    answer = "There was a major flood in Texas that affected several counties."
    keywords = ["flood", "Texas"]
    from app.rag_pipeline import validate_answer
    assert validate_answer(answer, keywords) is True
def test_validate_answer_negative():
    answer = "I don't know."
    keywords = ["earthquake", "California"]
    from app.rag_pipeline import validate_answer
    assert validate_answer(answer, keywords) is False    

# Test Full RAG Pipeline End-to-End
def test_full_rag_pipeline_output(rag_pipeline_fixture):
    query = "active flood in Louisiana"
    response = rag_pipeline_fixture.chat_rag_fn(query, [])
    assert "Answer" in response
    assert "Evaluation" in response