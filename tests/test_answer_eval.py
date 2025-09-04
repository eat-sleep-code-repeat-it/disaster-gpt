import pytest
from unittest.mock import patch
from app import answer_eval


def test_validate_answer_passes():
    answer = "A flood occurred in Houston, Texas in 2022."
    keywords = ["flood", "Texas"]
    result = answer_eval.validate_answer(answer, keywords)
    assert result is True


def test_validate_answer_fails_on_short_answer():
    answer = "Yes."
    keywords = ["earthquake"]
    result = answer_eval.validate_answer(answer, keywords)
    assert result is False


def test_validate_answer_fails_on_missing_keyword():
    answer = "There was a disaster in Florida."
    keywords = ["earthquake", "California"]
    result = answer_eval.validate_answer(answer, keywords)
    assert result is False

"""


from unittest.mock import patch
@patch("openai.chat.completions.create")
def test_evaluate_with_gpt_judge_mock(mock_create):
    # Setup mock response
    mock_create.return_value = type("obj", (object,), {
        "choices": [type("obj", (object,), {"message": type("obj", (object,), {"content": '{"relevance": 5, "accuracy": 5, "completeness": 5, "comments": "Good."}'})() })()]
    })()

    result = answer_eval.evaluate_with_gpt_judge("query", "context", "answer")
    assert "relevance" in result
"""