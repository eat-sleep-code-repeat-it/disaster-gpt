
from typing import List
from app.ai_client import OpenAIClient
from app.constants import EVAL_MODEL, MIN_ANSWER_LENGTH

_ai_client = None
def set_ai_client(ai_client: OpenAIClient):
    global _ai_client
    _ai_client = ai_client

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
    judge_response = _ai_client.client.chat.completions.create(
        model=EVAL_MODEL,
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
    if len(answer.strip()) < MIN_ANSWER_LENGTH:
        return False
    for keyword in required_keywords:
        if keyword.lower() not in answer.lower():
            return False
    return True
