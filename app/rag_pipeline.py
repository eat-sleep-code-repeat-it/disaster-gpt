from typing import List, Optional, Tuple
from datetime import date
import openai
import numpy as np
import faiss
from datetime import datetime
from app.answer_eval import evaluate_with_gpt_judge, validate_answer
from app.constants import CHAT_MODEL, TOP_K
from app.embedding_utils import create_text_for_embedding, get_embedding
from app.models import DisasterDeclaration

"""
main "chat" function that ties retrieval + generation + validation + evaluation together
""" 
def setup_rag_pipeline(index, declarations):
    global _index, _indexed_declarations
    _index = index
    _indexed_declarations = declarations
def chat_rag_fn(user_message: str, chat_history: list) -> Tuple[str, list]:
    only_active = "active" in user_message.lower()
    results = search_similar_declarations(user_message, _index, _indexed_declarations, top_k=TOP_K, only_active=only_active)
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
        f"🙋 **Question:** {user_message}\n\n"
        f"**Top Matches:**\n{match_summary}\n\n"
        f"🧠 **Answer:** {answer}\n\n"
        f"📊 **Evaluation:**\n{evaluation}"
    )
    return response
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
        model=CHAT_MODEL,
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
def search_similar_declarations(
    query: str,
    index: faiss.IndexFlatL2,
    declarations: List[DisasterDeclaration],
    top_k=TOP_K,
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
def is_active_disaster(declaration: DisasterDeclaration, today: Optional[date] = None) -> bool:
    if today is None:
        today = datetime.today().date()
    # If no end date, assume it's still active if today >= begin date
    if declaration.incidentEndDate is None:
        return declaration.incidentBeginDate <= today
    return declaration.incidentBeginDate <= today <= declaration.incidentEndDate


