
from src.rag_chatbot.rag.retrieval_utils import retrieve_context
from src.rag_chatbot.rag.env import deployment_name, client


def generate_response(context: list[dict], user_query: str) -> str:
    context_texts = [doc["content"] for doc in context]

    context_block = "\n\n---\n\n".join(context_texts)

    system_prompt = """
    You are a helpful assistant.

    Use the retrieved document context when it is relevant to the user's question.
    If the user's question is about the retrieved documents, answer from that context and do not invent missing facts.
    If the retrieved context is irrelevant or insufficient and the user is asking a general question, answer using your general knowledge.
    If the user is specifically asking about the documents and the answer is not contained in them, say that the documents do not contain enough information.

    When useful, make it clear whether your answer is based on the documents or on general knowledge.
    """

    user_prompt = f"""
    Retrieved context:
    {context_block}

    User question:
    {user_query}

    Answer:
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        #temperature=0.2
    )

    return response.choices[0].message.content



def generate_contextualized_response(inputs: dict) -> dict:
    user_query = inputs["question"]

    user_query = user_query.strip()
    context_results = retrieve_context(user_query)
    answer = generate_response(context_results, user_query)
    return {
        "answer": answer,
        "prompt": user_query,
        "retrieved": context_results
    }
