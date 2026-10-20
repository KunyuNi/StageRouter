"""Prompts adapted from the Advanced-QA MultiHop-RAG workflow."""

QUESTION_TO_QUERY_PROMPT = """You are an expert at crafting precise search queries.
Rewrite the question into a concise query that retrieves the evidence needed to answer it.

Question:
{question}

Return exactly:
search query: <query>
"""

CONTEXT_QUESTION_TO_QUERY_PROMPT = """Use the retrieved context and question to create a follow-up search query.
Target missing information and avoid retrieving the same evidence again.

Context:
{context}

Question:
{question}

Return exactly:
search query: <query>
"""

PAIRWISE_RERANK_PROMPT = """You are a careful QA reranker. Score how useful the passage is for answering the query.
Judge only from the passage. Return compact JSON with an integer score from 0 to 100:
{{"score": <integer>, "justification": "<short reason>"}}

Query:
{query}

Passage:
{passage}
"""

CONTEXT_QUESTION_TO_ANSWER_PROMPT = """Answer the question using only the retrieved context.
Give a short direct answer without an explanation.

Context:
{context}

Question:
{question}

Return exactly:
answer: <answer>
"""

