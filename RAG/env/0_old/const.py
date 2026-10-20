MAX_ROUND = 3  # max try times of one agent talk

question_to_query_template = """
You are an expert at crafting precise search queries based on a provided question. Your task is to generate a well-structured search query that will help retrieve relevant external documents containing information needed to answer the question.

==============================
Here are several typical examples:

Demonstration example 1:
question: “Gustav Mahler composed a beautiful piece performed by the Bach-Elgar Choir. What is the name of that piece??”
search query: “Gustav Mahler piece performed by Bach-Elgar Choir”

Demonstration example 2:
question: “Merle Reagle did crosswords for what magazine that has a focus on aging issues?”
search query: “Merle Reagle crosswords magazine aging issues”

==============================
Here is a new question:

question:
{question}

Now give your search query following this format strictly without other explanation:

search query:

"""

context_question_to_query_template = """
You are good at extract relevant details from the provided context and question. Your task is to propose an effective search query that will help retrieve additional information to answer the question. The search query should target the missing information while avoiding redundancy.

==============================
Here is the context and question:

context:
{context}
question:
{question}

Now give your search query following this format strictly without other explanation:

search query:

"""

context_question_to_answer_template = """
You are an expert at answering questions based on provided documents. Your task is to formulate a clear, accurate, and concise answer to the given question by using the retrieved context (documents) as your source of information. Please ensure that your answer is well-grounded in the context and directly addresses the question.

Answer it shortly. I don't need any explanations, just the direct answer. 
For example, if your answer woule be: "No, the provided context does not support this statement. The documents do not mention 'The Age' or 'TechCrunch', and the information attributed to 'The Verge' suggests Google's deals were seen as effective for gaining market share, not unnecessary."
Just output "No". Do not output later explanation!

==============================
Here are several typical examples:

Demonstration 1:
context: ["The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.",
    "She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.",
    "The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means."
]

question: “Does 'The Verge' article suggest that Google's deals with companies like Apple are unnecessary due to a plethora of alternatives, while 'The Age' and 'TechCrunch' articles imply Google's actions are primarily driven by profit maximization and anticompetitive behavior, respectively?”

answer: “no”

Demonstration 2:
context: ["Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.",
    "It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.",
    "Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.",
    "The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means."
]

question: “Which company, reported by TechCrunch, has been involved in antitrust cases for its practices in becoming the default search engine, influencing app distribution and payment processing markets, showcasing new hardware developments, and impacting news publishers' content and revenue?”

answer: “Google”


==============================
Here is the new context, question:

context:
{context}
question:
{question}

Now give your answer following this format strictly without other explanation:

answer:

"""

context_question_to_answer_cot_template = """
You are an expert at answering questions based on provided documents. Your task is to formulate a clear, accurate, and concise answer to the given question by using the retrieved context (documents) as your source of information. Please ensure that your answer is well-grounded in the context and directly addresses the question.
Answer it shortly. I don't need any explanations, just direct answer.

Answer it shortly. I don't need any explanations, just the direct answer. 
For example, if your answer woule be: "No, the provided context does not support this statement. The documents do not mention 'The Age' or 'TechCrunch', and the information attributed to 'The Verge' suggests Google's deals were seen as effective for gaining market share, not unnecessary."
Just output "No". Do not output later explanation!

==============================
Here are several typical examples:

Demonstration 1:
context: ["The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.",
    "She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.",
    "The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means."
]

question: “Does 'The Verge' article suggest that Google's deals with companies like Apple are unnecessary due to a plethora of alternatives, while 'The Age' and 'TechCrunch' articles imply Google's actions are primarily driven by profit maximization and anticompetitive behavior, respectively?”

answer: “no”

Demonstration 2:
context: ["Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.",
    "It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.",
    "Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.",
    "The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means."
]

question: “Which company, reported by TechCrunch, has been involved in antitrust cases for its practices in becoming the default search engine, influencing app distribution and payment processing markets, showcasing new hardware developments, and impacting news publishers' content and revenue?”

answer: “Google”

==============================
Here is the new context, question:

context:
{context}
question:
{question}

Let's solve this problem step by step before giving the final response, your answer should follow this format strictly without other explanation

answer:

"""

PAIRWISE_RERANK_PROMPT = """You are a careful QA reranker. Score how useful the passage is for answering the user’s query.

Output strictly as compact JSON on one line: {{"score": <0-100 integer>, "justification": "<≤200 chars>"}}

Scoring rubric (0–100):
- 90–100: Directly answers the query (facts, entities, numbers, or steps are present). High coverage, low ambiguity.
- 70–89: Strongly relevant; contains key evidence but may miss a detail or require synthesis.
- 40–69: Tangential or partial; mentions some entities/terms but insufficient to answer fully.
- 1–39: Mostly off-topic, meta, or background noise.
- 0: Contradictory or unrelated.

Rules:
- Base judgment ONLY on the passage text. Do not use outside knowledge.
- Reward faithful evidence (entities, numbers, definitions, steps).
- Penalize contradictions, vague mentions, or generic boilerplate.
- If the question asks for a definition, formula, or specific fact, require it to appear in the passage.

Query:
{query}

Passage:
{passage}
"""

LISTWISE_REFINE_PROMPT = """You are a reranker. Given a user query and multiple candidate passages, produce a strict JSON array of objects sorted from most to least useful.

Return one line of JSON:
[{{"id":"<id>","score":<0-100 integer>,"why":"<≤160 chars>"}}, ...]

Rules:
- Judge ONLY from the passage contents provided.
- Prefer passages that directly answer the query with specific evidence (entities, numbers, definitions, steps).
- If two are similar, break ties by precision and completeness.
- Do not invent information.

Query:
{query}

Candidates:
{candidates_block}
"""
