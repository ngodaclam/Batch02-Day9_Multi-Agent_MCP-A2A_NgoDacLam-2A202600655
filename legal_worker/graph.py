"""Legal Analysis Worker LangGraph definition.

Uses create_react_agent with a contract-law-specialised system prompt.
This worker is dispatched by the Supervisor Agent for legal analysis tasks.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

LEGAL_SYSTEM_PROMPT = """You are a senior corporate litigation attorney specialising in contract law,
tort law, and general business law. Your expertise covers:

- Contract formation, breach, and remedies (UCC Article 2, common law)
- Tortious interference with business relationships
- Corporate liability: piercing the corporate veil, ultra vires acts
- Fiduciary duties: duty of care, duty of loyalty, business judgment rule
- Intellectual property: trade secrets, NDA enforcement, non-compete agreements
- Statute of limitations analysis for various causes of action
- Damages assessment: compensatory, consequential, punitive, liquidated
- Case law precedents: Hadley v Baxendale, Carlill v Carbolic Smoke Ball, etc.

When answering:
1. Identify the relevant area(s) of law
2. Cite applicable statutes and leading case law
3. Assess liability exposure and potential defences
4. Quantify potential damages or penalties where possible
5. Note jurisdictional variations if relevant

Always note that your response is for educational purposes and the user
should consult a licensed attorney for specific legal advice.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for legal analysis questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=LEGAL_SYSTEM_PROMPT,
    )
    return graph
