"""Supervisor Agent LangGraph StateGraph definition.

Implements the Supervisor-Workers pattern:
  classify_question → dispatch_workers → (parallel) call_legal + call_tax + call_compliance → synthesize → END

The Supervisor does NOT perform legal analysis itself. Instead, it:
1. Classifies the question to determine which specialist workers are needed
2. Dispatches workers in parallel via LangGraph Send API
3. Synthesizes all worker results into a comprehensive final answer
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class SupervisorState(TypedDict):
    question: str
    context_id: str
    trace_id: str
    delegation_depth: int
    # Routing decisions
    needs_legal: bool
    needs_tax: bool
    needs_compliance: bool
    # Worker results (Annotated so parallel branches can both write without conflict)
    legal_result: Annotated[str, _last_wins]
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    # Final synthesized answer
    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def classify_question(state: SupervisorState) -> dict:
    """Supervisor classifies the question to determine which workers are needed.

    Uses LLM to analyse the question and return a JSON routing decision.
    This node does NOT perform any legal analysis — it only decides routing.
    """
    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:
        logger.info("Max delegation depth reached (%d); skipping all workers", depth)
        return {"needs_legal": False, "needs_tax": False, "needs_compliance": False}

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                'You are a Supervisor Agent that routes legal questions to specialist workers.\n'
                'Based on the question, decide which specialist workers are needed.\n'
                'Reply with ONLY valid JSON — no markdown, no extra text:\n'
                '{"needs_legal": <true|false>, "needs_tax": <true|false>, "needs_compliance": <true|false>}\n\n'
                'needs_legal = true → question involves contract law, tort, corporate liability, IP, general legal issues\n'
                'needs_tax = true → question involves tax law, IRS, tax evasion, penalties, corporate tax\n'
                'needs_compliance = true → question involves regulatory compliance, SEC, SOX, AML, FCPA, GDPR'
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    raw = result.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Routing LLM returned non-JSON: %r — defaulting to all=True", raw)
        parsed = {"needs_legal": True, "needs_tax": True, "needs_compliance": True}

    needs_legal = bool(parsed.get("needs_legal", True))
    needs_tax = bool(parsed.get("needs_tax", True))
    needs_compliance = bool(parsed.get("needs_compliance", True))
    logger.info(
        "Supervisor routing decision: needs_legal=%s needs_tax=%s needs_compliance=%s",
        needs_legal, needs_tax, needs_compliance,
    )
    return {
        "needs_legal": needs_legal,
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
    }


def dispatch_workers(state: SupervisorState) -> list[Send]:
    """Routing function: dispatch parallel Send objects to workers based on classification.

    This function is used with add_conditional_edges; it returns a list of
    Send objects which LangGraph executes as parallel branches.
    """
    sends: list[Send] = []
    if state.get("needs_legal"):
        sends.append(Send("call_legal", state))
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    if not sends:
        # No workers needed — go straight to synthesis
        sends.append(Send("synthesize", state))
    logger.info("Supervisor dispatching %d worker(s): %s", len(sends), [s.node for s in sends])
    return sends


async def call_legal(state: SupervisorState) -> dict:
    """Delegate to the Legal Analysis Worker via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("legal_analysis_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Legal Worker returned %d chars", len(result))
        return {"legal_result": result}
    except Exception as exc:
        logger.exception("call_legal failed: %s", exc)
        return {"legal_result": f"[Legal analysis unavailable: {exc}]"}


async def call_tax(state: SupervisorState) -> dict:
    """Delegate to the Tax Worker via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("tax_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Tax Worker returned %d chars", len(result))
        return {"tax_result": result}
    except Exception as exc:
        logger.exception("call_tax failed: %s", exc)
        return {"tax_result": f"[Tax analysis unavailable: {exc}]"}


async def call_compliance(state: SupervisorState) -> dict:
    """Delegate to the Compliance Worker via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("compliance_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Compliance Worker returned %d chars", len(result))
        return {"compliance_result": result}
    except Exception as exc:
        logger.exception("call_compliance failed: %s", exc)
        return {"compliance_result": f"[Compliance analysis unavailable: {exc}]"}


async def synthesize(state: SupervisorState) -> dict:
    """Supervisor synthesizes all worker results into a comprehensive final answer.

    The Supervisor acts as an aggregator — it does NOT add its own legal analysis.
    It combines and organises the specialist worker outputs into a cohesive response.
    """
    llm = get_llm()

    sections: list[str] = []
    if state.get("legal_result"):
        sections.append(f"## Legal Analysis (from Legal Worker)\n{state['legal_result']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis (from Tax Worker)\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Regulatory Compliance Analysis (from Compliance Worker)\n{state['compliance_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "You are a Supervisor Agent synthesising specialist worker analyses into a "
                "comprehensive, well-structured response for the client. You received results "
                "from up to 3 specialist workers: Legal Analysis Worker, Tax Worker, and "
                "Compliance Worker.\n\n"
                "Combine the following analyses into a cohesive answer with clear sections. "
                "Do not add your own legal analysis — only organise and synthesise the worker "
                "outputs. Avoid redundancy. Highlight key risks and recommended actions.\n\n"
                "End with a brief disclaimer that the analysis is educational and the client "
                "should consult licensed attorneys for their specific situation."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the Supervisor Agent StateGraph.

    Topology:
        classify_question → dispatch_workers → (parallel) call_legal + call_tax + call_compliance → synthesize → END
    """
    graph = StateGraph(SupervisorState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("call_legal", call_legal)
    graph.add_node("call_tax", call_tax)
    graph.add_node("call_compliance", call_compliance)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("classify_question")

    # Conditional parallel dispatch: after classify_question, dispatch_workers
    # returns a list of Send objects (to call_legal, call_tax, call_compliance, or synthesize)
    graph.add_conditional_edges(
        "classify_question",
        dispatch_workers,
        ["call_legal", "call_tax", "call_compliance", "synthesize"],
    )

    graph.add_edge("call_legal", "synthesize")
    graph.add_edge("call_tax", "synthesize")
    graph.add_edge("call_compliance", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()