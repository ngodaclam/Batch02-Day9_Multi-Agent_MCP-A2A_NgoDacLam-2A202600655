"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter, or a mock if no API key or MOCK_LLM env is set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    # Use mock if API key missing or mock mode enabled
    if not api_key or os.getenv("MOCK_LLM") == "1":
        from langchain_core.messages import AIMessage
        from uuid import uuid4

        class MockLLM:
            def __init__(self, tools=None):
                self.tools = tools or []

            def bind_tools(self, tools, **kwargs):
                return MockLLM(tools=tools)

            async def ainvoke(self, messages, **kwargs):
                parts = []
                last_user_message = ""
                for m in messages:
                    if hasattr(m, "content"):
                        content = m.content
                    elif isinstance(m, dict) and "content" in m:
                        content = m["content"]
                    else:
                        content = str(m)
                    parts.append(content)
                    
                    if hasattr(m, "type") and m.type == "human":
                        last_user_message = m.content
                    elif isinstance(m, dict) and m.get("type") == "human":
                        last_user_message = m.get("content", "")

                user_content = " ".join(parts)
                
                # If there are tools bound, simulate tool call
                has_delegate = any(getattr(t, "name", "") == "delegate_to_legal_agent" for t in self.tools)
                if has_delegate:
                    tool_call = {
                        "name": "delegate_to_legal_agent",
                        "args": {"question": last_user_message or "legal question"},
                        "id": f"call_{uuid4().hex[:12]}",
                        "type": "tool_call"
                    }
                    return AIMessage(content="", tool_calls=[tool_call])

                # Route check
                if "Reply with ONLY valid JSON" in user_content or "needs_tax" in user_content:
                    return AIMessage(content='{"needs_tax": true, "needs_compliance": true}')

                # Specialist agent responses
                if "tax" in user_content.lower() and "senior tax attorney" in user_content.lower():
                    # Exercise 5.3: Bullet points, <= 3 lines
                    return AIMessage(
                        content="- Evasion of corporate taxes leads to severe civil and criminal penalties under IRS § 7201.\n"
                                "- Company faces back-taxes plus 75% fraud penalty under IRC § 6663.\n"
                                "- Officers directing the evasion are personally liable to criminal prosecution."
                    )
                elif "compliance" in user_content.lower() and "regulatory compliance" in user_content.lower():
                    return AIMessage(
                        content="- SEC regulations breach results in strict civil fines and SOX audit sanctions.\n"
                                "- Personal officer liability applies for signing off on false financial records.\n"
                                "- Investigation may be referred to DOJ for potential criminal prosecution."
                    )
                elif "corporate litigation attorney" in user_content.lower():
                    return AIMessage(
                        content="Breaching the contract triggers liability for compensatory, expectation, and consequential damages."
                    )
                elif "synthesising" in user_content.lower() or "synthesise" in user_content.lower():
                    return AIMessage(
                        content="## Synthesized Legal Analysis\n\n"
                                "### 1. Contract Breach (Law Agent)\n"
                                "Breaching the contract triggers liability for compensatory, expectation, and consequential damages.\n\n"
                                "### 2. Tax Consequences (Tax Agent)\n"
                                "- Evasion of corporate taxes leads to severe civil and criminal penalties under IRS § 7201.\n"
                                "- Company faces back-taxes plus 75% fraud penalty under IRC § 6663.\n"
                                "- Officers directing the evasion are personally liable to criminal prosecution.\n\n"
                                "### 3. Regulatory Compliance (Compliance Agent)\n"
                                "- SEC regulations breach results in strict civil fines and SOX audit sanctions.\n"
                                "- Personal officer liability applies for signing off on false financial records.\n"
                                "- Investigation may be referred to DOJ for potential criminal prosecution.\n\n"
                                "Disclaimer: This analysis is for educational purposes. Consult a licensed attorney for specific cases."
                    )

                return AIMessage(content=f"Mock response for: {last_user_message or user_content}")

        return MockLLM()

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
    )