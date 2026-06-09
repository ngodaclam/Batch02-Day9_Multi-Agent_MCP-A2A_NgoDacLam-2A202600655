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
        from langchain_core.language_models.chat_models import SimpleChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatResult, ChatGeneration
        from typing import Any, List, Optional
        from uuid import uuid4

        class MockChatModel(SimpleChatModel):
            bound_tools: List[Any] = []

            @property
            def _llm_type(self) -> str:
                return "mock-chat-model"

            def bind_tools(self, tools: List[Any], **kwargs: Any) -> Any:
                # Store tools and return a bound instance
                return MockChatModel(bound_tools=tools)

            def _call(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[Any] = None,
                **kwargs: Any,
            ) -> str:
                return "Mock response"

            async def _agenerate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[Any] = None,
                **kwargs: Any,
            ) -> ChatResult:
                parts = []
                last_user_message = ""
                has_tool_response = False
                last_tool_content = ""

                for m in messages:
                    # Check if there is already a tool response in history to break loop
                    is_tool = (hasattr(m, "type") and m.type == "tool") or (isinstance(m, dict) and m.get("type") == "tool")
                    if is_tool:
                        has_tool_response = True
                        if hasattr(m, "content"):
                            last_tool_content = m.content
                        elif isinstance(m, dict):
                            last_tool_content = m.get("content", "")

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
                
                # Break infinite loop if tool already executed
                if has_tool_response:
                    message = AIMessage(content=last_tool_content or "Here is the final synthesized legal analysis.")
                    return ChatResult(generations=[ChatGeneration(message=message)])

                # Check for tool call trigger
                has_delegate = any(getattr(t, "name", "") == "delegate_to_legal_agent" for t in self.bound_tools)
                if has_delegate:
                    tool_call = {
                        "name": "delegate_to_legal_agent",
                        "args": {"question": last_user_message or "legal question"},
                        "id": f"call_{uuid4().hex[:12]}",
                        "type": "tool_call"
                    }
                    message = AIMessage(content="", tool_calls=[tool_call])
                    return ChatResult(generations=[ChatGeneration(message=message)])

                # Specialist responses - Check aggregate/synthesise first to avoid false matches
                if "synthesising" in user_content.lower() or "synthesise" in user_content.lower():
                    message = AIMessage(
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
                elif "Reply with ONLY valid JSON" in user_content or "needs_tax" in user_content:
                    message = AIMessage(content='{"needs_tax": true, "needs_compliance": true}')
                    return ChatResult(generations=[ChatGeneration(message=message)])
                elif "tax" in user_content.lower() and "senior tax attorney" in user_content.lower():
                    message = AIMessage(
                        content="- Evasion of corporate taxes leads to severe civil and criminal penalties under IRS § 7201.\n"
                                "- Company faces back-taxes plus 75% fraud penalty under IRC § 6663.\n"
                                "- Officers directing the evasion are personally liable to criminal prosecution."
                    )
                elif "compliance" in user_content.lower() and "regulatory compliance" in user_content.lower():
                    message = AIMessage(
                        content="- SEC regulations breach results in strict civil fines and SOX audit sanctions.\n"
                                "- Personal officer liability applies for signing off on false financial records.\n"
                                "- Investigation may be referred to DOJ for potential criminal prosecution."
                    )
                elif "corporate litigation attorney" in user_content.lower():
                    message = AIMessage(
                        content="Breaching the contract triggers liability for compensatory, expectation, and consequential damages."
                    )
                else:
                    message = AIMessage(content=f"Mock response for: {last_user_message or user_content}")

                return ChatResult(generations=[ChatGeneration(message=message)])

        return MockChatModel()

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
    )