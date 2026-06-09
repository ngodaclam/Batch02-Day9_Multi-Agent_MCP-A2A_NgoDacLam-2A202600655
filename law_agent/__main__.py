"""Supervisor Agent server entry point — port 10101.

Implements the Supervisor-Workers pattern. This agent:
1. Classifies incoming legal questions
2. Dispatches specialist workers (Legal, Tax, Compliance) in parallel
3. Synthesizes worker results into a comprehensive response

Keeps port 10101 and task 'legal_question' for backward compatibility
with the Customer Agent.
"""

from __future__ import annotations

import asyncio
import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from common.registry_client import register
from law_agent.agent_executor import SupervisorAgentExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [supervisor_agent] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PORT = 10101
AGENT_ENDPOINT = f"http://localhost:{PORT}"


async def _register_with_retry(max_attempts: int = 10, delay: float = 2.0) -> None:
    """Retry registration until the registry is up."""
    info = {
        "agent_name": "supervisor-agent",
        "version": "2.0",
        "description": (
            "Supervisor Agent: orchestrates Legal, Tax, and Compliance workers "
            "using the Supervisor-Workers pattern for comprehensive legal analysis"
        ),
        "tasks": ["legal_question"],
        "endpoint": AGENT_ENDPOINT,
        "tags": ["supervisor", "orchestrator", "legal", "multi-agent"],
    }
    for attempt in range(1, max_attempts + 1):
        try:
            await register(info)
            logger.info("Registered with registry (attempt %d)", attempt)
            return
        except Exception as exc:
            logger.warning(
                "Registry not ready (attempt %d/%d): %s — retrying in %.0fs",
                attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)
    logger.error("Failed to register after %d attempts", max_attempts)


async def main() -> None:
    await _register_with_retry()

    agent_card = AgentCard(
        name="Supervisor Agent",
        description=(
            "Supervisor Agent implementing the Supervisor-Workers pattern. "
            "Classifies legal questions and dispatches specialist workers "
            "(Legal Analysis, Tax, Compliance) in parallel, then synthesizes "
            "a comprehensive response from all worker results."
        ),
        url=AGENT_ENDPOINT,
        version="2.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="legal_question",
                name="Legal Question Orchestration",
                description=(
                    "Orchestrate comprehensive legal analysis by dispatching "
                    "specialist workers (Legal, Tax, Compliance) in parallel "
                    "and synthesizing their results."
                ),
                tags=["supervisor", "orchestrator", "legal", "multi-agent"],
            )
        ],
    )

    executor = SupervisorAgentExecutor()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    app_builder = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = app_builder.build()

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    logger.info("Supervisor Agent listening on port %d", PORT)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())