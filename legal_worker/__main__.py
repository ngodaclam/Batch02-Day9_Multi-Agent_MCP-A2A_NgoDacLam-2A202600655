"""Legal Analysis Worker server entry point — port 10104."""

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
from legal_worker.agent_executor import LegalWorkerExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [legal_worker] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PORT = 10104
AGENT_ENDPOINT = f"http://localhost:{PORT}"


async def _register_with_retry(max_attempts: int = 10, delay: float = 2.0) -> None:
    """Retry registration until the registry is up."""
    info = {
        "agent_name": "legal-worker",
        "version": "1.0",
        "description": "Specialist contract law and civil litigation worker agent",
        "tasks": ["legal_analysis_question"],
        "endpoint": AGENT_ENDPOINT,
        "tags": ["legal", "contract", "litigation", "tort", "worker"],
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
        name="Legal Analysis Worker",
        description=(
            "Specialist contract law and civil litigation worker. Analyses legal "
            "questions covering contract breach, tort liability, corporate law, "
            "fiduciary duties, and intellectual property disputes."
        ),
        url=AGENT_ENDPOINT,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="legal_analysis_question",
                name="Legal Analysis",
                description=(
                    "Provide detailed legal analysis of contract law, tort law, "
                    "corporate liability, and related legal questions."
                ),
                tags=["legal", "contract", "litigation", "worker"],
            )
        ],
    )

    executor = LegalWorkerExecutor()
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
    logger.info("Legal Analysis Worker listening on port %d", PORT)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
