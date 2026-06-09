"""Registry client helpers.

Provides `discover(task)` to look up an agent endpoint from the registry,
and `register(agent_info)` for agents to self-register on startup.
"""

import os
import time
from typing import Dict, Tuple

import httpx

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:10000")

# Cache store: task -> (endpoint, timestamp_fetched)
_endpoint_cache: Dict[str, Tuple[str, float]] = {}
CACHE_TTL = 300.0  # 5 minutes in seconds


async def discover(task: str) -> str:
    """Return the endpoint URL of the agent that handles the given task.

    Utilises local caching with a TTL of 5 minutes to reduce network latency.

    Args:
        task: The task identifier (e.g. "legal_question", "tax_question").

    Returns:
        The HTTP endpoint base URL of the matching agent.

    Raises:
        httpx.HTTPStatusError: If no agent is found or the registry is unreachable.
    """
    now = time.time()
    
    # Check cache validity
    if task in _endpoint_cache:
        endpoint, fetched_at = _endpoint_cache[task]
        if now - fetched_at < CACHE_TTL:
            return endpoint

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{REGISTRY_URL}/discover/{task}")
        resp.raise_for_status()
        endpoint = resp.json()["endpoint"]
        
        # Save to cache
        _endpoint_cache[task] = (endpoint, now)
        return endpoint


async def register(agent_info: dict) -> None:
    """Register an agent with the registry.

    Args:
        agent_info: Dict with keys: agent_name, version, description,
                    tasks, endpoint, tags.

    Raises:
        httpx.HTTPStatusError: If registration fails.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{REGISTRY_URL}/register", json=agent_info)
        resp.raise_for_status()