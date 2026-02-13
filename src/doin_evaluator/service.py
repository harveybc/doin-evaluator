"""Evaluator service — pull-based worker that processes tasks from node queues.

The evaluator:
1. Polls any node's /tasks/pending endpoint for available work
2. Claims a task via /tasks/claim
3. Processes it (verification or inference)
4. Reports result via /tasks/complete
5. All events are flooded by the node to the network and logged on-chain

Two task types:
- OPTIMAE_VERIFICATION: Train model with claimed params, independently verify fitness
- INFERENCE_REQUEST: Run inference using current best optimae, return result to client
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

from doin_core.crypto.identity import PeerIdentity
from doin_core.models.domain import Domain, DomainConfig
from doin_core.models.optimae import Optimae
from doin_core.models.task import TaskType
from doin_core.plugins.base import InferencePlugin, SyntheticDataPlugin
from doin_core.plugins.loader import load_inference_plugin, load_synthetic_data_plugin

logger = logging.getLogger(__name__)


@dataclass
class DomainRuntime:
    """Runtime state for a domain on this evaluator."""

    domain: Domain
    inference_plugin: InferencePlugin
    synthetic_data_plugin: SyntheticDataPlugin | None = None
    current_optimae: Optimae | None = None


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator service."""

    host: str = "0.0.0.0"
    port: int = 8471
    node_endpoint: str = "localhost:8470"
    poll_interval: float = 5.0  # Seconds between polling for tasks
    node_endpoints: list[str] = field(default_factory=list)  # Multiple nodes to poll


class EvaluatorService:
    """Pull-based evaluator that processes tasks from the DON work queue.

    Exposes an HTTP API for direct access (health, domains) and
    runs a background loop that pulls tasks from nodes.
    """

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        identity: PeerIdentity | None = None,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self.identity = identity or PeerIdentity.generate()
        self._domains: dict[str, DomainRuntime] = {}
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._session: ClientSession | None = None
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None

        # Stats
        self._requests_served = 0
        self._verifications_done = 0
        self._tasks_processed = 0

        # Node endpoints to poll (primary + extras)
        self._node_endpoints: list[str] = []
        if self.config.node_endpoint:
            self._node_endpoints.append(self.config.node_endpoint)
        self._node_endpoints.extend(self.config.node_endpoints)

        # Routes (for direct HTTP access + health checks)
        self._app.router.add_post("/infer", self._handle_infer)
        self._app.router.add_post("/verify", self._handle_verify)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/domains", self._handle_domains)

    @property
    def peer_id(self) -> str:
        return self.identity.peer_id

    def register_domain(
        self,
        domain: Domain,
        inference_plugin: InferencePlugin | None = None,
        synthetic_data_plugin: SyntheticDataPlugin | None = None,
    ) -> None:
        """Register a domain with its plugins."""
        if inference_plugin is None:
            inf_cls = load_inference_plugin(domain.config.inference_plugin)
            inference_plugin = inf_cls()
            inference_plugin.configure(domain.config.plugin_config)

        if synthetic_data_plugin is None and domain.config.synthetic_data_plugin:
            syn_cls = load_synthetic_data_plugin(domain.config.synthetic_data_plugin)
            synthetic_data_plugin = syn_cls()
            synthetic_data_plugin.configure(domain.config.plugin_config)

        self._domains[domain.id] = DomainRuntime(
            domain=domain,
            inference_plugin=inference_plugin,
            synthetic_data_plugin=synthetic_data_plugin,
        )
        logger.info("Domain registered: %s", domain.id)

    def set_domain_plugins(
        self,
        domain_id: str,
        domain: Domain,
        inference_plugin: InferencePlugin,
        synthetic_data_plugin: SyntheticDataPlugin | None = None,
    ) -> None:
        """Manually set plugins for a domain (for testing)."""
        self._domains[domain_id] = DomainRuntime(
            domain=domain,
            inference_plugin=inference_plugin,
            synthetic_data_plugin=synthetic_data_plugin,
        )

    def update_optimae(self, domain_id: str, optimae: Optimae) -> None:
        """Update the current best optimae for a domain."""
        runtime = self._domains.get(domain_id)
        if runtime:
            runtime.current_optimae = optimae
            logger.info(
                "Updated optimae for %s: performance=%.4f",
                domain_id,
                optimae.reported_performance,
            )

    # ----------------------------------------------------------------
    # Core evaluation logic
    # ----------------------------------------------------------------

    def verify(
        self,
        domain_id: str,
        parameters: dict[str, Any],
        use_synthetic: bool = True,
    ) -> dict[str, Any]:
        """Verify parameters by running inference independently."""
        runtime = self._domains.get(domain_id)
        if runtime is None:
            raise ValueError(f"Unknown domain: {domain_id}")

        data = None
        used_synthetic = False
        if use_synthetic and runtime.synthetic_data_plugin:
            data = runtime.synthetic_data_plugin.generate()
            used_synthetic = True

        performance = runtime.inference_plugin.evaluate(parameters, data)
        self._verifications_done += 1

        return {
            "domain_id": domain_id,
            "verified_performance": performance,
            "used_synthetic_data": used_synthetic,
        }

    def infer(self, domain_id: str, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run inference using current best optimae."""
        runtime = self._domains.get(domain_id)
        if runtime is None:
            raise ValueError(f"Unknown domain: {domain_id}")
        if runtime.current_optimae is None:
            raise ValueError(f"No optimae available for domain: {domain_id}")

        performance = runtime.inference_plugin.evaluate(
            runtime.current_optimae.parameters,
            input_data,
        )
        self._requests_served += 1

        return {
            "domain_id": domain_id,
            "optimae_id": runtime.current_optimae.id,
            "performance": performance,
        }

    # ----------------------------------------------------------------
    # Pull-based task processing loop
    # ----------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Background loop: poll nodes for pending tasks and process them."""
        while self._running:
            try:
                await self._poll_and_process()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in poll loop")

            await asyncio.sleep(self.config.poll_interval)

    async def _poll_and_process(self) -> None:
        """Poll one node for pending tasks and process the first available."""
        if not self._session or not self._node_endpoints:
            return

        domain_ids = list(self._domains.keys())
        if not domain_ids:
            return

        # Try each node endpoint until we find work
        for endpoint in self._node_endpoints:
            try:
                tasks = await self._fetch_pending_tasks(endpoint, domain_ids)
                if not tasks:
                    continue

                # Process the first available task
                task_data = tasks[0]
                task_id = task_data["id"]
                domain_id = task_data["domain_id"]
                task_type = task_data["task_type"]

                # Claim it
                claimed = await self._claim_task(endpoint, task_id)
                if not claimed:
                    continue

                logger.info(
                    "Claimed task %s (%s) for domain %s from %s",
                    task_id[:12], task_type, domain_id, endpoint,
                )

                # Process based on type
                if task_type == TaskType.OPTIMAE_VERIFICATION.value:
                    result = await self._process_verification(task_data)
                elif task_type == TaskType.INFERENCE_REQUEST.value:
                    result = await self._process_inference(task_data)
                else:
                    logger.warning("Unknown task type: %s", task_type)
                    continue

                # Report completion
                await self._complete_task(endpoint, task_id, result)
                self._tasks_processed += 1

                logger.info("Completed task %s → %s", task_id[:12], result.get("status", "ok"))
                return  # Process one task per poll cycle

            except Exception:
                logger.debug("Failed to poll %s", endpoint, exc_info=True)
                continue

    async def _fetch_pending_tasks(
        self, endpoint: str, domain_ids: list[str]
    ) -> list[dict[str, Any]]:
        """GET /tasks/pending from a node."""
        assert self._session is not None
        domains_param = ",".join(domain_ids)
        url = f"http://{endpoint}/tasks/pending?domains={domains_param}&limit=5"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("tasks", [])

    async def _claim_task(self, endpoint: str, task_id: str) -> bool:
        """POST /tasks/claim on a node."""
        assert self._session is not None
        url = f"http://{endpoint}/tasks/claim"
        async with self._session.post(url, json={
            "task_id": task_id,
            "evaluator_id": self.peer_id,
        }) as resp:
            return resp.status == 200

    async def _complete_task(
        self, endpoint: str, task_id: str, result: dict[str, Any]
    ) -> bool:
        """POST /tasks/complete on a node."""
        assert self._session is not None
        url = f"http://{endpoint}/tasks/complete"
        payload = {"task_id": task_id, **result}
        async with self._session.post(url, json=payload) as resp:
            return resp.status == 200

    async def _process_verification(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process an OPTIMAE_VERIFICATION task."""
        domain_id = task_data["domain_id"]
        parameters = task_data["parameters"]

        try:
            result = self.verify(domain_id, parameters, use_synthetic=True)
            return {
                "verified_performance": result["verified_performance"],
                "result": {"used_synthetic": result.get("used_synthetic_data", False)},
                "status": "verified",
            }
        except Exception as e:
            logger.exception("Verification failed for task %s", task_data.get("id", "?")[:12])
            return {
                "result": {"error": str(e)},
                "status": "failed",
            }

    async def _process_inference(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process an INFERENCE_REQUEST task."""
        domain_id = task_data["domain_id"]
        input_data = task_data.get("parameters", {})

        try:
            result = self.infer(domain_id, input_data)
            return {
                "result": result,
                "status": "completed",
            }
        except Exception as e:
            logger.exception("Inference failed for task %s", task_data.get("id", "?")[:12])
            return {
                "result": {"error": str(e)},
                "status": "failed",
            }

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def start(self) -> None:
        """Start the evaluator HTTP service and task polling loop."""
        self._session = ClientSession(timeout=ClientTimeout(total=30))
        self._running = True

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()

        # Start background poll loop
        self._poll_task = asyncio.create_task(self._poll_loop())

        logger.info(
            "Evaluator started: peer=%s, port=%d, domains=%s, polling %s every %.1fs",
            self.peer_id[:12],
            self.config.port,
            list(self._domains.keys()),
            self._node_endpoints,
            self.config.poll_interval,
        )

    async def stop(self) -> None:
        """Stop the evaluator service."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        if self._runner:
            await self._runner.cleanup()
        logger.info(
            "Evaluator stopped (tasks processed: %d, verifications: %d, inferences: %d)",
            self._tasks_processed,
            self._verifications_done,
            self._requests_served,
        )

    # --- HTTP Handlers (for direct access) ---

    async def _handle_infer(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            result = self.infer(data["domain_id"], data.get("input_data", {}))
            return web.json_response(result)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception:
            logger.exception("Inference error")
            return web.json_response({"error": "internal error"}, status=500)

    async def _handle_verify(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            result = self.verify(
                data["domain_id"],
                data["parameters"],
                data.get("use_synthetic", True),
            )
            return web.json_response(result)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception:
            logger.exception("Verification error")
            return web.json_response({"error": "internal error"}, status=500)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "peer_id": self.peer_id[:12],
            "domains": list(self._domains.keys()),
            "requests_served": self._requests_served,
            "verifications_done": self._verifications_done,
            "tasks_processed": self._tasks_processed,
            "polling": self._node_endpoints,
        })

    async def _handle_domains(self, request: web.Request) -> web.Response:
        domains = []
        for did, runtime in self._domains.items():
            domains.append({
                "id": did,
                "name": runtime.domain.name,
                "has_optimae": runtime.current_optimae is not None,
                "has_synthetic_data": runtime.synthetic_data_plugin is not None,
            })
        return web.json_response({"domains": domains})
