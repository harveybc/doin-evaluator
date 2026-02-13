"""CLI entry point for running a DON evaluator."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from pathlib import Path

from doin_core.crypto.identity import PeerIdentity
from doin_core.models.domain import Domain

from doin_evaluator.service import EvaluatorConfig, EvaluatorService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DON evaluator service",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=8471, help="Listen port")
    parser.add_argument("--node", default="localhost:8470", help="Node endpoint")
    parser.add_argument("--domains", required=True, help="Domains JSON config file")
    parser.add_argument("--key-file", default=None, help="PEM private key file")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def run_evaluator(args: argparse.Namespace) -> None:
    identity = None
    if args.key_file:
        identity = PeerIdentity.from_file(args.key_file)

    config = EvaluatorConfig(
        host=args.host,
        port=args.port,
        node_endpoint=args.node,
    )

    service = EvaluatorService(config, identity)

    # Load and register domains
    domains = json.loads(Path(args.domains).read_text())
    for d in domains:
        domain = Domain.model_validate(d)
        service.register_domain(domain)

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    await service.start()
    await shutdown.wait()
    await service.stop()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_evaluator(args))


if __name__ == "__main__":
    main()
