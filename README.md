# doin-evaluator

Standalone evaluator service for DOIN (Decentralized Optimization and Inference Network).

## What It Does

`doin-evaluator` is a pull-based worker that processes verification and inference tasks from DOIN nodes. It:

1. Polls node(s) for pending tasks via `/tasks/pending`
2. Claims a task via `/tasks/claim`
3. Processes it — either verification (test optimae on synthetic data) or inference (run model)
4. Reports results via `/tasks/complete`
5. Exposes its own HTTP API for direct access and health checks

Two task types:
- **OPTIMAE_VERIFICATION**: Independently verify claimed fitness using synthetic data
- **INFERENCE_REQUEST**: Run inference using the current best optimae

## Install

```bash
pip install git+https://github.com/harveybc/doin-core.git
pip install git+https://github.com/harveybc/doin-evaluator.git
```

## Usage

```bash
doin-evaluator --node localhost:8470 --port 8471
```

### Programmatic Usage

```python
from doin_evaluator.service import EvaluatorService, EvaluatorConfig

config = EvaluatorConfig(
    host="0.0.0.0",
    port=8471,
    node_endpoint="localhost:8470",
    poll_interval=5.0,  # seconds between polling
    node_endpoints=["node2:8470"],  # additional nodes to poll
)

service = EvaluatorService(config)
service.register_domain(domain, inference_plugin, synthetic_data_plugin)
await service.start()
```

### EvaluatorConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | str | `0.0.0.0` | Bind address for HTTP API |
| `port` | int | `8471` | HTTP port |
| `node_endpoint` | str | `localhost:8470` | Primary node to poll for tasks |
| `poll_interval` | float | `5.0` | Seconds between task polls |
| `node_endpoints` | list[str] | `[]` | Additional nodes to poll |

## HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (peer ID, domains, stats) |
| `/domains` | GET | List registered domains |
| `/verify` | POST | Direct verification request |
| `/infer` | POST | Direct inference request |

## Architecture

The evaluator runs two concurrent systems:
1. **HTTP server** (aiohttp) — for direct access, health checks, and manual verify/infer requests
2. **Poll loop** — background task that pulls work from node task queues

Each evaluator generates its own `PeerIdentity`. Verification uses per-evaluator deterministic seeds to generate unique synthetic test data, preventing overfitting. Results are reported back to the node, which floods them to the network and records them on-chain.

## Tests

```bash
python -m pytest tests/ -v
# 7 tests passing
```

## Part of DOIN

- [doin-core](https://github.com/harveybc/doin-core) — Consensus, models, crypto
- [doin-node](https://github.com/harveybc/doin-node) — Unified node
- [doin-optimizer](https://github.com/harveybc/doin-optimizer) — Standalone optimizer
- [doin-plugins](https://github.com/harveybc/doin-plugins) — Domain plugins

## License

MIT
