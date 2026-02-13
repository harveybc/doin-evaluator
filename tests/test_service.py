"""Tests for the evaluator service."""

from typing import Any

import pytest

from doin_core.models.domain import Domain, DomainConfig
from doin_core.models.optimae import Optimae
from doin_core.plugins.base import InferencePlugin, SyntheticDataPlugin
from doin_evaluator.service import EvaluatorService


class MockInferencePlugin(InferencePlugin):
    """Mock inference plugin that returns a fixed performance."""

    def __init__(self, performance: float = 0.85) -> None:
        self._performance = performance

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def evaluate(
        self, parameters: dict[str, Any], data: dict[str, Any] | None = None
    ) -> float:
        # Simulate: performance depends on a param value
        w = parameters.get("w", 1)
        return self._performance + w * 0.01


class MockSyntheticDataPlugin(SyntheticDataPlugin):
    """Mock synthetic data generator."""

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        return {"x": [1, 2, 3], "y": [0.1, 0.2, 0.3], "synthetic": True}


def _make_domain(domain_id: str = "d1") -> Domain:
    return Domain(
        id=domain_id,
        name="Test Domain",
        performance_metric="accuracy",
        higher_is_better=True,
        weight=1.0,
        config=DomainConfig(
            optimization_plugin="test_opt",
            inference_plugin="test_inf",
        ),
    )


def _make_optimae(domain_id: str = "d1") -> Optimae:
    return Optimae(
        domain_id=domain_id,
        optimizer_id="opt-1",
        parameters={"w": 5, "bias": 0.1},
        reported_performance=0.90,
    )


class TestEvaluatorService:
    def _make_service(self) -> EvaluatorService:
        service = EvaluatorService()
        domain = _make_domain()
        service.set_domain_plugins(
            "d1", domain,
            MockInferencePlugin(),
            MockSyntheticDataPlugin(),
        )
        return service

    def test_infer_no_optimae_raises(self) -> None:
        service = self._make_service()
        with pytest.raises(ValueError, match="No optimae"):
            service.infer("d1", {})

    def test_infer_with_optimae(self) -> None:
        service = self._make_service()
        service.update_optimae("d1", _make_optimae())
        result = service.infer("d1", {"x": [1, 2, 3]})
        assert result["domain_id"] == "d1"
        assert "performance" in result
        assert result["performance"] == pytest.approx(0.90, abs=0.1)

    def test_infer_unknown_domain_raises(self) -> None:
        service = self._make_service()
        with pytest.raises(ValueError, match="Unknown domain"):
            service.infer("nonexistent", {})

    def test_verify_with_synthetic(self) -> None:
        service = self._make_service()
        result = service.verify("d1", {"w": 3, "bias": 0.1})
        assert result["used_synthetic_data"] is True
        assert "verified_performance" in result

    def test_verify_without_synthetic(self) -> None:
        service = self._make_service()
        result = service.verify("d1", {"w": 3}, use_synthetic=False)
        assert result["used_synthetic_data"] is False

    def test_verify_no_synthetic_plugin(self) -> None:
        service = EvaluatorService()
        domain = _make_domain("d2")
        service.set_domain_plugins("d2", domain, MockInferencePlugin())
        result = service.verify("d2", {"w": 1})
        assert result["used_synthetic_data"] is False

    def test_update_optimae(self) -> None:
        service = self._make_service()
        optimae = _make_optimae()
        service.update_optimae("d1", optimae)
        # Now inference should work
        result = service.infer("d1", {})
        assert result["optimae_id"] == optimae.id
