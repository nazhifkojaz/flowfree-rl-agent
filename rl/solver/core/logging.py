from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class RunLogger:
    """Minimal logging interface used by training routines."""

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        raise NotImplementedError

    def log_artifact(self, path: Path) -> None:
        raise NotImplementedError

    def set_tags(self, tags: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class NullRunLogger(RunLogger):
    """No-op logger used when MLflow (or similar) is disabled."""

    def log_params(self, params: Dict[str, Any]) -> None:
        return None

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        return None

    def log_artifact(self, path: Path) -> None:
        return None

    def set_tags(self, tags: Dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class MLflowRunLogger(RunLogger):
    """MLflow-backed implementation of RunLogger."""

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: Dict[str, Any] | None = None,
    ) -> None:
        try:
            import mlflow
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "MLflow is not installed. Install the RL extras via `poetry install -E rl`."
            ) from exc

        self._mlflow = mlflow
        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            self._mlflow.set_experiment(experiment_name)

        if self._mlflow.active_run() is not None:
            self._mlflow.end_run()
        self._run = self._mlflow.start_run(run_name=run_name)
        if tags:
            self.set_tags(tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        serialisable: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, Path):
                serialisable[key] = str(value)
            else:
                serialisable[key] = value
        self._mlflow.log_params(serialisable)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, path: Path) -> None:
        resolved = Path(path)
        if resolved.exists():
            self._mlflow.log_artifact(str(resolved))

    def set_tags(self, tags: Dict[str, Any]) -> None:
        serialisable = {k: (str(v) if isinstance(v, Path) else v) for k, v in tags.items()}
        self._mlflow.set_tags(serialisable)

    def close(self) -> None:
        if self._mlflow.active_run() is not None:
            self._mlflow.end_run()


__all__ = ["RunLogger", "NullRunLogger", "MLflowRunLogger"]
