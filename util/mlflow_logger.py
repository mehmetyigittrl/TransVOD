"""
Thin MLflow wrapper used by main.py and engine_single.py.

Degrades gracefully: if MLflow is not installed, the tracking server cannot be
reached, or logging is disabled, every call becomes a no-op and training
continues normally.
"""
import os
from typing import Optional

_ENABLED = False
_mlflow = None
_RUN_STARTED = False


def init_mlflow(enabled: bool,
                tracking_uri: str,
                experiment_name: str,
                run_name: Optional[str] = None,
                tags: Optional[dict] = None,
                log_system_metrics: bool = False,
                system_metrics_interval: float = 10.0) -> bool:
    """Start an MLflow run. Returns True iff logging is now active.

    When ``log_system_metrics=True`` the MLflow client samples CPU / RAM / GPU
    util / GPU mem / power every ``system_metrics_interval`` seconds and pushes
    them as ``system/*`` metrics. Requires ``psutil`` and ``pynvml`` (or
    ``nvidia-ml-py``) importable; if not, the MLflow collector thread disables
    itself and training continues.
    """
    global _ENABLED, _mlflow, _RUN_STARTED

    if not enabled:
        return False
    try:
        import mlflow  # type: ignore
    except ImportError:
        print("[mlflow] package not installed; disabling. Install with `pip install mlflow`.")
        return False
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        if log_system_metrics:
            try:
                from mlflow.system_metrics import set_system_metrics_sampling_interval  # type: ignore
                set_system_metrics_sampling_interval(float(system_metrics_interval))
            except Exception as exc:
                print(f"[mlflow] could not set system metrics interval: {exc}")
        start_kwargs = {'run_name': run_name, 'tags': tags or {}}
        if log_system_metrics:
            start_kwargs['log_system_metrics'] = True
        try:
            mlflow.start_run(**start_kwargs)
        except TypeError:
            # older MLflow versions don't accept log_system_metrics on start_run
            start_kwargs.pop('log_system_metrics', None)
            mlflow.start_run(**start_kwargs)
    except Exception as exc:
        print(f"[mlflow] could not start run at {tracking_uri}: {exc}; disabling.")
        return False

    _mlflow = mlflow
    _ENABLED = True
    _RUN_STARTED = True
    try:
        run_id = mlflow.active_run().info.run_id
    except Exception:
        run_id = "?"
    sys_str = f" system_metrics@{system_metrics_interval}s" if log_system_metrics else ""
    print(f"[mlflow] logging to {tracking_uri} "
          f"(experiment={experiment_name}, run={run_name}, run_id={run_id}{sys_str})")
    return True


def is_enabled() -> bool:
    return _ENABLED


def log_params(params: dict) -> None:
    """Log a flat dict of hyper-parameters. Values are stringified and truncated
    to MLflow's 500-char limit so unusual values (paths, lists) don't crash."""
    if not _ENABLED:
        return
    clean = {}
    for k, v in params.items():
        s = "" if v is None else str(v)
        if len(s) > 500:
            s = s[:497] + "..."
        clean[str(k)] = s
    try:
        _mlflow.log_params(clean)
    except Exception as exc:
        print(f"[mlflow] log_params failed: {exc}")


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log numeric metrics at the given global step."""
    if not _ENABLED:
        return
    clean = {}
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f or f in (float('inf'), float('-inf')):
            continue
        clean[str(k)] = f
    if not clean:
        return
    try:
        _mlflow.log_metrics(clean, step=step)
    except Exception as exc:
        print(f"[mlflow] log_metrics failed at step={step}: {exc}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """Upload a single file to the run as an artifact."""
    if not _ENABLED:
        return
    if not local_path or not os.path.exists(local_path):
        return
    try:
        _mlflow.log_artifact(local_path, artifact_path=artifact_path)
    except Exception as exc:
        print(f"[mlflow] log_artifact({local_path}) failed: {exc}")


def log_artifacts(local_dir: str, artifact_path: Optional[str] = None) -> None:
    """Upload every file under local_dir."""
    if not _ENABLED:
        return
    if not local_dir or not os.path.isdir(local_dir):
        return
    try:
        _mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
    except Exception as exc:
        print(f"[mlflow] log_artifacts({local_dir}) failed: {exc}")


def end_run(status: str = "FINISHED") -> None:
    """Close the active run. Safe to call multiple times."""
    global _ENABLED, _RUN_STARTED
    if not _RUN_STARTED:
        return
    try:
        _mlflow.end_run(status=status)
    except Exception as exc:
        print(f"[mlflow] end_run failed: {exc}")
    _ENABLED = False
    _RUN_STARTED = False
