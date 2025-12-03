"""Logging utilities for ER Admission Agentic AI."""

import json
import os
import time
import uuid
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import copy

from pydantic import BaseModel

from ...config import get_config


# Global execution context for traceability
_execution_context = {
    "execution_id": None,
    "visit_id": None,
    "start_time": None,
}


def init_execution_context(visit_id: int) -> str:
    """Initialize execution context for a new workflow run."""
    _execution_context["execution_id"] = str(uuid.uuid4())
    _execution_context["visit_id"] = visit_id
    _execution_context["start_time"] = time.time()
    return _execution_context["execution_id"]


def get_execution_id() -> Optional[str]:
    """Get current execution ID for traceability."""
    return _execution_context.get("execution_id")


def _make_json_safe(obj):
    """
    Recursively convert objects into JSON-serializable forms.

    Rules:
    - dict     → clean keys and values
    - list/tuple → clean each element
    - basic types (int, float, str, bool, None) → keep as-is
    - anything else (custom classes like VitalSigns) → convert to str(obj)
    """
    # Basic scalar types that JSON can handle directly
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    # dict: clean each key/value
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    # list or tuple: clean each element
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]

    # Fallback: custom object, dataclass, pydantic model, etc.
    # We serialize it via its string representation.
    return str(obj)


def log_event(
    step: str,
    input_state: dict,
    output_state: dict,
    meta: dict = None,
    execution_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None
):
    """
    Enhanced logging utility with execution tracking and performance metrics.

    Args:
        step: Name of the step/node
        input_state: Input state dictionary
        output_state: Output state dictionary
        meta: Additional metadata
        execution_id: Unique execution ID for traceability
        duration_ms: Execution duration in milliseconds
        error: Error message if any
    """
    execution_id = execution_id or get_execution_id()
    cfg = get_config()

    record = {
        "timestamp": datetime.now().isoformat(),
        "execution_id": execution_id,
        "step": step,
        "input_state": _make_json_safe(input_state),
        "output_state": _make_json_safe(output_state),
        "meta": _make_json_safe(meta or {}),
    }

    # Add performance metrics
    if duration_ms is not None:
        record["duration_ms"] = duration_ms
        record["meta"]["performance"] = {"duration_ms": duration_ms}

    # Add error information if present
    if error:
        record["error"] = error
        record["meta"]["error"] = error

    # Append one line per event
    try:
        with open(cfg.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[WARNING] Failed to write to log file: {e}")

    # Optional console debug
    log_msg = f"[LOG] execution_id={execution_id} step={step}"
    if duration_ms is not None:
        log_msg += f" duration={duration_ms:.2f}ms"
    if error:
        log_msg += f" ERROR={error[:100]}"
    print(log_msg)


def log_error(step: str, error: Exception, state: dict = None, execution_id: Optional[str] = None):
    """
    Log errors with full context for debugging.

    Args:
        step: Name of the step where error occurred
        error: Exception object
        state: State at time of error
        execution_id: Unique execution ID
    """
    execution_id = execution_id or get_execution_id()
    cfg = get_config()

    error_record = {
        "timestamp": datetime.now().isoformat(),
        "execution_id": execution_id,
        "step": step,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "state": _make_json_safe(state or {}),
    }

    try:
        with open(cfg.error_log_path, "a") as f:
            f.write(json.dumps(error_record) + "\n")
    except Exception as e:
        print(f"[WARNING] Failed to write to error log file: {e}")

    print(f"[ERROR] execution_id={execution_id} step={step} error={type(error).__name__}: {str(error)}")


@contextmanager
def track_performance(step_name: str):
    """Context manager to track execution time of a step."""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        print(f"[PERF] {step_name} took {duration_ms:.2f}ms")


def _get_safe_default_output(node_name: str, state: dict) -> dict:
    """
    Return safe default outputs for nodes that fail.
    This allows the workflow to continue with degraded functionality.
    """
    defaults = {
        "ml_model": {"ml_score": 0.5},  # Neutral probability
        "llm_model": {"llm_score": 0.5},  # Neutral probability
        "human_input": {},  # No output needed
        "fusion": {
            "fused_prob": 0.5,
            "p_final": 0.5,
            "fusion_decision": "Error",
            "fusion_rationale": "Fusion failed, using default neutral probability."
        },
        "confidence_check": state,  # Pass through
        "human_review": {
            "fused_prob": state.get("fused_prob", state.get("p_final", 0.5)),
            "p_final": state.get("fused_prob", state.get("p_final", 0.5))
        },
        "finalize": {
            "decision": "UNKNOWN",
            "rationale": f"Workflow encountered errors. Node {node_name} failed.",
            "p_final": state.get("fused_prob", state.get("p_final", 0.5))
        }
    }
    return defaults.get(node_name, {})


def make_logged_node(fn, name: str, max_retries: int = 0, retry_delay: float = 0.1):
    """
    Enhanced node wrapper with:
    - Comprehensive logging with execution IDs
    - Performance tracking
    - Error handling with graceful degradation
    - Retry logic for transient failures
    - State validation

    Args:
        fn: Node function to wrap
        name: Name of the node for logging
        max_retries: Maximum number of retries for transient failures (0 = no retries)
        retry_delay: Delay between retries in seconds
    """
    def _to_serializable_dict(obj):
        """Recursively converts Pydantic BaseModel instances to dictionaries within a dict/list."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: _to_serializable_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_serializable_dict(elem) for elem in obj]
        return obj

    @wraps(fn)
    def wrapped(state):
        execution_id = get_execution_id()
        start_time = time.time()

        # Create a deep copy of the state and convert Pydantic objects for logging input
        serializable_input_state = _to_serializable_dict(copy.deepcopy(state))

        # Log input
        log_event(
            f"{name}_INPUT",
            serializable_input_state,
            {},
            execution_id=execution_id
        )

        # Retry logic for transient failures
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Run original node
                with track_performance(f"{name}_node"):
                    out = fn(state)

                # Validate output
                if out is None:
                    out = {}  # Ensure we always return a dict

                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Create a deep copy of the output and convert Pydantic objects for logging output
                serializable_current_state_for_output_log = _to_serializable_dict(copy.deepcopy(state))
                serializable_output_from_node = _to_serializable_dict(copy.deepcopy(out))

                # Log output with performance metrics
                log_event(
                    f"{name}_OUTPUT",
                    serializable_current_state_for_output_log,
                    serializable_output_from_node,
                    execution_id=execution_id,
                    duration_ms=duration_ms
                )

                return out

            except Exception as e:
                last_error = e
                duration_ms = (time.time() - start_time) * 1000

                # Log error
                log_error(name, e, state, execution_id)

                # If this is the last attempt, handle the error
                if attempt == max_retries:
                    # Log failed attempt
                    log_event(
                        f"{name}_ERROR",
                        serializable_input_state,
                        {},
                        execution_id=execution_id,
                        duration_ms=duration_ms,
                        error=f"{type(e).__name__}: {str(e)}"
                    )

                    # For critical nodes, we might want to raise
                    # For non-critical nodes, return a safe default
                    if name in ["fetch_data", "severity_gate"]:
                        # Critical nodes - re-raise
                        raise
                    else:
                        # Non-critical nodes - return safe defaults
                        print(f"[WARNING] {name} failed after {max_retries + 1} attempts, using safe defaults")
                        return _get_safe_default_output(name, state)
                else:
                    # Wait before retry
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    print(f"[RETRY] {name} attempt {attempt + 1}/{max_retries + 1}")

        # Should never reach here, but just in case
        return _get_safe_default_output(name, state)

    return wrapped

