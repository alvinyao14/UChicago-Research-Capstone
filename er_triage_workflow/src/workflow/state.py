"""State definitions for ER Triage Workflow."""

from typing import TypedDict, Optional, Dict
from pydantic import BaseModel


class VitalSigns(BaseModel):
    """Pydantic model for validating vital signs."""
    sex: Optional[str] = None
    age_bucket: Optional[str] = None
    heart_rate: Optional[float] = None
    resp_rate: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    temperature_C: Optional[float] = None
    ESI: Optional[int] = None
    mental_status: Optional[str] = None
    recent_admissions_30d: Optional[int] = None


class ERState(TypedDict, total=False):
    """Defines the state of our graph with optional fields for robustness."""
    visit_id: int
    human_prompt: str
    patient_data: Dict
    vitals_validated: VitalSigns
    triage_text: str
    ml_score: Optional[float]
    llm_score: Optional[float]
    severe: Optional[bool]
    p_final: Optional[float]
    fused_prob: Optional[float]
    decision: Optional[str]
    final_decision: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    fusion_decision: Optional[str]
    fusion_rationale: Optional[str]
    human_override: Optional[float]
    execution_id: Optional[str]  # For traceability
    error_log: Optional[list]  # Track errors during execution


def validate_initial_state(state: dict) -> tuple[bool, Optional[str]]:
    """
    Validate initial state before workflow execution.

    Returns:
        (is_valid, error_message)
    """
    required = ["visit_id"]
    missing = [key for key in required if key not in state]
    if missing:
        return False, f"Missing required initial state keys: {missing}"

    if not isinstance(state.get("visit_id"), int) or state["visit_id"] <= 0:
        return False, f"Invalid visit_id: {state.get('visit_id')}"

    return True, None


def validate_state_transition(state: dict, node_name: str) -> tuple[bool, Optional[str]]:
    """
    Validate state before node execution based on node requirements.

    Returns:
        (is_valid, error_message)
    """
    node_requirements = {
        "fetch_data": ["visit_id"],
        "severity_gate": ["vitals_validated"],
        "ml_model": ["patient_data"],
        "llm_model": ["patient_data"],
        "human_input": ["human_prompt"],
        "fusion": ["ml_score", "llm_score"],
        "confidence_check": ["ml_score", "llm_score", "fused_prob"],
        "human_review": ["fused_prob"],
        "finalize": ["fused_prob"],
    }

    required = node_requirements.get(node_name, [])
    missing = [key for key in required if key not in state or state[key] is None]
    if missing:
        return False, f"Node {node_name} missing required state keys: {missing}"

    return True, None

