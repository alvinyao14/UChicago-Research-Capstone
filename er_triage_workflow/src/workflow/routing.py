"""Conditional routing functions for ER Triage Workflow."""

from .state import ERState
from ..utils.risk_scoring import calculate_patient_risk_score


def conditional_severity_gate(state: ERState) -> str:
    """
    Conditional Severity Gate Routing
    ----------------------------------
    Routes the workflow based on severity assessment:
    - If severe (critical vitals) → "end" (early exit to END)
    - If not severe → "run_models" (proceed to parallel model execution)
    """
    if state.get("severe", False):
        print("[Routing] Severe case detected → early exit to END")
        return "end"  # Go straight to the end (early exit path)
    else:
        print("[Routing] Non-severe case → proceeding to run_models")
        return "run_models"  # Proceed to parallel branches


def conditional_confidence_routing(state: ERState) -> str:
    """
    Enhanced Conditional Confidence Routing
    ---------------------------------------
    Determines whether the workflow should auto-complete or trigger human review.
    
    NEW: Adjusts thresholds based on patient risk score.
    """
    ml = state.get("ml_score")
    llm = state.get("llm_score")
    patient_data = state.get("patient_data", {})

    # Missing scores → force human review
    if ml is None or llm is None:
        print("[Routing] Missing ML/LLM scores → LOW confidence → human_review")
        return "low_confidence"

    # Calculate patient risk score
    patient_risk = calculate_patient_risk_score(patient_data)
    
    # Adjust thresholds based on patient risk
    if patient_risk > 0.5:  # High-risk patient: more conservative
        HIGH_CONF_GAP = 0.15  # Tighter agreement required
        HIGH_CONF_THRESH = 0.65  # Lower threshold (more conservative)
    else:  # Low-risk patient: can be more lenient
        HIGH_CONF_GAP = 0.25
        HIGH_CONF_THRESH = 0.75

    prob_gap = abs(ml - llm)
    avg_prob = (ml + llm) / 2

    # High confidence path
    if prob_gap < HIGH_CONF_GAP and avg_prob > HIGH_CONF_THRESH:
        print(f"[Routing] HIGH confidence (gap={prob_gap:.2f}, avg={avg_prob:.2f}, risk={patient_risk:.2f}) → finalize")
        return "high_confidence"

    # Low confidence path
    print(f"[Routing] LOW confidence (gap={prob_gap:.2f}, avg={avg_prob:.2f}, risk={patient_risk:.2f}) → human_review")
    return "low_confidence"

