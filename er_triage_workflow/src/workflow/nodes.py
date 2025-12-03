"""Workflow node definitions for ER Triage Workflow."""

from typing import Dict
from .state import ERState, VitalSigns
from ..database.queries import fetch_patient_data
from ..models.ml_model import ml_predict_proba
from ..models.llm_model import llm_predict_proba, format_for_llm_classifier
from ..models.fusion_agent import run_fusion_agent
from ..utils.logging import log_error, get_execution_id, init_execution_context, log_event
from ..utils.risk_scoring import calculate_patient_risk_score
from config import get_config


# Global admission threshold (can be updated after evaluation)
ADMISSION_THRESHOLD = 0.5


def fetch_data_node(state: ERState) -> Dict:
    """
    Enhanced Fetch Data Node with Patient History
    ----------------------------------------------
    Takes a visit_id, connects to the DB, and fetches the patient's
    de-identified data from Visit_Details, Triage_Notes, and ESI.
    
    NEW: Also fetches patient history (previous visits, admission patterns).
    """
    visit_id = state.get('visit_id')
    execution_id = get_execution_id()

    # Validate input
    if not visit_id or not isinstance(visit_id, int) or visit_id <= 0:
        raise ValueError(f"Invalid visit_id: {visit_id}")

    print(f"--- 1. Fetching data for visit_id: {visit_id} (execution_id: {execution_id}) ---")

    # Initialize execution context if not already set
    if execution_id is None:
        init_execution_context(visit_id)
        execution_id = get_execution_id()

    # Fetch patient data with history
    result = fetch_patient_data(visit_id)
    
    # Update state with execution ID for traceability
    result["execution_id"] = execution_id

    return result


def severity_gate_node(state: ERState) -> Dict:
    """
    Enhanced Severity Gate Node
    ---------------------------
    Checks for critical vital signs, ESI level, and high-risk patterns.
    
    NEW: Incorporates ESI level, readmission risk, and age-based risk factors.
    """
    print("--- 2. Checking severity gate ---")
    v = state["vitals_validated"]
    patient_data = state.get("patient_data", {})
    
    # Critical vital signs (existing)
    critical_vitals = (
        (v.oxygen_saturation is not None and v.oxygen_saturation < 88) or
        (v.bp_systolic is not None and v.bp_systolic < 80) or
        (v.resp_rate is not None and (v.resp_rate > 35 or v.resp_rate < 8))
    )
    
    # ESI-based severity (ESI 1-2 are critical per triage standards)
    esi = patient_data.get('ESI')
    critical_esi = esi is not None and esi <= 2
    
    # High readmission risk (2+ admissions in 30 days)
    recent_admissions = patient_data.get('recent_admissions_30d', 0)
    high_readmission = recent_admissions >= 2
    
    # Elderly with concerning vitals
    age_bucket = patient_data.get('age_bucket', '')
    elderly_risk = age_bucket == '65+' and (
        (v.heart_rate is not None and v.heart_rate > 100) or 
        (v.temperature_C is not None and v.temperature_C > 38.5) or 
        (v.oxygen_saturation is not None and v.oxygen_saturation < 92)
    )
    
    # Combined severity assessment
    is_severe = critical_vitals or critical_esi or (high_readmission and elderly_risk)
    
    if is_severe:
        severity_reasons = []
        if critical_vitals: severity_reasons.append("Critical vitals")
        if critical_esi: severity_reasons.append(f"ESI level {esi}")
        if high_readmission: severity_reasons.append(f"{recent_admissions} recent admissions")
        if elderly_risk: severity_reasons.append("Elderly with concerning vitals")
        
        print(f" -> CRITICAL: Patient is severe. Reasons: {', '.join(severity_reasons)}")
        return {
            "severe": True,
            "decision": "Admit",
            "p_final": 1.0,
            "rationale": f"Severe case: {', '.join(severity_reasons)}. Immediate admission required.",
            "severity_factors": severity_reasons
        }

    print(" -> OK: Patient is not severe. Proceeding to models.")
    return {"severe": False}


def ml_model_node(state: ERState) -> Dict:
    """
    ML Model Node
    -------------
    Runs the patient data through the retrained ML pipeline.

    Enhanced with:
    - Input validation
    - Error handling with fallback
    - Score validation (0-1 range)
    """
    print("--- 3a. Running ML Model ---")

    # Validate input
    patient_data = state.get('patient_data')
    if not patient_data:
        raise ValueError("Missing patient_data for ML model")

    try:
        score = ml_predict_proba(patient_data)

        # Validate score is in valid range
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            print(f"[WARNING] ML model returned invalid score: {score}, clamping to [0, 1]")
            score = max(0.0, min(1.0, float(score)))

        print(f" -> ML Score (P_Admit): {score:.4f}")
        return {"ml_score": float(score)}

    except Exception as e:
        log_error("ml_model", e, state, get_execution_id())
        # Return neutral score on error
        print(f"[WARNING] ML model failed, using neutral score: {e}")
        return {"ml_score": 0.5}


def llm_model_node(state: ERState) -> Dict:
    """
    LLM Model Node
    --------------
    Runs the patient data through the retrained LLM pipeline.

    Enhanced with:
    - Input validation
    - Error handling with fallback
    - Score validation (0-1 range)
    """
    print("--- 3b. Running LLM Classifier ---")

    # Validate input
    patient_data = state.get('patient_data')
    if not patient_data:
        raise ValueError("Missing patient_data for LLM model")

    try:
        formatted_text = format_for_llm_classifier(patient_data)

        # Validate formatted text is not empty
        if not formatted_text or not formatted_text.strip():
            print("[WARNING] Empty formatted text for LLM, using default")
            formatted_text = "No patient data available."

        score = llm_predict_proba(formatted_text)

        # Validate score is in valid range
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            print(f"[WARNING] LLM model returned invalid score: {score}, clamping to [0, 1]")
            score = max(0.0, min(1.0, float(score)))

        print(f" -> LLM Classifier Score (P_Admit): {score:.4f}")
        return {"llm_score": float(score)}

    except Exception as e:
        log_error("llm_model", e, state, get_execution_id())
        # Return neutral score on error
        print(f"[WARNING] LLM model failed, using neutral score: {e}")
        return {"llm_score": 0.5}


def human_input_node(state: ERState) -> Dict:
    """
    Human Input Node
    ---------------
    This node acknowledges and processes the human prompt/note.
    """
    print("--- 3c. Acknowledging Human Input ---")
    print(f" -> Human Note: '{state['human_prompt']}'")
    return {}


def fusion_node(state: ERState) -> Dict:
    """
    Enhanced Fusion Node with Patient Context
    -----------------------------------------
    Fuses the outputs from human_input, llm_model, and ml_model.
    
    NEW: Incorporates patient history context into fusion agent prompt.
    """
    print("--- 4. Fusing Inputs with LLM Agent (combining human_input, llm_model, ml_model) ---")

    # Validate and extract inputs with defaults
    ml_prob = state.get("ml_score")
    llm_prob = state.get("llm_score")
    human_note = (state.get("human_prompt") or "").strip()
    patient_data = state.get("patient_data", {})

    # Validate scores exist and are valid
    if ml_prob is None:
        print("[WARNING] ml_score missing, using default 0.5")
        ml_prob = 0.5
    if llm_prob is None:
        print("[WARNING] llm_score missing, using default 0.5")
        llm_prob = 0.5

    # Ensure scores are in valid range
    ml_prob = max(0.0, min(1.0, float(ml_prob)))
    llm_prob = max(0.0, min(1.0, float(llm_prob)))

    execution_id = get_execution_id()

    # Build patient context for fusion agent
    context_parts = []
    recent_admissions = patient_data.get('recent_admissions_30d', 0)
    if recent_admissions > 0:
        context_parts.append(f"Patient has {recent_admissions} recent admission(s) in past 30 days")
    
    hist_visits = patient_data.get('historical_visit_count', 0)
    hist_admissions = patient_data.get('historical_admission_count', 0)
    if hist_visits > 0:
        admission_rate = hist_admissions / hist_visits
        context_parts.append(f"Historical admission rate: {admission_rate:.1%} ({hist_admissions}/{hist_visits} visits)")
    
    esi = patient_data.get('ESI', 3)
    if esi <= 2:
        context_parts.append(f"High acuity triage (ESI {esi})")
    
    context_str = ". ".join(context_parts) if context_parts else "No significant historical patterns."

    # 1) Call fusion agent with error handling
    fusion_output = None
    fusion_error = None
    try:
        # Build context-enhanced human note
        enhanced_human_note = human_note
        if context_parts:
            enhanced_human_note = f"{human_note} [Context: {context_str}]"
        
        # Use the run_fusion_agent function (which handles JSON parsing internally)
        fusion_output = run_fusion_agent(
            ml_prob=ml_prob,
            llm_prob=llm_prob,
            human_note=enhanced_human_note,
            max_retries=2
        )

        # Validate fusion output (more lenient)
        if not isinstance(fusion_output, dict):
            raise ValueError("Fusion agent returned non-dict output")
        
        # Ensure we have at least a decision
        if "decision" not in fusion_output or fusion_output.get("decision") == "Error":
            # Infer decision from probabilities if missing or error
            if ml_prob > 0.7 or llm_prob > 0.7:
                fusion_output["decision"] = "Admit"
            else:
                fusion_output["decision"] = "Discharge"
            fusion_output["rationale"] = fusion_output.get("rationale", 
                f"Inferred decision from probabilities (ML: {ml_prob:.2f}, LLM: {llm_prob:.2f})")

    except Exception as e:
        fusion_error = str(e)
        log_error("fusion_agent", e, state, execution_id)
        print(f" -> Fusion agent raised an exception: {e}")
        fusion_output = {
            "decision": "Error",
            "rationale": f"Exception during fusion agent call: {e}. Using weighted average.",
        }

    fusion_decision = fusion_output.get("decision", "Error") if fusion_output else "Error"
    fusion_rationale = fusion_output.get(
        "rationale",
        "No rationale returned by fusion agent. Using weighted average of ML and LLM scores."
    ) if fusion_output else "Fusion agent failed. Using weighted average."

    # 2) Numeric fused probability (always compute as fallback)
    fused_prob = 0.5 * ml_prob + 0.5 * llm_prob

    # If fusion agent failed, use weighted average as decision
    if fusion_error or fusion_decision == "Error":
        if fused_prob >= 0.5:
            fusion_decision = "Admit"
        else:
            fusion_decision = "Discharge"
        fusion_rationale = f"Fusion agent unavailable. Using weighted average (0.5*ML + 0.5*LLM) = {fused_prob:.3f}. Decision: {fusion_decision}."

    print(
        f" -> Final P(Admit) Score (numeric): {fused_prob:.4f} | "
        f"Fusion Agent Decision: {fusion_decision}"
    )
    print(f" -> Fusion Agent Rationale (inside fusion_node): {fusion_rationale}")

    # 3) Return all fields
    return {
        "fused_prob": float(fused_prob),
        "p_final": float(fused_prob),
        "fusion_decision": fusion_decision,
        "fusion_rationale": fusion_rationale,
    }


def confidence_check_node(state: ERState) -> Dict:
    """
    Confidence Check Node
    ---------------------
    This node is executed after the fusion step.

    Purpose:
    - Validate that ML and LLM scores exist in the state.
    - Print debugging information for verification.
    - Pass-through node (no state modification).

    NOTE:
    The actual routing decision (high vs. low confidence)
    is performed by `conditional_confidence_routing`.
    """
    ml = state.get("ml_score")
    llm = state.get("llm_score")
    fused_prob = state.get("fused_prob", state.get("p_final"))
    print("--- 5. Confidence Check Node ---")
    print(f"[ConfidenceCheck] ml_score={ml:.4f}, llm_score={llm:.4f}, fused_prob={fused_prob:.4f}")
    return state


def human_review_node(state: ERState) -> Dict:
    """
    Human Review Node (HITL - Human In The Loop)
    ---------------------------------------------
    This node is executed when confidence_check routes to low_confidence.

    Behavior:
    - If human_override exists → override fused_prob.
    - If not → use the original fused_prob.
    """
    print("--- 6. Human Review Node (HITL) ---")
    override = state.get("human_override", None)

    if override is not None:
        output = {"fused_prob": float(override), "p_final": float(override)}
        log_event("human_review_override", dict(state), output)
        print(f"[HITL] Human override applied → fused_prob={override:.4f}")
        return output

    # No override → retain original fused_prob
    fused_prob = state.get("fused_prob", state.get("p_final"))
    output = {"fused_prob": float(fused_prob), "p_final": float(fused_prob)}
    log_event("human_review_no_override", dict(state), output)
    print(f"[HITL] No override provided → using original fused_prob={fused_prob:.4f}")
    return output


def finalize_node(state: ERState) -> Dict:
    """
    Finalize Node
    -------------
    This node produces the final admission decision and rationale.

    Decision rule:
      - Uses fusion agent decision if available (takes precedence)
      - Otherwise: ADMIT if fused_prob >= ADMISSION_THRESHOLD
      - DISCHARGE if fused_prob < ADMISSION_THRESHOLD
    """
    print("--- 7. Finalize Node ---")

    # Prefer fused_prob, but fall back to p_final if needed
    fused = state.get("fused_prob", state.get("p_final", None))
    
    # Check if fusion agent made a decision (this takes precedence over numeric threshold)
    fusion_decision = state.get("fusion_decision")
    fusion_rationale = state.get("fusion_rationale", "")

    if fused is None:
        decision = "UNKNOWN"
        rationale = (
            "Missing fused_prob; unable to generate a final decision. "
            "Check fusion or human review steps."
        )
    elif fusion_decision and fusion_decision != "Error":
        # Fusion agent made a decision - use it (respects clinical reasoning)
        if fusion_decision.lower() in ["admit", "admission"]:
            decision = "ADMIT"
        elif fusion_decision.lower() in ["discharge", "discharged"]:
            decision = "DISCHARGE"
        else:
            # Fallback to threshold if fusion decision is unclear
            print(f"[Finalize] Unclear fusion decision '{fusion_decision}', using threshold")
            if fused >= ADMISSION_THRESHOLD:
                decision = "ADMIT"
            else:
                decision = "DISCHARGE"
        
        # Use fusion agent's rationale if available, otherwise create one
        if fusion_rationale and fusion_rationale.strip():
            rationale = f"Fusion agent decision: {fusion_decision}. {fusion_rationale}"
        else:
            rationale = (
                f"Fusion agent decision: {fusion_decision}. "
                f"Fused probability: {fused:.2f}."
            )
        
        print(f"[Finalize] Using fusion agent decision: {decision}")
    else:
        # No fusion agent decision or it failed - use numeric threshold
        print(f"[Finalize] Using Admission Threshold: {ADMISSION_THRESHOLD:.4f}")

        if fused >= ADMISSION_THRESHOLD:
            decision = "ADMIT"
            rationale = (
                f"Fused probability {fused:.2f} ≥ threshold {ADMISSION_THRESHOLD:.2f}; "
                "patient should be admitted."
            )
        else:
            decision = "DISCHARGE"
            rationale = (
                f"Fused probability {fused:.2f} < threshold {ADMISSION_THRESHOLD:.2f}; "
                "patient may be safely discharged."
            )

    # Merge decision fields back into the *existing* state
    new_state = dict(state)
    new_state["final_decision"] = decision
    new_state["rationale"] = rationale

    # Backward-compatible aliases
    new_state["decision"] = decision
    new_state["p_final"] = fused

    log_event("finalize_node", dict(state), new_state)
    print(f"[Finalize] Decision={decision}, Rationale={rationale}")

    return new_state


def run_models_node(state: ERState) -> Dict:
    """
    Run Models Router Node
    ----------------------
    This is a router node that triggers the parallel model runs.
    """
    print("--- 3. Fanning out to parallel models (human_input, llm_model, ml_model) ---")
    return {}

