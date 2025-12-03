"""
Enhanced Implementation Functions
==================================
Copy these enhanced functions into your notebook to replace the existing ones.

1. Enhanced run_fusion_agent (with robust JSON parsing)
2. Enhanced fetch_data_node (with patient history)
3. Enhanced severity_gate_node (with ESI and risk factors)
4. Enhanced conditional_confidence_routing (with risk-based thresholds)
"""

# ============================================================================
# 1. ENHANCED run_fusion_agent - FIXES JSON PARSING ISSUE
# ============================================================================

def run_fusion_agent(ml_prob: float, llm_prob: float, human_note: str, max_retries: int = 2) -> dict:
    """
    Enhanced fusion agent with robust JSON parsing and retry logic.
    
    Uses the generative LLM to synthesize inputs and make a final decision with rationale.
    Includes multiple fallback strategies for JSON parsing.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert ER triage physician. Your job is to synthesize three signals to make a final, clinically sound admission decision.

You are given three inputs:
1) p_ml:  probability of admission from a traditional ML model.
2) p_llm: probability of admission from an LLM classifier.
3) human_note: short free-text note from a nurse or physician providing real-time context.

Your task:
- Interpret all three signals.
- Resolve disagreements between the signals.
- Produce ONE final admission decision.
- Provide ONE rationale explaining exactly WHY you chose "Admit" or "Discharge".
  * Your rationale MUST explicitly reference p_ml, p_llm, and human_note.
  * It MUST give a clear clinical justification (e.g., high risk → admit, stable symptoms → discharge).

Output STRICTLY as a single valid JSON object with EXACTLY two keys:
{{
  "decision": "Admit" | "Discharge",
  "rationale": "string (2–4 sentences explaining the reason for your decision based on p_ml, p_llm, and human_note)"
}}

Do NOT output anything else.
Do NOT add comments or markdown.
Return ONLY the JSON object.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Please make a final decision based on this information:
- p_ml (ML model): {ml_prob:.2f}
- p_llm (LLM classifier): {llm_prob:.2f}
- human_note: "{human_note}"

Return ONLY the JSON object described above:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    device = llm_fusion_model.device
    
    for attempt in range(max_retries + 1):
        try:
            inputs = fusion_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            # Generate the response with improved parameters
            with torch.no_grad():
                outputs = llm_fusion_model.generate(
                    **inputs,
                    max_new_tokens=200,  # Increased for better JSON generation
                    eos_token_id=fusion_tokenizer.eos_token_id,
                    pad_token_id=fusion_tokenizer.pad_token_id,
                    temperature=0.3,  # Lower temperature for more consistent JSON
                    do_sample=True,
                    top_p=0.9
                )

            # Decode and clean the output
            response_text = fusion_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # DEBUG: Print raw response for debugging (first attempt only)
            if attempt == 0:
                print(f"[DEBUG] Raw LLM response (first 200 chars): {response_text[:200]}")
            
            # Use robust JSON parsing with fallback strategies
            parsed_json = parse_json_with_fallback(response_text)
            
            if parsed_json:
                # More lenient validation - check if we have at least a decision
                if "decision" in parsed_json:
                    decision = str(parsed_json["decision"]).strip()
                    # Normalize decision values
                    if decision.lower() in ["admit", "admission", "admitted"]:
                        parsed_json["decision"] = "Admit"
                    elif decision.lower() in ["discharge", "discharged", "discharging"]:
                        parsed_json["decision"] = "Discharge"
                    else:
                        # If decision is not recognized, infer from context
                        print(f"[WARNING] Unrecognized decision value: '{decision}', inferring from probabilities")
                        if ml_prob > 0.7 or llm_prob > 0.7:
                            parsed_json["decision"] = "Admit"
                        else:
                            parsed_json["decision"] = "Discharge"
                    
                    # Ensure rationale exists (create default if missing)
                    if "rationale" not in parsed_json or not parsed_json["rationale"]:
                        parsed_json["rationale"] = f"Based on ML probability {ml_prob:.2f} and LLM probability {llm_prob:.2f}, decision: {parsed_json['decision']}."
                    
                    return parsed_json
                else:
                    print(f"[WARNING] JSON missing 'decision' key. Keys found: {list(parsed_json.keys())}. Attempt {attempt + 1}/{max_retries + 1}")
                    print(f"[DEBUG] Parsed JSON content: {parsed_json}")
            else:
                print(f"[WARNING] Failed to parse JSON. Raw response: {response_text[:300]}. Attempt {attempt + 1}/{max_retries + 1}")
                if attempt < max_retries:
                    # Try with a more explicit prompt on retry
                    prompt = prompt.replace("Return ONLY the JSON object described above:", 
                                          "CRITICAL: You must return ONLY valid JSON. Return ONLY the JSON object described above:")
                    continue
        
        except Exception as e:
            print(f"[WARNING] Fusion agent error on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                log_error("fusion_agent", e, {"ml_prob": ml_prob, "llm_prob": llm_prob}, get_execution_id())
    
    # All attempts failed - return error response
    return {
        "decision": "Error",
        "rationale": f"Fusion agent failed after {max_retries + 1} attempts. Using weighted average fallback."
    }


# ============================================================================
# 2. ENHANCED fetch_data_node - WITH PATIENT HISTORY
# ============================================================================

def fetch_data_node(state: ERState):
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

    # Update state with execution ID for traceability
    state_update = {"execution_id": execution_id}

    conn = None
    try:
        # Validate DB path exists
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database file not found: {DB_PATH}")

        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row

        # ENHANCED QUERY WITH PATIENT HISTORY
        query = """
        WITH current_visit AS (
            SELECT 
                v.visit_id, v.patient_id, v.sex, v.age_bucket,
                v.heart_rate, v.bp_systolic, v.bp_diastolic, v.resp_rate,
                v.temperature_C, v.oxygen_saturation, v.recent_admissions_30d,
                v.admitted, v.Admission_Date,
                t.triage_notes_redacted,
                e.ESI
            FROM Visit_Details v
            LEFT JOIN Triage_Notes t ON v.visit_id = t.visit_id AND v.patient_id = t.patient_id
            LEFT JOIN ESI e ON v.visit_id = e.visit_id AND v.patient_id = e.patient_id
            WHERE v.visit_id = ?
        ),
        patient_history AS (
            SELECT 
                patient_id,
                COUNT(*) as total_visits,
                SUM(admitted) as total_admissions,
                AVG(heart_rate) as avg_hr_history,
                AVG(bp_systolic) as avg_bp_sys_history,
                MAX(Admission_Date) as last_admission_date
            FROM Visit_Details
            WHERE patient_id = (SELECT patient_id FROM current_visit)
              AND visit_id < (SELECT visit_id FROM current_visit)
            GROUP BY patient_id
        )
        SELECT 
            cv.*,
            COALESCE(ph.total_visits, 0) as historical_visit_count,
            COALESCE(ph.total_admissions, 0) as historical_admission_count,
            ph.avg_hr_history,
            ph.avg_bp_sys_history,
            ph.last_admission_date
        FROM current_visit cv
        LEFT JOIN patient_history ph ON cv.patient_id = ph.patient_id
        """

        cursor = conn.cursor()
        cursor.execute(query, (visit_id,))
        row = cursor.fetchone()

    except sqlite3.Error as e:
        error_msg = f"Database error for visit_id {visit_id}: {str(e)}"
        log_error("fetch_data", e, state, execution_id)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error fetching data for visit_id {visit_id}: {str(e)}"
        log_error("fetch_data", e, state, execution_id)
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    if row is None:
        error_msg = f"No data found for visit_id: {visit_id} in {DB_PATH}"
        raise ValueError(error_msg)

    patient_data = dict(row)

    # Validate that we have essential data
    if not patient_data.get('visit_id'):
        raise ValueError(f"Invalid patient data returned for visit_id: {visit_id}")

    # Validate vitals with error handling
    try:
        vitals_validated = VitalSigns(**patient_data)
    except Exception as e:
        log_error("fetch_data_vitals_validation", e, {"patient_data": patient_data}, execution_id)
        # Try to create with minimal required fields
        vitals_validated = VitalSigns(
            sex=patient_data.get('sex'),
            age_bucket=patient_data.get('age_bucket'),
            heart_rate=patient_data.get('heart_rate'),
            resp_rate=patient_data.get('resp_rate'),
            bp_systolic=patient_data.get('bp_systolic'),
            bp_diastolic=patient_data.get('bp_diastolic'),
            oxygen_saturation=patient_data.get('oxygen_saturation'),
            temperature_C=patient_data.get('temperature_C'),
            ESI=patient_data.get('ESI'),
            recent_admissions_30d=patient_data.get('recent_admissions_30d')
        )

    # Extract temporal features
    temporal_features = extract_temporal_features(patient_data)
    if temporal_features:
        patient_data.update(temporal_features)

    state_update.update({
        "patient_data": patient_data,
        "vitals_validated": vitals_validated,
        "triage_text": patient_data.get('triage_notes_redacted', '')
    })

    return state_update


# ============================================================================
# 3. ENHANCED severity_gate_node - WITH ESI AND RISK FACTORS
# ============================================================================

def severity_gate_node(state: ERState):
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


# ============================================================================
# 4. ENHANCED conditional_confidence_routing - WITH RISK-BASED THRESHOLDS
# ============================================================================

def conditional_confidence_routing(state: ERState):
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


# ============================================================================
# 5. ENHANCED fusion_node - WITH PATIENT CONTEXT
# ============================================================================

def fusion_node(state: ERState):
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
    age_bucket = patient_data.get('age_bucket', 'Unknown')

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

