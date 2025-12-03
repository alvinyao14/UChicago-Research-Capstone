# Database-Driven Workflow Enhancements

Based on analysis of `ED_Simulated_Database_Fixed.db` and your current workflow implementation, here are specific, actionable enhancements.

---

## 🔍 Database Schema Analysis

### Available Tables:
1. **Visit_Details** (400 visits)
   - Core patient data: vitals, demographics, admission status
   - `recent_admissions_30d`: Historical admission frequency
   - `Admission_Date`: Temporal information
   
2. **Triage_Notes**
   - Clinical narrative text (redacted)
   
3. **ESI** (Emergency Severity Index)
   - Triage acuity level (1-5)
   
4. **Patient_PII**
   - Patient identifiers (for linking historical data)

### Current Usage Gaps:
- ❌ `recent_admissions_30d` not used in decision logic
- ❌ `Admission_Date` not used for temporal features
- ❌ No patient history lookup across visits
- ❌ ESI level not incorporated into severity gate
- ❌ No trend analysis (vital sign changes over time)

---

## 🚀 Priority Enhancements

### 1. **Enhanced Data Fetching with Patient History**

#### Current Issue:
```python
# Current query only gets single visit
query = """
SELECT v.visit_id, v.patient_id, ...
FROM Visit_Details v
WHERE v.visit_id = ?
"""
```

#### Enhancement:
```python
def fetch_data_with_history_node(state: ERState):
    """
    Enhanced fetch_data that includes patient history and trends.
    """
    visit_id = state.get('visit_id')
    
    # Main query with patient history
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
        LEFT JOIN Triage_Notes t ON v.visit_id = t.visit_id 
        LEFT JOIN ESI e ON v.visit_id = e.visit_id
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
    ),
    recent_trends AS (
        SELECT 
            patient_id,
            AVG(CASE WHEN visit_id >= (SELECT visit_id FROM current_visit) - 3 
                     THEN heart_rate END) as recent_avg_hr,
            AVG(CASE WHEN visit_id >= (SELECT visit_id FROM current_visit) - 3 
                     THEN bp_systolic END) as recent_avg_bp
        FROM Visit_Details
        WHERE patient_id = (SELECT patient_id FROM current_visit)
        GROUP BY patient_id
    )
    SELECT 
        cv.*,
        COALESCE(ph.total_visits, 0) as historical_visit_count,
        COALESCE(ph.total_admissions, 0) as historical_admission_count,
        ph.avg_hr_history,
        ph.avg_bp_sys_history,
        ph.last_admission_date,
        rt.recent_avg_hr,
        rt.recent_avg_bp,
        CASE 
            WHEN cv.heart_rate > rt.recent_avg_hr * 1.2 THEN 'increasing'
            WHEN cv.heart_rate < rt.recent_avg_hr * 0.8 THEN 'decreasing'
            ELSE 'stable'
        END as hr_trend
    FROM current_visit cv
    LEFT JOIN patient_history ph ON cv.patient_id = ph.patient_id
    LEFT JOIN recent_trends rt ON cv.patient_id = rt.patient_id
    """
    
    # Execute and return enhanced data
    # ...
```

**Benefits:**
- Patient-specific risk factors (frequent flyer detection)
- Trend analysis (deteriorating vs. improving)
- Historical context for decision-making

---

### 2. **ESI-Enhanced Severity Gate**

#### Current Issue:
```python
# Only checks basic vital signs
if (v.oxygen_saturation < 88) or (v.bp_systolic < 80) or ...
```

#### Enhancement:
```python
def enhanced_severity_gate_node(state: ERState):
    """
    Enhanced severity gate incorporating ESI and historical patterns.
    """
    v = state["vitals_validated"]
    patient_data = state.get("patient_data", {})
    
    # Critical vital signs (existing)
    critical_vitals = (
        (v.oxygen_saturation is not None and v.oxygen_saturation < 88) or
        (v.bp_systolic is not None and v.bp_systolic < 80) or
        (v.resp_rate is not None and (v.resp_rate > 35 or v.resp_rate < 8))
    )
    
    # ESI-based severity (ESI 1-2 are critical)
    esi = patient_data.get('ESI')
    critical_esi = esi is not None and esi <= 2
    
    # High-risk patterns
    recent_admissions = patient_data.get('recent_admissions_30d', 0)
    high_readmission_risk = recent_admissions >= 2  # 2+ admissions in 30 days
    
    # Age-based risk adjustment
    age_bucket = patient_data.get('age_bucket', '')
    elderly_risk = age_bucket == '65+' and (
        v.heart_rate > 100 or 
        v.temperature_C > 38.5 or 
        v.oxygen_saturation < 92
    )
    
    # Trend-based risk (if available)
    hr_trend = patient_data.get('hr_trend', 'stable')
    deteriorating = hr_trend == 'increasing' and v.heart_rate > 100
    
    # Combined severity assessment
    is_severe = (
        critical_vitals or 
        critical_esi or 
        (high_readmission_risk and elderly_risk) or
        deteriorating
    )
    
    if is_severe:
        severity_reason = []
        if critical_vitals: severity_reason.append("Critical vitals")
        if critical_esi: severity_reason.append(f"ESI level {esi}")
        if high_readmission_risk: severity_reason.append("High readmission risk")
        if deteriorating: severity_reason.append("Deteriorating trend")
        
        return {
            "severe": True,
            "decision": "Admit",
            "p_final": 1.0,
            "rationale": f"Severe case: {', '.join(severity_reason)}. Immediate admission required.",
            "severity_factors": severity_reason
        }
    
    return {"severe": False}
```

**Benefits:**
- More comprehensive risk assessment
- Incorporates clinical triage standards (ESI)
- Identifies high-risk patient patterns

---

### 3. **Temporal Feature Engineering**

#### Enhancement:
```python
def extract_temporal_features(patient_data: dict) -> dict:
    """
    Extract temporal features from admission date.
    """
    admission_date = patient_data.get('Admission_Date')
    if not admission_date:
        return {}
    
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(admission_date.replace('Z', '+00:00'))
        
        return {
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),  # 0=Monday, 6=Sunday
            "is_weekend": dt.weekday() >= 5,
            "is_night": 22 <= dt.hour or dt.hour < 6,
            "month": dt.month,
            "is_holiday_season": dt.month in [11, 12, 1],  # Nov-Dec-Jan
        }
    except:
        return {}

# Use in fusion node
def fusion_node_with_temporal(state: ERState):
    """
    Enhanced fusion with temporal context.
    """
    # ... existing fusion logic ...
    
    # Add temporal features
    temporal_features = extract_temporal_features(state.get("patient_data", {}))
    
    # Adjust decision based on temporal factors
    # E.g., weekend/night admissions might have different thresholds
    if temporal_features.get("is_weekend") or temporal_features.get("is_night"):
        # Lower threshold for weekend/night (more conservative)
        adjusted_threshold = ADMISSION_THRESHOLD * 0.9
    else:
        adjusted_threshold = ADMISSION_THRESHOLD
    
    # Include temporal context in human note
    temporal_context = []
    if temporal_features.get("is_weekend"):
        temporal_context.append("Weekend presentation")
    if temporal_features.get("is_night"):
        temporal_context.append("Night shift")
    
    enhanced_human_note = state.get("human_prompt", "")
    if temporal_context:
        enhanced_human_note += f" [Context: {', '.join(temporal_context)}]"
    
    # ... rest of fusion logic with enhanced context ...
```

**Benefits:**
- Captures time-of-day and day-of-week patterns
- Can identify seasonal trends
- Adjusts thresholds based on operational context

---

### 4. **Patient Risk Scoring with History**

#### Enhancement:
```python
def calculate_patient_risk_score(patient_data: dict) -> float:
    """
    Calculate comprehensive patient risk score using all available data.
    """
    risk_factors = []
    risk_score = 0.0
    
    # 1. Age risk (0-0.2)
    age_bucket = patient_data.get('age_bucket', '')
    age_risk = {
        '0-17': 0.1,
        '18-34': 0.0,
        '35-49': 0.05,
        '50-64': 0.1,
        '65+': 0.2
    }.get(age_bucket, 0.0)
    risk_score += age_risk
    
    # 2. ESI risk (0-0.3)
    esi = patient_data.get('ESI', 3)
    esi_risk = {
        1: 0.3, 2: 0.25, 3: 0.1, 4: 0.05, 5: 0.0
    }.get(esi, 0.1)
    risk_score += esi_risk
    
    # 3. Readmission risk (0-0.2)
    recent_adm = patient_data.get('recent_admissions_30d', 0)
    readmission_risk = min(recent_adm * 0.1, 0.2)
    risk_score += readmission_risk
    
    # 4. Historical admission rate (0-0.15)
    hist_visits = patient_data.get('historical_visit_count', 0)
    hist_admissions = patient_data.get('historical_admission_count', 0)
    if hist_visits > 0:
        admission_rate = hist_admissions / hist_visits
        risk_score += admission_rate * 0.15
    
    # 5. Vital sign abnormalities (0-0.15)
    vitals_risk = 0.0
    hr = patient_data.get('heart_rate', 70)
    if hr > 100 or hr < 60:
        vitals_risk += 0.05
    bp_sys = patient_data.get('bp_systolic', 120)
    if bp_sys > 160 or bp_sys < 90:
        vitals_risk += 0.05
    o2 = patient_data.get('oxygen_saturation', 98)
    if o2 < 95:
        vitals_risk += 0.05
    risk_score += min(vitals_risk, 0.15)
    
    return min(risk_score, 1.0)  # Cap at 1.0

# Use in confidence routing
def conditional_confidence_routing_with_risk(state: ERState):
    """
    Enhanced confidence routing that considers patient risk score.
    """
    ml = state.get("ml_score")
    llm = state.get("llm_score")
    patient_data = state.get("patient_data", {})
    
    # Calculate patient risk score
    patient_risk = calculate_patient_risk_score(patient_data)
    
    # Adjust confidence thresholds based on patient risk
    if patient_risk > 0.5:  # High-risk patient
        HIGH_CONF_GAP = 0.15  # Tighter agreement required
        HIGH_CONF_THRESH = 0.65  # Lower threshold (more conservative)
    else:  # Low-risk patient
        HIGH_CONF_GAP = 0.25
        HIGH_CONF_THRESH = 0.75
    
    # ... rest of routing logic ...
```

**Benefits:**
- Personalized risk assessment
- More nuanced confidence routing
- Better handling of edge cases

---

### 5. **Enhanced Database Query with Indexing**

#### Current Issue:
- No explicit indexes (except MRN)
- Sequential scans for history lookups

#### Enhancement:
```python
# Add indexes for common queries
CREATE_INDEX_QUERIES = [
    "CREATE INDEX IF NOT EXISTS idx_visit_patient ON Visit_Details(patient_id, visit_id);",
    "CREATE INDEX IF NOT EXISTS idx_visit_date ON Visit_Details(Admission_Date);",
    "CREATE INDEX IF NOT EXISTS idx_visit_admitted ON Visit_Details(admitted, visit_id);",
    "CREATE INDEX IF NOT EXISTS idx_esi_level ON ESI(ESI, visit_id);",
]

def initialize_database_indexes(db_path: str):
    """Create indexes for performance."""
    conn = sqlite3.connect(db_path)
    for query in CREATE_INDEX_QUERIES:
        conn.execute(query)
    conn.commit()
    conn.close()
```

---

### 6. **Fusion Agent with Database Context**

#### Enhancement:
```python
def run_fusion_agent_with_context(ml_prob: float, llm_prob: float, 
                                   human_note: str, patient_data: dict) -> dict:
    """
    Enhanced fusion agent that incorporates patient history context.
    """
    # Extract relevant context
    recent_admissions = patient_data.get('recent_admissions_30d', 0)
    historical_admissions = patient_data.get('historical_admission_count', 0)
    historical_visits = patient_data.get('historical_visit_count', 0)
    esi = patient_data.get('ESI', 3)
    age_bucket = patient_data.get('age_bucket', 'Unknown')
    
    # Build context string
    context_parts = []
    if recent_admissions > 0:
        context_parts.append(f"Patient has {recent_admissions} recent admission(s) in past 30 days")
    if historical_visits > 0:
        admission_rate = historical_admissions / historical_visits if historical_visits > 0 else 0
        context_parts.append(f"Historical admission rate: {admission_rate:.1%} ({historical_admissions}/{historical_visits} visits)")
    if esi <= 2:
        context_parts.append(f"High acuity triage (ESI {esi})")
    
    context_str = ". ".join(context_parts) if context_parts else "No significant historical patterns."
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert ER triage physician making admission decisions.

**Patient Context:**
- Age group: {age_bucket}
- ESI level: {esi}
- Historical context: {context_str}

**Model Predictions:**
- ML model probability: {ml_prob:.2f}
- LLM classifier probability: {llm_prob:.2f}
- Human note: "{human_note}"

**Your Task:**
1. Consider the patient's history (frequent visits, admission patterns)
2. Evaluate model agreement/disagreement
3. Incorporate clinical context from human note
4. Make final decision with clear rationale

Output JSON:
{{
  "decision": "Admit" | "Discharge",
  "rationale": "2-4 sentences",
  "confidence": 0.0-1.0,
  "key_factors": ["factor1", "factor2"]
}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Make your decision.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    # ... rest of implementation ...
```

---

### 7. **Data Quality Validation**

#### Enhancement:
```python
def validate_database_data(patient_data: dict) -> tuple[bool, List[str]]:
    """
    Validate data quality from database.
    """
    warnings = []
    
    # Check for missing critical fields
    critical_fields = ['visit_id', 'patient_id', 'heart_rate', 'bp_systolic']
    missing = [f for f in critical_fields if not patient_data.get(f)]
    if missing:
        warnings.append(f"Missing critical fields: {missing}")
    
    # Check for data anomalies
    hr = patient_data.get('heart_rate')
    if hr and (hr < 20 or hr > 220):
        warnings.append(f"Suspicious heart rate: {hr}")
    
    # Check for inconsistent data
    bp_sys = patient_data.get('bp_systolic')
    bp_dia = patient_data.get('bp_diastolic')
    if bp_sys and bp_dia and bp_sys < bp_dia:
        warnings.append(f"BP inconsistency: systolic ({bp_sys}) < diastolic ({bp_dia})")
    
    # Check ESI vs vitals consistency
    esi = patient_data.get('ESI', 3)
    o2 = patient_data.get('oxygen_saturation', 98)
    if esi <= 2 and o2 > 95:  # High ESI but normal O2
        warnings.append(f"ESI {esi} may not match vitals (O2: {o2}%)")
    
    return len(warnings) == 0, warnings
```

---

## 📊 Implementation Priority

### Phase 1 (Immediate - High Impact):
1. ✅ Enhanced data fetching with patient history
2. ✅ ESI-enhanced severity gate
3. ✅ Patient risk scoring

### Phase 2 (Next - Medium Impact):
4. ⚠️ Temporal feature engineering
5. ⚠️ Database indexing
6. ⚠️ Data quality validation

### Phase 3 (Future - Nice to Have):
7. 📋 Advanced trend analysis
8. 📋 Predictive readmission modeling
9. 📋 Real-time database updates

---

## 🔧 Quick Implementation Example

Here's a complete enhanced `fetch_data_node`:

```python
def fetch_data_node_enhanced(state: ERState):
    """Enhanced fetch with history and validation."""
    visit_id = state.get('visit_id')
    
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    
    # Enhanced query with history
    query = """
    WITH current AS (
        SELECT v.*, t.triage_notes_redacted, e.ESI
        FROM Visit_Details v
        LEFT JOIN Triage_Notes t ON v.visit_id = t.visit_id
        LEFT JOIN ESI e ON v.visit_id = e.visit_id
        WHERE v.visit_id = ?
    )
    SELECT 
        c.*,
        (SELECT COUNT(*) FROM Visit_Details v2 
         WHERE v2.patient_id = c.patient_id 
           AND v2.visit_id < c.visit_id) as historical_visit_count,
        (SELECT SUM(admitted) FROM Visit_Details v2 
         WHERE v2.patient_id = c.patient_id 
           AND v2.visit_id < c.visit_id) as historical_admission_count
    FROM current c
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (visit_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"No data for visit_id: {visit_id}")
    
    patient_data = dict(row)
    
    # Validate data quality
    is_valid, warnings = validate_database_data(patient_data)
    if warnings:
        log_event("data_quality_warnings", state, {"warnings": warnings})
    
    # Validate vitals
    vitals_validated = VitalSigns(**patient_data)
    
    return {
        "patient_data": patient_data,
        "vitals_validated": vitals_validated,
        "triage_text": patient_data.get('triage_notes_redacted', ''),
        "data_quality_warnings": warnings if warnings else None
    }
```

---

## 📈 Expected Improvements

1. **Accuracy**: +3-5% from patient history context
2. **Robustness**: Better handling of edge cases
3. **Clinical Relevance**: ESI and history-based decisions
4. **Performance**: Faster queries with indexes
5. **Traceability**: Better audit trail with data quality checks

---

*Last Updated: 2024*
*Based on ED_Simulated_Database_Fixed.db schema analysis*

