"""Database queries for patient data and history."""

import sqlite3
import os
from typing import Dict, Optional

from ...config import get_config
from ..workflow.state import VitalSigns
from ..utils.logging import log_error, get_execution_id, init_execution_context
from ..utils.risk_scoring import extract_temporal_features


def fetch_patient_data(visit_id: int) -> Dict:
    """
    Fetch patient data and history from database.
    
    Args:
        visit_id: Visit ID to fetch
        
    Returns:
        Dictionary with patient data including history
        
    Raises:
        ValueError: If visit_id is invalid or no data found
        RuntimeError: If database error occurs
    """
    cfg = get_config()
    execution_id = get_execution_id()
    
    # Initialize execution context if not already set
    if execution_id is None:
        init_execution_context(visit_id)
        execution_id = get_execution_id()
    
    # Validate input
    if not visit_id or not isinstance(visit_id, int) or visit_id <= 0:
        raise ValueError(f"Invalid visit_id: {visit_id}")
    
    # Validate DB path exists
    if not os.path.exists(cfg.db_path):
        raise FileNotFoundError(f"Database file not found: {cfg.db_path}")
    
    conn = None
    try:
        conn = sqlite3.connect(str(cfg.db_path), timeout=10.0)
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
        log_error("fetch_patient_data", e, {"visit_id": visit_id}, execution_id)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error fetching data for visit_id {visit_id}: {str(e)}"
        log_error("fetch_patient_data", e, {"visit_id": visit_id}, execution_id)
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
    
    if row is None:
        error_msg = f"No data found for visit_id: {visit_id} in {cfg.db_path}"
        raise ValueError(error_msg)
    
    patient_data = dict(row)
    
    # Validate that we have essential data
    if not patient_data.get('visit_id'):
        raise ValueError(f"Invalid patient data returned for visit_id: {visit_id}")
    
    # Validate vitals with error handling
    try:
        vitals_validated = VitalSigns(**patient_data)
    except Exception as e:
        log_error("fetch_patient_data_vitals_validation", e, {"patient_data": patient_data}, execution_id)
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
    
    return {
        "patient_data": patient_data,
        "vitals_validated": vitals_validated,
        "triage_text": patient_data.get('triage_notes_redacted', ''),
        "execution_id": execution_id
    }

