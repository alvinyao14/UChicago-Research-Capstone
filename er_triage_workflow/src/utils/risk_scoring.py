"""Patient risk scoring and temporal feature extraction utilities."""

from datetime import datetime
from typing import Dict
from ..utils.logging import log_error, get_execution_id


def extract_temporal_features(patient_data: dict) -> Dict[str, int]:
    """
    Extract temporal features from Admission_Date.
    
    Args:
        patient_data: Patient data dictionary containing 'Admission_Date'
        
    Returns:
        Dictionary with temporal features:
        - hour_of_day: Hour (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - is_weekend: Boolean (1 if weekend)
        - is_night: Boolean (1 if night shift: 22:00-06:00)
        - month: Month (1-12)
        - is_holiday_season: Boolean (1 if Nov, Dec, or Jan)
    """
    admission_date = patient_data.get('Admission_Date')
    if not admission_date:
        return {}
    
    try:
        # Handle different date formats
        if 'T' in str(admission_date):
            dt = datetime.fromisoformat(str(admission_date).replace('Z', '+00:00'))
        else:
            # Try common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S']:
                try:
                    dt = datetime.strptime(str(admission_date), fmt)
                    break
                except ValueError:
                    continue
            else:
                return {}
        
        return {
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),  # 0=Monday, 6=Sunday
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
            "is_night": 1 if (22 <= dt.hour or dt.hour < 6) else 0,
            "month": dt.month,
            "is_holiday_season": 1 if dt.month in [11, 12, 1] else 0,
        }
    except Exception as e:
        # TODO: Uncomment when logging.py is created
        # log_error("temporal_extraction", e, patient_data, get_execution_id())
        print(f"[WARNING] Temporal extraction error: {e}")
        return {}


def calculate_patient_risk_score(patient_data: dict) -> float:
    """
    Calculate comprehensive patient risk score using all available data.
    
    Risk factors considered:
    - Age group (0-0.2)
    - ESI level (0-0.3)
    - Recent readmissions (0-0.2)
    - Historical admission rate (0-0.15)
    - Vital sign abnormalities (0-0.15)
    
    Args:
        patient_data: Patient data dictionary
        
    Returns:
        Risk score from 0.0 to 1.0
    """
    risk = 0.0
    
    # Age risk (0-0.2)
    age_bucket = patient_data.get('age_bucket', '')
    age_risk = {
        '0-17': 0.1, '18-34': 0.0, '35-49': 0.05,
        '50-64': 0.1, '65+': 0.2
    }.get(age_bucket, 0.0)
    risk += age_risk
    
    # ESI risk (0-0.3)
    esi = patient_data.get('ESI', 3)
    esi_risk = {1: 0.3, 2: 0.25, 3: 0.1, 4: 0.05, 5: 0.0}.get(esi, 0.1)
    risk += esi_risk
    
    # Readmission risk (0-0.2)
    recent_adm = patient_data.get('recent_admissions_30d', 0)
    risk += min(recent_adm * 0.1, 0.2)
    
    # Historical admission rate (0-0.15)
    hist_visits = patient_data.get('historical_visit_count', 0)
    hist_admissions = patient_data.get('historical_admission_count', 0)
    if hist_visits > 0:
        admission_rate = hist_admissions / hist_visits
        risk += admission_rate * 0.15
    
    # Vital sign abnormalities (0-0.15)
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
    risk += min(vitals_risk, 0.15)
    
    return min(risk, 1.0)  # Cap at 1.0

