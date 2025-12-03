"""Robust JSON parsing utilities with multiple fallback strategies."""

import json
import re
from typing import Optional


def parse_json_with_fallback(text: str) -> Optional[dict]:
    """
    Robust JSON parsing with multiple fallback strategies.
    
    Strategies:
    1. Direct JSON parsing
    2. Extract JSON from markdown code blocks
    3. Extract JSON using brace matching (handles nested objects)
    4. Fix common JSON issues (trailing commas, unquoted keys)
    5. Manual key-value extraction as last resort
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Extract JSON using brace matching (handles nested objects correctly)
    brace_count = 0
    start_idx = text.find('{')
    if start_idx != -1:
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Strategy 4: Try to fix common issues
                        # Remove trailing commas
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                    break
    
    # Strategy 5: Fallback - extract key-value pairs manually
    decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    
    if decision_match:
        result = {"decision": decision_match.group(1)}
        if rationale_match:
            result["rationale"] = rationale_match.group(1)
        else:
            # Try multi-line rationale (may have escaped quotes or newlines)
            rationale_match = re.search(r'"rationale"\s*:\s*(.+?)(?:\s*[,}])', text, re.IGNORECASE | re.DOTALL)
            if rationale_match:
                rationale = rationale_match.group(1).strip().strip('"').strip("'")
                result["rationale"] = rationale
        return result
    
    return None

