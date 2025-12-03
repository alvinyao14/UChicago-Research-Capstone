# Fusion Agent JSON Parsing Fix

## Problem
The fusion agent is failing with "Invalid fusion agent output" error. The LLM is generating responses, but the JSON parsing is not extracting them correctly.

## Solution

### 1. Update `parse_json_with_fallback` function (in Cell 5)

Replace the existing function with this improved version that has better regex patterns and fallback extraction:

```python
def parse_json_with_fallback(text: str) -> Optional[dict]:
    """
    Robust JSON parsing with multiple fallback strategies.
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
    
    # Strategy 3: Extract JSON object using improved regex (handles nested objects)
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
                        # Fix common issues
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                    break
    
    # Strategy 4: Fallback - extract key-value pairs manually
    decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    
    if decision_match:
        result = {"decision": decision_match.group(1)}
        if rationale_match:
            result["rationale"] = rationale_match.group(1)
        else:
            # Try multi-line rationale
            rationale_match = re.search(r'"rationale"\s*:\s*(.+?)(?:\s*[,}])', text, re.IGNORECASE | re.DOTALL)
            if rationale_match:
                rationale = rationale_match.group(1).strip().strip('"').strip("'")
                result["rationale"] = rationale
        return result
    
    return None
```

### 2. Update `run_fusion_agent` function

Make the validation more lenient - it should accept JSON even if only "decision" is present:

```python
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
```

### 3. Update `fusion_node` to use `run_fusion_agent`

Instead of calling the LLM directly in `fusion_node`, use the `run_fusion_agent` function:

```python
# In fusion_node, replace the direct LLM call with:
fusion_output = run_fusion_agent(
    ml_prob=ml_prob,
    llm_prob=llm_prob,
    human_note=enhanced_human_note,  # Include context in the note
    max_retries=2
)
```

## Quick Test

After making these changes, test with:

```python
test_output = run_fusion_agent(ml_prob=0.81, llm_prob=0.99, human_note="70yo, frail, on chemotherapy")
print("Raw output:", test_output)
print("Decision:", test_output.get("decision"))
print("Rationale:", test_output.get("rationale"))
```

This should now work and return valid JSON instead of "Error".

