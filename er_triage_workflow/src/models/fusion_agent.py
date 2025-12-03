"""Fusion Agent LLM for synthesizing model predictions."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict

from ...config import get_config
from ..utils.json_parser import parse_json_with_fallback
from ..utils.logging import log_error, get_execution_id


class FusionAgent:
    """Wrapper for Fusion Agent LLM."""
    
    def __init__(self, config=None):
        """Load Fusion Agent LLM model and tokenizer."""
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Configure 4-bit quantization to save memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_id,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("\nSuccessfully loaded base OpenBioLLM *Fusion Agent* model.")
    
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate response from prompt."""
        device = self.model.device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response_text


# Global fusion agent instance (lazy loaded)
_fusion_agent_instance = None


def get_fusion_agent() -> FusionAgent:
    """Get or create global fusion agent instance."""
    global _fusion_agent_instance
    if _fusion_agent_instance is None:
        _fusion_agent_instance = FusionAgent()
    return _fusion_agent_instance


def run_fusion_agent(ml_prob: float, llm_prob: float, human_note: str, max_retries: int = 2) -> Dict:
    """
    Enhanced fusion agent with robust JSON parsing and retry logic.
    
    Uses the generative LLM to synthesize inputs and make a final decision with rationale.
    Includes multiple fallback strategies for JSON parsing.
    
    Args:
        ml_prob: ML model probability (0-1)
        llm_prob: LLM classifier probability (0-1)
        human_note: Human-provided clinical note
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with 'decision' and 'rationale' keys
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

    fusion_agent = get_fusion_agent()
    
    for attempt in range(max_retries + 1):
        try:
            response_text = fusion_agent.generate(prompt, max_new_tokens=200)
            
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
                        parsed_json["rationale"] = (
                            f"Based on ML probability {ml_prob:.2f} and LLM probability {llm_prob:.2f}, "
                            f"decision: {parsed_json['decision']}."
                        )
                    
                    return parsed_json
                else:
                    print(f"[WARNING] JSON missing 'decision' key. Keys found: {list(parsed_json.keys())}. Attempt {attempt + 1}/{max_retries + 1}")
            else:
                print(f"[WARNING] Failed to parse JSON. Raw response: {response_text[:300]}. Attempt {attempt + 1}/{max_retries + 1}")
                if attempt < max_retries:
                    # Try with a more explicit prompt on retry
                    prompt = prompt.replace(
                        "Return ONLY the JSON object described above:",
                        "CRITICAL: You must return ONLY valid JSON. Return ONLY the JSON object described above:"
                    )
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

