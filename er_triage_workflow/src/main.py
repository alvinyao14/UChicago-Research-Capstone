"""Main entry point for ER Admission Agentic AI."""

import time
from datetime import datetime
from typing import Dict, Optional

from .workflow.graph import compile_workflow
from .workflow.state import validate_initial_state
from .utils.logging import init_execution_context, get_execution_id, log_error
from .utils.logging import _execution_context


def invoke_workflow_with_tracking(graph, inputs: Dict, config: Dict = None) -> Dict:
    """
    Enhanced workflow invocation with automatic execution tracking.

    This function:
    1. Validates initial state
    2. Initializes execution context
    3. Invokes the graph
    4. Returns results with execution metadata

    Args:
        graph: Compiled LangGraph workflow
        inputs: Input state dictionary
        config: LangGraph configuration (optional)

    Returns:
        Final state dictionary with execution metadata
    """
    # Validate initial state
    is_valid, error_msg = validate_initial_state(inputs)
    if not is_valid:
        raise ValueError(f"Invalid initial state: {error_msg}")

    # Initialize execution context
    visit_id = inputs.get('visit_id')
    execution_id = init_execution_context(visit_id)

    print(f"\n{'='*60}")
    print(f"Starting workflow execution")
    print(f"  Execution ID: {execution_id}")
    print(f"  Visit ID: {visit_id}")
    print(f"{'='*60}\n")

    try:
        # Invoke graph
        if config is None:
            config = {"configurable": {"thread_id": f"exec-{execution_id}"}}

        final_state = graph.invoke(inputs, config)

        # Add execution metadata
        execution_time = time.time() - _execution_context["start_time"]
        final_state["execution_metadata"] = {
            "execution_id": execution_id,
            "visit_id": visit_id,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        }

        print(f"\n{'='*60}")
        print(f"Workflow execution completed")
        print(f"  Execution ID: {execution_id}")
        print(f"  Total time: {execution_time:.2f}s")
        print(f"  Final decision: {final_state.get('decision', 'N/A')}")
        print(f"{'='*60}\n")

        return final_state

    except Exception as e:
        execution_time = time.time() - _execution_context["start_time"]
        log_error("workflow_invocation", e, inputs, execution_id)

        print(f"\n{'='*60}")
        print(f"Workflow execution failed")
        print(f"  Execution ID: {execution_id}")
        print(f"  Time before failure: {execution_time:.2f}s")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        print(f"{'='*60}\n")

        raise


def run_simulation(visit_id: int, human_prompt: str, thread_id: Optional[str] = None):
    """
    Run a single workflow simulation.
    
    Args:
        visit_id: Visit ID to process
        human_prompt: Human-provided clinical note
        thread_id: Optional thread ID for LangGraph checkpointing
        
    Returns:
        Final state dictionary
    """
    # Compile the workflow
    graph = compile_workflow()
    
    # Prepare inputs
    inputs = {
        "visit_id": visit_id,
        "human_prompt": human_prompt
    }
    
    if thread_id is None:
        thread_id = f"sim-{visit_id}"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the workflow
    final_state = invoke_workflow_with_tracking(graph, inputs, config)
    
    return final_state


def main():
    """Example main function demonstrating workflow usage."""
    print("ER Admission Agentic AI")
    print("=" * 60)
    
    # Example 1: High-Risk Patient Note
    print("\n--- Running Simulation 1 (High-Risk Note) ---")
    final_state_1 = run_simulation(
        visit_id=1,
        human_prompt="Patient is 70yo, frail, and on chemotherapy."
    )
    
    print("\n--- Final State 1 ---")
    print(f"Decision: {final_state_1.get('decision')}")
    print(f"P(Admit): {final_state_1.get('p_final'):.4f}")
    print(f"Final Rationale: {final_state_1.get('rationale')}")
    print(f"Fusion Rationale: {final_state_1.get('fusion_rationale')}")
    
    # Example 2: Low-Risk Patient Note
    print("\n\n--- Running Simulation 2 (Low-Risk Note) ---")
    final_state_2 = run_simulation(
        visit_id=5,
        human_prompt="Patient looks stable, likely just needs follow-up."
    )
    
    print("\n--- Final State 2 ---")
    print(f"Decision: {final_state_2.get('decision')}")
    print(f"P(Admit): {final_state_2.get('p_final'):.4f}")
    print(f"Final Rationale: {final_state_2.get('rationale')}")
    print(f"Fusion Rationale: {final_state_2.get('fusion_rationale')}")


if __name__ == "__main__":
    main()

