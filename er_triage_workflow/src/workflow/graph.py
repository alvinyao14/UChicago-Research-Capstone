"""Graph construction for ER Triage Workflow."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ERState
from .nodes import (
    fetch_data_node,
    severity_gate_node,
    ml_model_node,
    llm_model_node,
    human_input_node,
    fusion_node,
    confidence_check_node,
    human_review_node,
    finalize_node,
    run_models_node,
    ADMISSION_THRESHOLD,
)
from .routing import conditional_severity_gate, conditional_confidence_routing
from ..utils.logging import make_logged_node


def build_workflow() -> StateGraph:
    """
    Build and configure the LangGraph workflow.
    
    Returns:
        Compiled LangGraph workflow
    """
    # Initialize the graph
    workflow = StateGraph(ERState)

    # Add all the nodes with enhanced error handling and retry logic
    # Critical nodes (fetch_data, severity_gate) have no retries - failures should be explicit
    # Non-critical nodes can retry transient failures
    workflow.add_node("fetch_data", make_logged_node(fetch_data_node, "fetch_data", max_retries=0))
    workflow.add_node("severity_gate", make_logged_node(severity_gate_node, "severity_gate", max_retries=0))
    workflow.add_node("run_models", make_logged_node(run_models_node, "run_models", max_retries=0))   # router node
    workflow.add_node("ml_model", make_logged_node(ml_model_node, "ml_model", max_retries=1, retry_delay=0.5))
    workflow.add_node("llm_model", make_logged_node(llm_model_node, "llm_model", max_retries=1, retry_delay=0.5))
    workflow.add_node("human_input", make_logged_node(human_input_node, "human_input", max_retries=0))
    workflow.add_node("fusion", make_logged_node(fusion_node, "fusion", max_retries=1, retry_delay=1.0))
    workflow.add_node("confidence_check", make_logged_node(confidence_check_node, "confidence_check", max_retries=0))
    workflow.add_node("human_review", make_logged_node(human_review_node, "human_review", max_retries=0))
    workflow.add_node("finalize", make_logged_node(finalize_node, "finalize", max_retries=0))

    # Define the graph flow
    # START → fetch_data (entry point)
    workflow.set_entry_point("fetch_data")

    # fetch_data → severity_gate (solid arrow)
    workflow.add_edge("fetch_data", "severity_gate")

    # severity_gate → conditional routing
    #   - "end" (early exit) → END
    #   - "run_models" (continue) → run_models
    workflow.add_conditional_edges(
        "severity_gate",
        conditional_severity_gate,
        {
            "run_models": "run_models",  # Continue path
            "end": END,                   # Early exit path
        },
    )

    # run_models → parallel fan-out
    # All three models run in parallel
    workflow.add_edge("run_models", "ml_model")
    workflow.add_edge("run_models", "llm_model")
    workflow.add_edge("run_models", "human_input")

    # Converge into fusion
    # All three parallel branches must complete before fusion runs
    workflow.add_edge("ml_model", "fusion")
    workflow.add_edge("llm_model", "fusion")
    workflow.add_edge("human_input", "fusion")

    # fusion → confidence_check (solid arrow)
    workflow.add_edge("fusion", "confidence_check")

    # confidence_check → conditional routing
    #   - "high_confidence" → finalize
    #   - "low_confidence" → human_review
    workflow.add_conditional_edges(
        "confidence_check",
        conditional_confidence_routing,
        {
            "high_confidence": "finalize",
            "low_confidence": "human_review",
        },
    )

    # human_review → finalize (solid arrow)
    workflow.add_edge("human_review", "finalize")

    # finalize → END (solid arrow)
    workflow.add_edge("finalize", END)

    return workflow


def compile_workflow(interrupt_before: list = None) -> any:
    """
    Compile the workflow with checkpointer.
    
    Args:
        interrupt_before: List of node names to interrupt before (for HITL)
        
    Returns:
        Compiled graph ready for execution
    """
    workflow = build_workflow()
    memory = MemorySaver()
    
    compile_kwargs = {"checkpointer": memory}
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    
    graph = workflow.compile(**compile_kwargs)

    print("\n--- LangGraph Compiled Successfully! ---")
    print("Graph flow matches the diagram:")
    print("  START → fetch_data → severity_gate → (end OR run_models)")
    print("    → run_models → [human_input, llm_model, ml_model] (parallel)")
    print("    → fusion → confidence_check → (high_confidence OR low_confidence)")
    print("    → (high_confidence → finalize) OR (low_confidence → human_review → finalize)")
    print("    → finalize → END")
    
    return graph

