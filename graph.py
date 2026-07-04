"""
GRAPH
LangGraph graph definition for the Insurance Claim Processing Agent.

Graph topology:

              +------------------------+
              |         START          |
              +------------------------+
                         |
      ---------------------------------------------
      |                  |                          |
Document Verify   Eligibility Check           Fraud Detection      <- run in PARALLEL
      |                  |                          |
      ---------------------------------------------
                         |
                 Claim Summary Agent
                         |
                  Decision Engine
                         |
             (conditional edge)
              /          |          \
       Rejected      Approved      Escalated
          |              |              |
          |              |     Human Approval Agent (interrupt before)
          |              |              |
          +------ Finalize Node --------+
                         |
                        END
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import ClaimState
from nodes import (
    document_verification_node,
    eligibility_check_node,
    fraud_detection_node,
    claim_summary_node,
    decision_node,
    human_approval_node,
    finalize_auto_node,
)


def route_after_decision(state: ClaimState) -> str:
    """Conditional edge: send to human approval if flagged, else finalize directly."""
    if state.get("requires_human_review"):
        return "human_approval"
    return "finalize_auto"


def build_graph():
    graph = StateGraph(ClaimState)

    # Register nodes
    graph.add_node("document_verification", document_verification_node)
    graph.add_node("eligibility_check", eligibility_check_node)
    graph.add_node("fraud_detection", fraud_detection_node)
    graph.add_node("claim_summary", claim_summary_node)
    graph.add_node("decision_engine", decision_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalize_auto", finalize_auto_node)

    # Parallel fan-out from START into the three verification agents
    graph.add_edge(START, "document_verification")
    graph.add_edge(START, "eligibility_check")
    graph.add_edge(START, "fraud_detection")

    # Fan-in: all three parallel branches must complete before summary runs
    graph.add_edge("document_verification", "claim_summary")
    graph.add_edge("eligibility_check", "claim_summary")
    graph.add_edge("fraud_detection", "claim_summary")

    graph.add_edge("claim_summary", "decision_engine")

    # Conditional routing based on decision outcome
    graph.add_conditional_edges(
        "decision_engine",
        route_after_decision,
        {
            "human_approval": "human_approval",
            "finalize_auto": "finalize_auto",
        },
    )

    graph.add_edge("human_approval", END)
    graph.add_edge("finalize_auto", END)

    # Checkpointer enables human-in-the-loop: we interrupt BEFORE human_approval
    # so the app can pause, show the case to a human reviewer, collect their
    # decision, and then resume the graph.
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
    return compiled


# Singleton compiled graph for reuse across the Streamlit app
claim_graph = build_graph()
