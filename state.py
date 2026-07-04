"""
STATE
State schema for the Insurance Claim Processing Agent.
Defines the shared state object that flows through the LangGraph graph.
"""
import operator
from typing import TypedDict, List, Optional, Literal, Annotated
from typing_extensions import NotRequired


class ClaimInput(TypedDict):
    """Raw input data submitted for a claim."""
    claim_id: str
    policy_number: str
    claimant_name: str
    policy_status: str                 # "active" | "expired" | "lapsed"
    policy_expiry_date: str            # ISO date string
    claim_type: str                    # e.g. "Health", "Auto", "Property"
    claim_amount: float
    policy_coverage_limit: float
    incident_date: str                 # ISO date string
    claim_filed_date: str              # ISO date string
    submitted_documents: List[str]     # e.g. ["ID Proof", "Hospital Bill"]
    incident_description: str
    claimant_claim_history_count: int  # number of prior claims by this claimant


class DocumentVerificationResult(TypedDict):
    status: Literal["Complete", "Incomplete"]
    missing_documents: List[str]
    provided_documents: List[str]
    notes: str


class EligibilityResult(TypedDict):
    status: Literal["Eligible", "Not Eligible"]
    reasons: List[str]
    policy_active: bool
    within_coverage_limit: bool


class FraudDetectionResult(TypedDict):
    risk_level: Literal["Low", "Medium", "High"]
    fraud_score: int                   # 0-100
    indicators: List[str]
    notes: str


class ClaimState(TypedDict):
    """The full graph state that is passed between nodes."""
    claim: ClaimInput

    # Populated by parallel-branch nodes
    document_result: NotRequired[DocumentVerificationResult]
    eligibility_result: NotRequired[EligibilityResult]
    fraud_result: NotRequired[FraudDetectionResult]

    # Populated by downstream nodes
    claim_summary: NotRequired[str]
    decision: NotRequired[Literal["Approved", "Rejected", "Escalated"]]
    decision_reason: NotRequired[str]
    requires_human_review: NotRequired[bool]
    human_decision: NotRequired[Optional[str]]   # set after human-in-the-loop input
    final_status: NotRequired[str]
    audit_log: NotRequired[Annotated[List[str], operator.add]]
