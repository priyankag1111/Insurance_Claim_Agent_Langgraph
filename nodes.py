"""
NODES
Agent / node implementations for the Insurance Claim Processing Agent.

Each node:
  1. Tries to use the LLM (GPT-4o-mini) with a structured-output style prompt.
  2. Falls back to transparent rule-based logic if no LLM is configured or
     the LLM call fails to parse — this keeps the demo fully runnable
     without an API key and keeps behavior auditable/deterministic for grading.

IMPORTANT: Nodes return only PARTIAL state dicts (deltas), not the full state.
This is required because document_verification, eligibility_check, and
fraud_detection run in PARALLEL (fan-out from START) — if each node returned
the whole state object (including the unmodified 'claim' key), LangGraph
would see multiple concurrent writes to the same key and raise
InvalidUpdateError. Returning only the keys each node actually changes avoids
that collision entirely.
"""
import json
import re
from datetime import datetime
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from state import ClaimState

REQUIRED_DOCS_BY_TYPE = {
    "Health": ["ID Proof", "Hospital Bill", "Discharge Summary", "Doctor Prescription"],
    "Auto": ["ID Proof", "Vehicle Registration", "Repair Estimate", "Police Report"],
    "Property": ["ID Proof", "Ownership Proof", "Damage Photos", "Repair Estimate"],
    "Life": ["ID Proof", "Death Certificate", "Policy Document", "Nominee ID Proof"],
}

HIGH_VALUE_THRESHOLD = 500_000  # currency units
SUSPICIOUS_FILING_DELAY_DAYS = 60
SUSPICIOUS_CLAIM_RATIO = 0.9  # claim amount / coverage limit


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# 1. Document Verification Agent
# ---------------------------------------------------------------------------
def document_verification_node(state: ClaimState) -> dict:
    claim = state["claim"]
    required = REQUIRED_DOCS_BY_TYPE.get(claim["claim_type"], ["ID Proof"])
    provided = claim.get("submitted_documents", [])

    llm = get_llm()
    result = None
    if llm:
        try:
            sys_prompt = (
                "You are a Document Verification Agent for an insurance company. "
                "Given the required documents and submitted documents, determine which "
                "required documents are missing. Respond ONLY with strict JSON in this "
                'schema: {"status": "Complete"|"Incomplete", "missing_documents": [..], '
                '"provided_documents": [..], "notes": "short note"}'
            )
            user_prompt = (
                f"Claim type: {claim['claim_type']}\n"
                f"Required documents: {required}\n"
                f"Submitted documents: {provided}"
            )
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            result = _extract_json(resp.content)
        except Exception:
            result = None

    if result is None:
        missing = [d for d in required if d not in provided]
        result = {
            "status": "Complete" if not missing else "Incomplete",
            "missing_documents": missing,
            "provided_documents": provided,
            "notes": "Rule-based check against required document checklist for claim type "
                     f"'{claim['claim_type']}'.",
        }

    log_line = f"[Document Verification] status={result['status']} missing={result.get('missing_documents')}"
    return {"document_result": result, "audit_log": [log_line]}


# ---------------------------------------------------------------------------
# 2. Eligibility Check Agent
# ---------------------------------------------------------------------------
def eligibility_check_node(state: ClaimState) -> dict:
    claim = state["claim"]

    policy_active = claim.get("policy_status", "").lower() == "active"
    try:
        expiry = datetime.fromisoformat(claim["policy_expiry_date"])
        incident = datetime.fromisoformat(claim["incident_date"])
        not_expired_at_incident = incident <= expiry
    except Exception:
        not_expired_at_incident = True

    within_limit = claim["claim_amount"] <= claim["policy_coverage_limit"]

    llm = get_llm()
    result = None
    if llm:
        try:
            sys_prompt = (
                "You are an Eligibility Check Agent for an insurance company. "
                "Evaluate whether the claim is eligible based on policy status, "
                "whether the policy was active at the time of incident, and whether "
                "the claim amount is within the coverage limit. Respond ONLY with strict "
                'JSON: {"status": "Eligible"|"Not Eligible", "reasons": [..], '
                '"policy_active": true|false, "within_coverage_limit": true|false}'
            )
            user_prompt = (
                f"Policy status: {claim['policy_status']}\n"
                f"Policy expiry date: {claim['policy_expiry_date']}\n"
                f"Incident date: {claim['incident_date']}\n"
                f"Claim amount: {claim['claim_amount']}\n"
                f"Policy coverage limit: {claim['policy_coverage_limit']}"
            )
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            result = _extract_json(resp.content)
        except Exception:
            result = None

    if result is None:
        reasons = []
        if not policy_active:
            reasons.append(f"Policy status is '{claim.get('policy_status')}', not active.")
        if not not_expired_at_incident:
            reasons.append("Policy had expired before the incident date.")
        if not within_limit:
            reasons.append("Claim amount exceeds policy coverage limit.")

        eligible = policy_active and not_expired_at_incident and within_limit
        result = {
            "status": "Eligible" if eligible else "Not Eligible",
            "reasons": reasons if reasons else ["Policy active, valid at incident date, within coverage limit."],
            "policy_active": policy_active and not_expired_at_incident,
            "within_coverage_limit": within_limit,
        }

    log_line = f"[Eligibility Check] status={result['status']} reasons={result.get('reasons')}"
    return {"eligibility_result": result, "audit_log": [log_line]}


# ---------------------------------------------------------------------------
# 3. Fraud Detection Agent
# ---------------------------------------------------------------------------
def fraud_detection_node(state: ClaimState) -> dict:
    claim = state["claim"]

    indicators = []
    score = 0

    ratio = claim["claim_amount"] / max(claim["policy_coverage_limit"], 1)
    if ratio >= SUSPICIOUS_CLAIM_RATIO:
        indicators.append("Claim amount is very close to or exceeds the policy coverage limit.")
        score += 35

    try:
        filed = datetime.fromisoformat(claim["claim_filed_date"])
        incident = datetime.fromisoformat(claim["incident_date"])
        delay_days = (filed - incident).days
        if delay_days > SUSPICIOUS_FILING_DELAY_DAYS:
            indicators.append(f"Claim filed {delay_days} days after incident (unusually delayed).")
            score += 20
        if delay_days < 0:
            indicators.append("Claim filed date is before incident date (data inconsistency).")
            score += 30
    except Exception:
        pass

    if claim.get("claimant_claim_history_count", 0) >= 3:
        indicators.append("Claimant has a history of 3 or more prior claims.")
        score += 20

    if claim["claim_amount"] >= HIGH_VALUE_THRESHOLD:
        indicators.append("High-value claim amount requiring additional scrutiny.")
        score += 15

    vague_terms = ["unknown", "not sure", "unclear", "n/a", ""]
    if claim.get("incident_description", "").strip().lower() in vague_terms:
        indicators.append("Incident description is vague or missing.")
        score += 15

    score = min(score, 100)
    rule_based_level = "Low" if score < 30 else "Medium" if score < 60 else "High"

    llm = get_llm()
    result = None
    if llm:
        try:
            sys_prompt = (
                "You are a Fraud Detection Agent for an insurance company. Analyze the "
                "claim details for potential fraud indicators such as: claim amount close "
                "to coverage limit, delayed filing, frequent claim history, vague incident "
                "description, or unusually high value. Respond ONLY with strict JSON: "
                '{"risk_level": "Low"|"Medium"|"High", "fraud_score": 0-100, '
                '"indicators": [..], "notes": "short note"}'
            )
            user_prompt = (
                f"Claim amount: {claim['claim_amount']}\n"
                f"Coverage limit: {claim['policy_coverage_limit']}\n"
                f"Incident date: {claim['incident_date']}\n"
                f"Claim filed date: {claim['claim_filed_date']}\n"
                f"Prior claims by claimant: {claim.get('claimant_claim_history_count', 0)}\n"
                f"Incident description: {claim.get('incident_description', '')}\n"
                f"(Rule-based reference score: {score}, indicators found: {indicators})"
            )
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            result = _extract_json(resp.content)
        except Exception:
            result = None

    if result is None:
        result = {
            "risk_level": rule_based_level,
            "fraud_score": score,
            "indicators": indicators if indicators else ["No significant fraud indicators detected."],
            "notes": "Rule-based fraud heuristic scoring.",
        }

    log_line = f"[Fraud Detection] risk={result['risk_level']} score={result.get('fraud_score')}"
    return {"fraud_result": result, "audit_log": [log_line]}


# ---------------------------------------------------------------------------
# 4. Claim Summary Agent
# ---------------------------------------------------------------------------
def claim_summary_node(state: ClaimState) -> dict:
    claim = state["claim"]
    doc = state.get("document_result", {})
    elig = state.get("eligibility_result", {})
    fraud = state.get("fraud_result", {})

    llm = get_llm()
    summary = None
    if llm:
        try:
            sys_prompt = (
                "You are a Claim Summary Agent for an insurance company. Write a concise, "
                "professional 4-6 sentence summary of this claim for a claims adjuster, "
                "covering: claim overview, document verification outcome, eligibility "
                "outcome, and fraud risk outcome. Do not make the final approval decision."
            )
            user_prompt = (
                f"Claim: {json.dumps(claim, default=str)}\n"
                f"Document verification: {json.dumps(doc)}\n"
                f"Eligibility: {json.dumps(elig)}\n"
                f"Fraud detection: {json.dumps(fraud)}"
            )
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            summary = resp.content.strip()
        except Exception:
            summary = None

    if summary is None:
        summary = (
            f"Claim {claim['claim_id']} filed by {claim['claimant_name']} for a "
            f"{claim['claim_type']} incident on {claim['incident_date']}, requesting "
            f"{claim['claim_amount']:,.2f} against a coverage limit of "
            f"{claim['policy_coverage_limit']:,.2f}. Document verification result: "
            f"{doc.get('status', 'Unknown')}"
            + (f" (missing: {', '.join(doc.get('missing_documents', []))})." if doc.get('missing_documents') else ".")
            + f" Eligibility result: {elig.get('status', 'Unknown')}. "
            + f"Fraud risk assessment: {fraud.get('risk_level', 'Unknown')} "
            f"(score {fraud.get('fraud_score', 'N/A')}/100)."
        )

    return {"claim_summary": summary, "audit_log": ["[Claim Summary] Generated summary."]}


# ---------------------------------------------------------------------------
# Decision routing logic (used by conditional edge, not an LLM node itself)
# ---------------------------------------------------------------------------
def decision_node(state: ClaimState) -> dict:
    doc = state.get("document_result", {})
    elig = state.get("eligibility_result", {})
    fraud = state.get("fraud_result", {})
    claim = state["claim"]

    # Hard rejects
    if doc.get("status") == "Incomplete":
        decision = "Rejected"
        reason = f"Missing required documents: {', '.join(doc.get('missing_documents', []))}.."
        requires_human = False

    elif elig.get("status") == "Not Eligible":
        decision = "Rejected"
        reason = "Claim failed eligibility check: " + "; ".join(elig.get("reasons", []))
        requires_human = False

    # High fraud risk or high value -> escalate to human
    elif fraud.get("risk_level") == "High":
        decision = "Escalated"
        reason = "High fraud risk detected: " + "; ".join(fraud.get("indicators", []))
        requires_human = True

    elif fraud.get("risk_level") == "Medium":
        decision = "Escalated"
        reason = "Medium fraud risk requires human judgment: " + "; ".join(fraud.get("indicators", []))
        requires_human = True

    elif claim["claim_amount"] >= HIGH_VALUE_THRESHOLD:
        decision = "Escalated"
        reason = (
            f"High-value claim ({claim['claim_amount']:,.2f}) requires manual approval "
            "even though documents and eligibility are in order."
        )
        requires_human = True

    else:
        decision = "Approved"
        reason = "All documents verified, policy eligibility confirmed, and fraud risk is low."
        requires_human = False

    log_line = f"[Decision Engine] decision={decision} human_review={requires_human}"
    return {
        "decision": decision,
        "decision_reason": reason,
        "requires_human_review": requires_human,
        "audit_log": [log_line],
    }


# ---------------------------------------------------------------------------
# 5. Human Approval Agent (Human-in-the-Loop)
# ---------------------------------------------------------------------------
def human_approval_node(state: ClaimState) -> dict:
    """
    This node marks the claim as pending human review. In the LangGraph app,
    execution is interrupted BEFORE this node when human_review is required
    (see graph.py's interrupt_before).
    The Streamlit UI collects the human's
    decision and resumes the graph, at which point this node finalizes status
    based on the human's input stored in state['human_decision'].
    """
    human_decision = state.get("human_decision")

    if human_decision is None:
        # Graph resumed without a human decision yet (shouldn't normally happen)
        return {
            "final_status": "Pending Human Review",
            "audit_log": ["[Human Approval] Awaiting human reviewer decision."],
        }

    if human_decision.lower() == "approve":
        final_status = "Approved (by Human Reviewer)"
    elif human_decision.lower() == "reject":
        final_status = "Rejected (by Human Reviewer)"
    else:
        final_status = "Pending Human Review"

    log_line = f"[Human Approval] Reviewer decision recorded: {human_decision}"
    return {"final_status": final_status, "audit_log": [log_line]}


def finalize_auto_node(state: ClaimState) -> dict:
    """Finalizes status for claims that did NOT require human review."""
    final_status = state["decision"]
    return {"final_status": final_status, "audit_log": [f"[Finalize] final_status={final_status}"]}
