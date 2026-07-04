#!pip install -q streamlit
"""
Streamlit application for the Insurance Claim Processing Agent (LangGraph).

Run with:
    streamlit run app.py

Optional: set GROQ_API_KEY in the environment (or sidebar) to use GROQ LLM
for the agent reasoning steps. Without a key, the app runs on transparent
rule-based fallbacks so it is always demoable.
"""
import os
import uuid
from dotenv import load_dotenv

import streamlit as st

from graph import claim_graph
from sample_claims import SAMPLE_CLAIMS
from state import ClaimInput

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Insurance Claim Processing Agent", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar: API key + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Claim Agent")
    st.caption("LangGraph-powered insurance claim automation")

    api_key_input = st.text_input(
        "GROQ API Key (optional)",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="If left empty, the app uses rule-based fallback logic for every agent so it still runs end-to-end.",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    st.divider()
    st.markdown("**Workflow**")
    st.markdown(
        "- 📄 Document Verification\n"
        "- ✅ Eligibility Check\n"
        "- 🕵️ Fraud Detection\n"
        "- 📝 Claim Summary\n"
        "- 👤 Human Approval (if flagged)"
    )
    st.divider()
    st.markdown(
        "Parallel checks run for **documents, eligibility, and fraud**, "
        "then results merge into a summary and decision engine that routes "
        "the claim to **Approve / Reject / Escalate**."
    )

st.title("Insurance Claim Processing Agent")
st.caption("Automates document verification, eligibility checks, fraud detection, summarization, and approval routing using LangGraph.")

tab_submit, tab_samples, tab_about = st.tabs(["📋 Submit / Review Claim", "🧪 Sample Scenarios", "ℹ️ About the Graph"])


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "awaiting_human" not in st.session_state:
    st.session_state.awaiting_human = False
if "form_defaults" not in st.session_state:
    st.session_state.form_defaults = list(SAMPLE_CLAIMS.values())[0]


def run_claim(claim: ClaimInput):
    """Starts a new graph run for the given claim."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"claim": claim, "audit_log": []}
    result = claim_graph.invoke(initial_state, config=config)

    st.session_state.thread_id = thread_id
    st.session_state.graph_state = result
    st.session_state.awaiting_human = bool(result.get("requires_human_review")) and "final_status" not in result


def resume_with_human_decision(decision: str):
    """Resumes an interrupted graph run with the human reviewer's decision."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    claim_graph.update_state(config, {"human_decision": decision})
    result = claim_graph.invoke(None, config=config)
    st.session_state.graph_state = result
    st.session_state.awaiting_human = False


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_result(result: dict):
    claim = result["claim"]

    decision = result.get("decision", "Pending")
    final_status = result.get("final_status")

    color_map = {
        "Approved": "🟢",
        "Rejected": "🔴",
        "Escalated": "🟡",
    }
    st.subheader(f"{color_map.get(decision, '⚪')} Decision: {decision}")
    if result.get("decision_reason"):
        st.write(result["decision_reason"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📄 Document Verification**")
        doc = result.get("document_result", {})
        st.write(f"Status: **{doc.get('status')}**")
        if doc.get("missing_documents"):
            st.error(f"Missing: {', '.join(doc['missing_documents'])}")
        else:
            st.success("All required documents present.")

    with col2:
        st.markdown("**✅ Eligibility Check**")
        elig = result.get("eligibility_result", {})
        st.write(f"Status: **{elig.get('status')}**")
        for r in elig.get("reasons", []):
            st.caption(f"• {r}")

    with col3:
        st.markdown("**🕵️ Fraud Detection**")
        fraud = result.get("fraud_result", {})
        risk = fraud.get("risk_level", "Unknown")
        risk_color = {"Low": "success", "Medium": "warning", "High": "error"}.get(risk, "info")
        getattr(st, risk_color)(f"Risk: {risk} (score {fraud.get('fraud_score', 'N/A')}/100)")
        for ind in fraud.get("indicators", []):
            st.caption(f"• {ind}")

    st.markdown("**📝 Claim Summary**")
    st.info(result.get("claim_summary", "No summary generated."))

    # Human-in-the-loop
    if result.get("requires_human_review"):
        if final_status:
            st.success(f"✅ Final Status: **{final_status}**")
        else:
            st.warning("⚠️ This claim has been escalated and requires **human approval**.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Approve Claim", use_container_width=True, type="primary"):
                    resume_with_human_decision("approve")
                    st.rerun()
            with c2:
                if st.button("❌ Reject Claim", use_container_width=True):
                    resume_with_human_decision("reject")
                    st.rerun()
    else:
        st.success(f"✅ Final Status: **{final_status or decision}**")

    with st.expander("🧾 Full Audit Log"):
        for line in result.get("audit_log", []):
            st.text(line)

    with st.expander("🔍 Raw Claim Data"):
        st.json(claim)


# ---------------------------------------------------------------------------
# Tab 1: Submit / Review Claim
# ---------------------------------------------------------------------------
with tab_submit:
    st.markdown("### Submit a New Claim")

    defaults = st.session_state.form_defaults
    doc_options = [
        "ID Proof", "Hospital Bill", "Discharge Summary", "Doctor Prescription",
        "Vehicle Registration", "Repair Estimate", "Police Report",
        "Ownership Proof", "Damage Photos", "Death Certificate",
        "Policy Document", "Nominee ID Proof",
    ]

    with st.form("claim_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            claim_id = st.text_input("Claim ID", value=defaults["claim_id"])
            policy_number = st.text_input("Policy Number", value=defaults["policy_number"])
            claimant_name = st.text_input("Claimant Name", value=defaults["claimant_name"])
            claim_type = st.selectbox(
                "Claim Type", ["Health", "Auto", "Property", "Life"],
                index=["Health", "Auto", "Property", "Life"].index(defaults["claim_type"]),
            )

        with c2:
            policy_status = st.selectbox(
                "Policy Status", ["active", "expired", "lapsed"],
                index=["active", "expired", "lapsed"].index(defaults["policy_status"]),
            )
            policy_expiry_date = st.text_input("Policy Expiry Date (YYYY-MM-DD)", value=defaults["policy_expiry_date"])
            incident_date = st.text_input("Incident Date (YYYY-MM-DD)", value=defaults["incident_date"])
            claim_filed_date = st.text_input("Claim Filed Date (YYYY-MM-DD)", value=defaults["claim_filed_date"])

        with c3:
            claim_amount = st.number_input("Claim Amount", min_value=0.0, value=float(defaults["claim_amount"]), step=1000.0)
            policy_coverage_limit = st.number_input("Policy Coverage Limit", min_value=0.0, value=float(defaults["policy_coverage_limit"]), step=1000.0)
            claimant_claim_history_count = st.number_input("Claimant Prior Claims Count", min_value=0, value=int(defaults["claimant_claim_history_count"]), step=1)

        submitted_documents = st.multiselect("Submitted Documents", doc_options, default=defaults["submitted_documents"])
        incident_description = st.text_area("Incident Description", value=defaults["incident_description"])

        submitted = st.form_submit_button("🚀 Process Claim", type="primary", use_container_width=True)

    if submitted:
        claim: ClaimInput = {
            "claim_id": claim_id,
            "policy_number": policy_number,
            "claimant_name": claimant_name,
            "policy_status": policy_status,
            "policy_expiry_date": policy_expiry_date,
            "claim_type": claim_type,
            "claim_amount": claim_amount,
            "policy_coverage_limit": policy_coverage_limit,
            "incident_date": incident_date,
            "claim_filed_date": claim_filed_date,
            "submitted_documents": submitted_documents,
            "incident_description": incident_description,
            "claimant_claim_history_count": claimant_claim_history_count,
        }
        with st.spinner("Running claim through the LangGraph agent workflow..."):
            run_claim(claim)

    st.divider()
    if st.session_state.graph_state:
        render_result(st.session_state.graph_state)
    else:
        st.caption("Fill out the form above (or load a sample scenario from the next tab) and click **Process Claim**.")


# ---------------------------------------------------------------------------
# Tab 2: Sample Scenarios
# ---------------------------------------------------------------------------
with tab_samples:
    st.markdown("### Load a Sample Scenario")
    st.caption("These match the example test cases from the assignment. Click one to load it into the form and auto-process it.")

    for name, claim in SAMPLE_CLAIMS.items():
        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{name}**")
                st.caption(
                    f"{claim['claim_type']} claim • Amount: {claim['claim_amount']:,.0f} / "
                    f"Limit: {claim['policy_coverage_limit']:,.0f} • Policy: {claim['policy_status']} • "
                    f"Docs: {len(claim['submitted_documents'])} submitted"
                )
            with cols[1]:
                if st.button("Load & Run", key=f"run_{claim['claim_id']}", use_container_width=True):
                    st.session_state.form_defaults = claim
                    with st.spinner("Running claim through the LangGraph agent workflow..."):
                        run_claim(claim)
                    st.rerun()

    if st.session_state.graph_state:
        st.divider()
        st.markdown("### Result")
        render_result(st.session_state.graph_state)


# ---------------------------------------------------------------------------
# Tab 3: About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("### Graph Architecture")
    st.markdown(
        """
This app is built with **LangGraph** and uses 5+ nodes wired as a StateGraph:

```
                         START
                           |
        -------------------------------------------
        |                  |                       |
Document Verify      Eligibility Check       Fraud Detection      <- PARALLEL fan-out
        |                  |                       |
        -------------------------------------------
                           |
                   Claim Summary Agent
                           |
                    Decision Engine
                           |
              (conditional edge on outcome)
             /             |               \
      Rejected         Approved          Escalated
          |                |                 |
          |                |        Human Approval Agent
          |                |         (interrupt_before)
          +---------  Finalize  --------------+
                           |
                          END
```

**Key technical elements**
- **Parallel execution**: `document_verification`, `eligibility_check`, and `fraud_detection` all fan out directly from `START` and run concurrently; their results fan back in to `claim_summary`.
- **Conditional edges**: `add_conditional_edges` routes each claim to either the `human_approval` node or straight to `finalize_auto`, based on the decision engine's output.
- **Human-in-the-loop**: the graph is compiled with `interrupt_before=["human_approval"]` and a `MemorySaver` checkpointer, so execution pauses and waits for a real reviewer decision (captured in the Streamlit UI) before resuming via `update_state` + `invoke(None, config)`.
- **LLM + fallback**: every agent node first attempts a GPT-4o-mini call for its reasoning step, and falls back to transparent rule-based logic if no API key is configured or the call fails — so the app is always runnable end-to-end.
"""
    )

    st.markdown("### Decision Rules Summary")
    st.markdown(
        """
| Condition | Outcome |
|---|---|
| Required documents missing | **Rejected** |
| Policy inactive / expired at incident date / over coverage limit | **Rejected** |
| Fraud risk = High or Medium | **Escalated** (Human Approval) |
| Claim amount ≥ high-value threshold (500,000) | **Escalated** (Human Approval) |
| Documents complete + Eligible + Low fraud risk + below high-value threshold | **Approved** |
"""
    )
