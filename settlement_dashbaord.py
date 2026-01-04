import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import json  # Used for JSON processing of API response

# --- Configuration for Gemini API (Keep empty) ---
API_KEY = ""  # API key is automatically provided in the environment if empty
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
GROUNDING_TOOL = [{"google_search": {}}]


# --- 1. Mock Data Generation ---

@st.cache_data
def generate_settlement_data(num_rows=1000):
    """Generates synthetic settlement instruction data."""
    np.random.seed(42)

    clients = ['Client A', 'Client B', 'Client C', 'Client D', 'Broker E']
    operation_types = ['Securities Buy', 'Securities Sell', 'Forex Swap', 'Cash Transfer']

    # Core Statuses (High Level)
    statuses = ['Completed', 'Pending', 'Blocked', 'Processing']

    # Sub-Statuses (Detailed reasons)
    sub_statuses = {
        'Completed': ['Settled', 'Partial Settled'],
        'Pending': ['Integrated', 'Integrating', 'Waiting for Counterparty'],
        'Blocked': ['Lack of Provision', 'Regulatory Hold', 'Mismatched Details'],
        'Processing': ['Awaiting Confirmation', 'Internal Review']
    }

    data = []
    for i in range(num_rows):
        instruction_id = f"INST-{i + 1:05d}"
        client = random.choice(clients)
        op_type = random.choice(operation_types)
        status = random.choices(statuses, weights=[0.6, 0.2, 0.15, 0.05], k=1)[0]  # Weighting statuses
        sub_status = random.choice(sub_statuses[status])
        value = np.round(random.uniform(10000, 5000000), 2)

        # Determine settlement date relative to today
        settlement_date = pd.to_datetime('today') + pd.Timedelta(days=random.randint(-15, 5))

        data.append({
            'Instruction ID': instruction_id,
            'Client': client,
            'Operation Type': op_type,
            'Value (USD)': value,
            'Status': status,
            'Sub-Status': sub_status,
            'Settlement Date': settlement_date.strftime('%Y-%m-%d')
        })

    df = pd.DataFrame(data)
    # Add a failure reason column for Blocked instructions
    failure_reasons = {
        'Lack of Provision': 'Insufficient liquid funds or securities inventory to cover the instruction.',
        'Regulatory Hold': 'Transaction flagged due to KYC mismatch or sanction list screening.',
        'Mismatched Details': 'Counterparty reference or amount differs from expectation.'
    }
    df['Failure Reason Detail'] = df.apply(lambda row: failure_reasons.get(row['Sub-Status'], 'N/A'), axis=1)

    return df


# --- 2. LLM Integration Logic (Python side) ---

def ask_llm_about_failure(instruction_id, sub_status, detail):
    """
    Simulates fetching an explanation from the Gemini API.

    NOTE: The 'fetch' call is provided here as a boilerplate structure.
    In a real Streamlit deployment, you would typically use the 'requests'
    library in Python for this, or a specific SDK. For demonstrating
    the Canvas API call structure, we outline the payload here.
    """
    st.info(f"Querying LLM for explanation: **'{sub_status}'**...")

    system_prompt = (
        "You are an expert financial settlement risk analyst. Your task is to provide a clear, concise, "
        "and professional explanation for the given settlement failure reason. If possible, provide "
        "a suggested action to resolve the issue. Ground your response using real-world financial knowledge."
    )
    user_query = (
        f"Explain the settlement failure sub-status: '{sub_status}', based on the detailed reason: "
        f"'{detail}'. What is the typical action required to resolve this issue for instruction {instruction_id}?"
    )

    # --- API Payload Structure (To be executed via a robust client) ---
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "tools": GROUNDING_TOOL,  # Use Google Search grounding
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # In a real environment, you would use requests.post or a similar method here
    # with robust retry logic (e.g., exponential backoff).
    # Since we cannot execute a live network request, we mock the result after a brief delay.

    with st.spinner("Analyzing complexity and generating grounded explanation..."):
        time.sleep(2)  # Simulate network latency

        # --- Mocked API Response for demonstration ---
        if "Lack of Provision" in sub_status:
            generated_text = (
                "The instruction failed due to a **Lack of Provision**. This typically means the necessary "
                "assets (cash or securities) were not available in the settlement account at the time of matching. "
                "This is a common liquidity risk. **Suggested Action:** The client must immediately fund the "
                "account with the required amount/assets, and the instruction should be re-released for settlement."
            )
        elif "Regulatory Hold" in sub_status:
            generated_text = (
                "The instruction is subject to a **Regulatory Hold**. This often results from a failed Know Your Customer (KYC) "
                "or Anti-Money Laundering (AML) screening, potentially due to a mismatch in counterparty data or a sanction flag. "
                "**Suggested Action:** The compliance team must review the alert, clear the flag, or request updated documentation from the client/counterparty."
            )
        else:
            generated_text = f"Explanation for '{sub_status}' is complex. Searching for detailed financial context... (Mock LLM response for other statuses.)"

        st.success("Analysis Complete!")

        return generated_text, ["Source 1: Financial News Today", "Source 2: SEC Guidelines"]  # Mock sources


# --- 3. Main Streamlit Dashboard UI ---

def main_dashboard():
    # Load data
    df = generate_settlement_data()

    # --- Streamlit UI Setup ---
    st.set_page_config(layout="wide", page_title="Settlement Failure Dashboard")
    st.title("Settlement Failure Analytics Dashboard")
    st.markdown("Monitor and analyze blocked and pending financial instructions.")

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Instructions")

    # Client Filter
    client_options = ['All'] + sorted(df['Client'].unique().tolist())
    selected_client = st.sidebar.selectbox("Select Client", client_options)

    # Operation Type Filter
    op_options = ['All'] + sorted(df['Operation Type'].unique().tolist())
    selected_op = st.sidebar.selectbox("Select Operation Type", op_options)

    # Status Filter
    status_options = ['All'] + sorted(df['Status'].unique().tolist())
    selected_status = st.sidebar.selectbox("Select Status", status_options)

    # Sub-Status Filter (Dynamic based on Status)
    filtered_df = df.copy()
    if selected_client != 'All':
        filtered_df = filtered_df[filtered_df['Client'] == selected_client]
    if selected_op != 'All':
        filtered_df = filtered_df[filtered_df['Operation Type'] == selected_op]
    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['Status'] == selected_status]

    sub_status_options = ['All'] + sorted(filtered_df['Sub-Status'].unique().tolist())
    selected_sub_status = st.sidebar.selectbox("Select Sub-Status", sub_status_options)

    if selected_sub_status != 'All':
        filtered_df = filtered_df[filtered_df['Sub-Status'] == selected_sub_status]

    st.sidebar.markdown(f"**Showing {len(filtered_df)} Instructions**")
    st.sidebar.markdown("---")

    # --- 4. Main Content: Metrics ---
    st.subheader("Key Performance Indicators")

    total_instructions = len(df)
    total_blocked = df[df['Status'] == 'Blocked']
    blocked_count = len(total_blocked)
    blocked_value = total_blocked['Value (USD)'].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Instructions", f"{total_instructions:,}")
    col2.metric("Total Blocked Instructions", f"{blocked_count:,}", delta=f"{blocked_count / total_instructions:.1%}")
    col3.metric("Blocked Value (USD)", f"${blocked_value:,.0f}")
    col4.metric("Pending/Processing Value",
                f"${df[df['Status'].isin(['Pending', 'Processing'])]['Value (USD)'].sum():,.0f}")

    st.markdown("---")

    # --- 5. Main Content: Charts ---
    st.subheader("Instruction Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("**Blocked Sub-Status Breakdown**")
        blocked_breakdown = total_blocked.groupby('Sub-Status').size().reset_index(name='Count')
        st.bar_chart(blocked_breakdown.set_index('Sub-Status'))

    with chart_col2:
        st.write("**Value by Operation Type (Filtered)**")
        op_value = filtered_df.groupby('Operation Type')['Value (USD)'].sum().reset_index()
        st.dataframe(op_value, hide_index=True, use_container_width=True)

    # --- 6. LLM Q&A Tool ---
    st.markdown("---")
    st.subheader("🤖 Failure Explanation Tool (LLM Powered)")

    # Dropdown for selecting a specific BLOCKed instruction
    blocked_instructions = total_blocked[['Instruction ID', 'Sub-Status', 'Failure Reason Detail']].drop_duplicates()

    if blocked_instructions.empty:
        st.warning("No 'Blocked' instructions found in the current dataset to analyze.")
    else:
        # Create a combined display string for the select box
        blocked_instructions['Display'] = blocked_instructions['Instruction ID'] + " (" + blocked_instructions[
            'Sub-Status'] + ")"

        selected_instruction_display = st.selectbox(
            "Select a Blocked Instruction for detailed analysis:",
            blocked_instructions['Display']
        )

        # Get the actual data for the selected instruction
        selected_row = blocked_instructions[blocked_instructions['Display'] == selected_instruction_display].iloc[0]

        if st.button("Ask LLM: Why is this instruction blocked?", type="primary"):
            explanation, sources = ask_llm_about_failure(
                selected_row['Instruction ID'],
                selected_row['Sub-Status'],
                selected_row['Failure Reason Detail']
            )

            st.markdown("### LLM Analysis & Resolution")
            st.code(explanation, language='markdown')
            st.caption("Grounding Sources (Mock): " + ", ".join(sources))

    # --- 7. Filtered Raw Data ---
    st.markdown("---")
    st.subheader("Filtered Instruction List")
    st.dataframe(filtered_df, use_container_width=True)


if __name__ == '__main__':
    main_dashboard()