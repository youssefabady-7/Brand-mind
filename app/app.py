"""BrandMind — Multi-Tenant Agentic Customer Service Demo UI.

Streamlit application providing:
1. Multi-tenant brand switching (Lumora vs UrbanKicks)
2. Interactive chat conversation with session state
3. Real-time telemetry inspector:
   - Intent badge & confidence
   - Language / Egyptian dialect indicator
   - Deterministic tool / decision execution cards
   - Retrieved ChromaDB RAG source citations with cosine similarity
   - Safety escalation triggers & reason codes
4. Quick demo prompt scenarios for instant testing and evaluation
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import uuid

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import streamlit as st

from src.agents.agent import AgentResponse, BrandMindAgent

# Page Setup
st.set_page_config(
    page_title="BrandMind | AI Customer Service Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .brand-header-lumora {
        background: linear-gradient(135deg, #1b3b36 0%, #2e5950 100%);
        color: #f0f7f4;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #52b788;
    }
    .brand-header-urbankicks {
        background: linear-gradient(135deg, #2b1d0c 0%, #4a3419 100%);
        color: #fff6eb;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #ff9f1c;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-lumora { background-color: #d8f3dc; color: #1b4332; }
    .badge-urbankicks { background-color: #ffe5d9; color: #7f2c00; }
    .badge-info { background-color: #e0f2fe; color: #0369a1; }
    .badge-warn { background-color: #fef3c7; color: #b45309; }
    .badge-danger { background-color: #fee2e2; color: #b91c1c; }
    .badge-success { background-color: #dcfce7; color: #15803d; }
    .telemetry-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "agent" not in st.session_state:
    st.session_state.agent = BrandMindAgent()


# Brand Metadata Dictionary
BRANDS = {
    "lumora": {
        "display_name": "Lumora Skincare",
        "tagline": "Minimalist, dermatologically mindful skincare in Egypt.",
        "tone": "Calm, empathetic, expert, and reassuring",
        "return_window": "14 days",
        "condition_rule": "Strictly unopened & seal intact for hygiene",
        "header_class": "brand-header-lumora",
        "badge_class": "badge-lumora",
        "greeting_en": "Welcome to Lumora. How may I assist you with your skincare routine or order today?",
        "greeting_ar": "أهلاً بك في لومورا. إزاي أقدر أساعدك النهاردة بخصوص منتجات العناية بالبشرة أو طلبك؟",
        "avatar": "🌿",
    },
    "urbankicks": {
        "display_name": "UrbanKicks Egypt",
        "tagline": "Youthful, energetic streetwear & limited sneakers.",
        "tone": "Energetic, hype, direct, and street-smart",
        "return_window": "14 days",
        "condition_rule": "Unworn, in original box with tags attached",
        "header_class": "brand-header-urbankicks",
        "badge_class": "badge-urbankicks",
        "greeting_en": "What's up! Welcome to UrbanKicks. Looking for the latest drops or checking an order?",
        "greeting_ar": "يا هلا بيك في أوربان كيكس! بتدور على دروب جديد ولا عاوز تتابع أوردرك؟",
        "avatar": "👟",
    },
}

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ BrandMind Control")

    selected_brand_key = st.selectbox(
        "Active Brand (Multi-Tenancy):",
        options=["lumora", "urbankicks"],
        format_func=lambda k: BRANDS[k]["display_name"],
        index=0,
    )
    brand_meta = BRANDS[selected_brand_key]

    # Connection Mode Toggle
    engine_mode = st.radio(
        "Execution Mode:",
        ["Direct Embedded Agent", "FastAPI Service (Port 8000)"],
        index=0,
        help="Direct mode runs the in-process Python pipeline. FastAPI mode sends HTTP POST requests to http://127.0.0.1:8000/chat.",
    )

    st.divider()

    # Brand Details Card
    st.markdown(
        f"""
        <div style="background-color: #f1f5f9; padding: 0.8rem; border-radius: 8px; font-size: 0.88rem;">
            <b>Brand:</b> {brand_meta['display_name']}<br/>
            <b>Tone:</b> {brand_meta['tone']}<br/>
            <b>Return Window:</b> {brand_meta['return_window']}<br/>
            <b>Condition Rule:</b> {brand_meta['condition_rule']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Pre-built Demo Scenarios
    st.subheader("🎯 Demo Quick-Triggers")
    st.caption("Click any scenario to simulate customer interaction:")

    demo_scenarios = [
        ("FAQ (Shipping Cairo)", "What are your shipping rates and delivery times for Cairo?"),
        ("Tool (Track Order LUM-1001)", "Can you check the status of my order LUM-1001?"),
        ("Policy: Eligible Return (10d)", "I bought a Vitamin C Serum 10 days ago. It is completely unopened with seal intact. Can I return it?"),
        ("Policy: Ineligible (18d)", "I bought a cleanser 18 days ago and opened it. Can I return it?"),
        ("Safety Escalation (Allergy)", "I used the serum yesterday and now my skin is burning and I have a severe allergic reaction!"),
        ("Human Escalation (Angry)", "This is the third time you delay! I am furious and want to speak with a human supervisor right now!"),
        ("Egyptian Arabic (Order Status)", "لو سمحت عاوز اعرف طلبي LUM-1001 وصل لفين؟"),
        ("Egyptian Arabic (Shipping FAQ)", "هو الشحن بياخد كام يوم للإسكندرية وبكام؟"),
    ]

    for label, prompt_text in demo_scenarios:
        if st.button(label, use_container_width=True):
            st.session_state["preset_prompt"] = prompt_text

    st.divider()

    if st.button("🔄 Clear Conversation & Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.caption(f"Session ID: `{st.session_state.session_id}`")


# Function to invoke agent via direct in-memory or FastAPI
def run_agent_turn(brand_id: str, prompt: str) -> AgentResponse:
    if "FastAPI" in engine_mode:
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "brand_id": brand_id,
                    "message": prompt,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return AgentResponse(**data)
            else:
                st.error(f"FastAPI error ({resp.status_code}): {resp.text}")
                # Fallback to direct agent
                return st.session_state.agent.handle_message(
                    session_id=st.session_state.session_id,
                    brand_id=brand_id,
                    message=prompt,
                )
        except Exception as ex:
            st.warning(f"Could not connect to FastAPI at port 8000 ({ex}). Falling back to Direct Embedded Agent.")
            return st.session_state.agent.handle_message(
                session_id=st.session_state.session_id,
                brand_id=brand_id,
                message=prompt,
            )
    else:
        return st.session_state.agent.handle_message(
            session_id=st.session_state.session_id,
            brand_id=brand_id,
            message=prompt,
        )


# Main Content Area: Two Column Layout
st.markdown(
    f"""
    <div class="{brand_meta['header_class']}">
        <div style="font-size: 1.6rem; font-weight: 700;">{brand_meta['avatar']} {brand_meta['display_name']}</div>
        <div style="font-size: 0.95rem; opacity: 0.9;">{brand_meta['tagline']}</div>
        <div style="margin-top: 0.5rem;">
            <span class="badge-pill {brand_meta['badge_class']}">Brand Voice Active</span>
            <span class="badge-pill badge-info">Deterministic Policy Engine</span>
            <span class="badge-pill badge-info">ChromaDB RAG Grounding</span>
            <span class="badge-pill badge-success">Egyptian Arabic & English</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_chat, col_telemetry = st.columns([6, 5], gap="medium")

# LEFT COLUMN: Interactive Chat Interface
with col_chat:
    st.subheader("💬 Customer Conversation")

    # If messages are empty, show greeting
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar=brand_meta["avatar"]):
            st.markdown(brand_meta["greeting_en"])
            st.caption(f"🇸🇦 {brand_meta['greeting_ar']}")

    # Render previous messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar=brand_meta["avatar"]):
                st.markdown(msg["content"])
                if msg.get("escalated"):
                    st.warning(f"⚠️ **Escalated to Human Agent** — Reason: `{msg.get('escalation_reason')}`")

    # Check for preset prompt from sidebar
    active_prompt = None
    if "preset_prompt" in st.session_state and st.session_state["preset_prompt"]:
        active_prompt = st.session_state["preset_prompt"]
        st.session_state["preset_prompt"] = None

    # Chat Input
    user_input = st.chat_input("Type your message here (English or Egyptian Arabic)...")
    prompt_to_process = active_prompt or user_input

    if prompt_to_process:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Run Agent
        with st.spinner("Processing through BrandMind constrained pipeline..."):
            response = run_agent_turn(selected_brand_key, prompt_to_process)

        # Store last response for telemetry inspection
        st.session_state.last_response = response

        # Display assistant reply
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.reply,
                "escalated": response.escalated,
                "escalation_reason": response.escalation_reason,
            }
        )

        with st.chat_message("assistant", avatar=brand_meta["avatar"]):
            st.markdown(response.reply)
            if response.escalated:
                st.warning(f"⚠️ **Escalated to Human Agent** — Reason: `{response.escalation_reason}`")

        st.rerun()


# RIGHT COLUMN: Real-Time Telemetry & Grounding Inspector
with col_telemetry:
    st.subheader("🔍 Real-Time Agent Telemetry")

    resp = st.session_state.last_response

    if resp is None:
        st.info("Send a message or click a demo scenario to view real-time NLU, RAG grounding, tool execution, and safety checklist telemetry.")
    else:
        # Metrics Top Bar
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Intent", resp.intent)
        with m2:
            st.metric("Language", resp.language.upper())
        with m3:
            st.metric("Escalated?", "YES" if resp.escalated else "NO")
        with m4:
            st.metric("Latency", f"{resp.latency_ms:.1f} ms")

        # Telemetry Tabs
        tab_decision, tab_rag, tab_nlu, tab_escalation = st.tabs(
            ["⚖️ Decision & Tools", "📚 Grounded RAG", "🧠 NLU & Entities", "🛡️ Safety Checklist"]
        )

        # TAB 1: Decision & Tools
        with tab_decision:
            st.markdown("#### Deterministic Decision Engine & Tool Calls")
            st.caption("Decisions are computed by deterministic Python code, never invented by LLM.")

            if resp.tool_calls:
                for idx, tool in enumerate(resp.tool_calls, 1):
                    with st.expander(f"Tool #{idx}: `{tool.tool_name}`", expanded=True):
                        st.markdown("**Arguments:**")
                        st.json(tool.arguments)
                        st.markdown("**Deterministic Output:**")
                        st.json(tool.output)
            else:
                st.write("No external tool or return decision required for this conversational turn.")

        # TAB 2: Grounded RAG
        with tab_rag:
            st.markdown("#### ChromaDB Knowledge Retrieval")
            st.caption("Brand policy and knowledge chunks retrieved via semantic similarity.")

            if resp.sources:
                for idx, src in enumerate(resp.sources, 1):
                    sim = src.get("similarity", 0.0)
                    sim_pct = f"{sim * 100:.1f}%"
                    badge_color = "green" if sim >= 0.50 else "orange"

                    with st.expander(f"Source #{idx}: `{src.get('chunk_id')}` (Similarity: :{badge_color}[{sim_pct}])", expanded=(idx == 1)):
                        st.markdown(f"**Document Title:** {src.get('title', 'N/A')}")
                        st.markdown(f"**Cosine Similarity:** `{sim:.4f}`")
                        st.markdown("**Retrieved Snippet:**")
                        st.code(src.get("snippet", ""), language="markdown")
            else:
                st.warning("No knowledge chunks retrieved or threshold fallback triggered (Policy Silence).")

        # TAB 3: NLU & Entities
        with tab_nlu:
            st.markdown("#### Natural Language Understanding")
            st.write(f"**Classified Intent:** `{resp.intent}`")
            st.write(f"**Detected Dialect / Language:** `{resp.language}`")
            st.markdown("**Extracted & Resolved Entities:**")
            st.json(resp.entities)

        # TAB 4: Safety Checklist & Escalation
        with tab_escalation:
            st.markdown("#### 7-Trigger Escalation Pre-Dispatch Gate")
            if resp.escalated:
                st.error(f"🚨 **Safety Gate Triggered:** `{resp.escalation_reason}`")
                st.markdown(
                    """
                    **Action Taken:**
                    - Automated conversation safely halted.
                    - Ticket routed to human customer support queue.
                    - Honest disclaimer displayed to customer.
                    """
                )
            else:
                st.success("✅ All 7 safety criteria passed. Automated reply safely delivered.")

            st.markdown(
                """
                **Safety Gate Checklist:**
                1. `EXPLICIT_HUMAN_REQUEST`: User requested a human
                2. `SAFETY_ALLERGY_HAZARD`: Medical, allergy, or physical reaction
                3. `COMPLAINT_OR_DISPUTE`: Furious customer, legal, or chargeback threat
                4. `PAYMENT_DISPUTE`: Double charge or payment processing issue
                5. `POLICY_SILENCE_UNANSWERABLE`: Retrieval score below threshold
                6. `LOW_CONFIDENCE`: Intent confidence below safety bound
                7. `MISSING_RETURN_FACTS`: Unresolved critical return facts
                """
            )
