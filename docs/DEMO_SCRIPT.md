# BrandMind: Live Demonstration Rehearsal Guide & Script

**Project:** BrandMind — AI-Powered Brand Customer Service Agent  
**Context:** DEPI Capstone Project (Team 4, AI & Data Science / AWS Machine Learning Engineer Track)  
**Deliverable:** Turn-by-Turn Live Demo Script with Telemetry Callouts & Fallback Contingencies  

---

## 1. Demo Setup & Environment Checklist

### 1.1 Local Launch Commands
Open two separate terminal windows inside the `Brand-mind` repository directory:

#### Terminal 1: FastAPI Backend (Optional / Standalone)
```powershell
cd Brand-mind
.\.venv\Scripts\Activate.ps1
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```
*Verify:* Open `http://127.0.0.1:8000/health` (should return `{"status": "healthy"}`).

#### Terminal 2: Streamlit Interactive User Interface
```powershell
cd Brand-mind
.\.venv\Scripts\Activate.ps1
streamlit run app/app.py
```
*Verify:* Browser opens automatically at `http://localhost:8501`.

### 1.2 Layout Overview
* **Left Column (Chat Window):** Clean conversational message bubbles with brand avatars and timestamps.
* **Right Column (Real-Time Telemetry Inspector):** Four live tabs inspecting internal agent state on every turn:
  1. `⚖️ Decision & Tools`: Exact tool arguments, deterministic outputs, and rule engine results.
  2. `📚 Grounded RAG`: Retrieved ChromaDB chunks, cosine similarity scores, and document titles.
  3. `🧠 NLU & Entities`: Detected intent, language/dialect badge, and extracted entity JSON.
  4. `🛡️ Safety Checklist`: Status of the 7-trigger pre-dispatch gate.

---

## 2. Turn-by-Turn Live Demonstration Script

### Scene 1: Grounded FAQ & Brand Voice (Lumora Skincare)
* **Active Brand:** `Lumora Skincare` (Selected in sidebar)
* **Goal:** Demonstrate ChromaDB RAG knowledge grounding and calm, reassuring brand voice.
* **Action:** Click sidebar quick-trigger `"FAQ (Shipping Cairo)"` or type:
  > *"What are your shipping rates and delivery times for Cairo?"*

#### Expected Agent Response:
> *"Welcome to Lumora. Shipping and delivery to Cairo is 45 EGP within 1 to 2 business days. We offer free shipping on orders over 800 EGP."*

#### Telemetry Callout for the Committee:
1. Point to **Top Bar:** Intent is `shipping`, Language is `EN`, Latency is `< 50 ms`.
2. Open **`📚 Grounded RAG` Tab:** Highlight chunk `lumora_02_shipping_and_delivery_policy_004` retrieved from ChromaDB with a high cosine similarity score (~0.47).
3. Open **`🛡️ Safety Checklist` Tab:** Point to green banner: *"All 7 safety criteria passed. Automated reply safely delivered."*

---

### Scene 2: Mocked Order Tracking via Tool Call
* **Active Brand:** `Lumora Skincare`
* **Goal:** Demonstrate autonomous tool selection and execution against synthetic database records.
* **Action:** Click sidebar quick-trigger `"Tool (Track Order LUM-1001)"` or type:
  > *"Can you check the status of my order LUM-1001?"*

#### Expected Agent Response:
> *"Your order has been delivered to Nour El-Din under tracking BST-EG-91001."*

#### Telemetry Callout for the Committee:
1. Point to **Top Bar:** Intent is `order_status`.
2. Open **`⚖️ Decision & Tools` Tab:** Show that the agent autonomously executed `get_order_status` with argument `{"order_id": "LUM-1001", "brand_id": "lumora"}`.
3. Show the deterministic JSON payload containing courier tracking `BST-EG-91001`, customer name `Nour El-Din Hassan`, and delivery status `delivered`.

---

### Scene 3: Deterministic Policy Engine — Eligible Return (10 Days, Sealed)
* **Active Brand:** `Lumora Skincare`
* **Goal:** Prove that the LLM NEVER invents return policy. Code decides; LLM explains.
* **Action:** Click sidebar quick-trigger `"Policy: Eligible Return (10d)"` or type:
  > *"I bought a Vitamin C Serum 10 days ago. It is completely unopened with seal intact. Can I return it?"*

#### Expected Agent Response:
> *"We are pleased to inform you that your request is eligible within our 14-day window with unopened seal. Instructions: our courier will collect the item."*

#### Telemetry Callout for the Committee:
1. Point to **`⚖️ Decision & Tools` Tab:** Show that the deterministic function `check_return_eligibility()` evaluated facts:
   - `days_since_delivery = 10` $\le 14$ (Within window)
   - `is_opened = false` (Hygiene compliant)
   - Matched Rule: `ELIGIBLE_WITHIN_WINDOW`
2. Explain to the jury: *"The LLM did not grant this return. Python code evaluated the condition matrix, and the decision was logged to `data/logs/audit_decisions.jsonl`."*

---

### Scene 4: Deterministic Policy Engine — Rejection (18 Days, Opened)
* **Active Brand:** `Lumora Skincare`
* **Goal:** Demonstrate strict enforcement of boundary conditions ($N > 14$) and cosmetics hygiene seals.
* **Action:** Click sidebar quick-trigger `"Policy: Ineligible (18d)"` or type:
  > *"I bought a cleanser 18 days ago and opened it. Can I return it?"*

#### Expected Agent Response:
> *"We apologize, but your request is ineligible because it exceeds our 14 days policy window."*

#### Telemetry Callout for the Committee:
1. Point to **`⚖️ Decision & Tools` Tab:** Rule engine evaluated $18 > 14$, returning `EXCEEDED_RETURN_WINDOW` and `OPENED_HYGIENE_RESTRICTION`.
2. Emphasize: *"Notice that even if the customer asks politely, the deterministic code refuses the return. The LLM cannot be prompted to bypass this rule."*

---

### Scene 5: Safety Gate Interception (Severe Allergic Reaction)
* **Active Brand:** `Lumora Skincare`
* **Goal:** Prove that customer health and safety override all marketing or automated loops.
* **Action:** Click sidebar quick-trigger `"Safety Escalation (Allergy)"` or type:
  > *"I used the serum yesterday and now my skin is burning and I have a severe allergic reaction!"*

#### Expected Agent Response:
> ⚠️ **Escalated to Human Agent**  
> *"سلامتك وصحتك أولويتنا القصوى في لومورا. نعتذر منك بشدة عن هذه التجربة، ونرجو منك التوقف فوراً عن استخدام المنتج وغسل البشرة بلطف بالماء البارد. تم تحويل محادثتك فوراً إلى أخصائي الرعاية وفريق خدمة العملاء للمتابعة العاجلة معك."*

#### Telemetry Callout for the Committee:
1. Point to the **Bright Orange Warning Alert** in chat: *"Escalated to Human Agent — Reason: `SAFETY_ALLERGY_HAZARD`"*.
2. Open **`🛡️ Safety Checklist` Tab:** Show that Trigger #2 (`SAFETY_ALLERGY_HAZARD`) fired instantly.
3. Show that no sales pitch or discount code was generated; an official human support ticket was registered via `escalate_to_human`.

---

### Scene 6: Multi-Tenancy & Tone Differentiation (UrbanKicks Egypt)
* **Active Brand Switch:** In the sidebar, change brand to **`UrbanKicks (Streetwear)`**.
* **Visual Shift:** Notice header shifts from calm sage green 🌿 to vibrant streetwear dark amber 👟.
* **Goal:** Prove multi-tenancy and tone differentiation on an identical shipping inquiry.
* **Action:** Type:
  > *"What are your shipping rates for Cairo?"*

#### Expected Agent Response:
> *"Yo! Welcome to UrbanKicks. Shipping and delivery to Cairo is 50 EGP within 1 to 2 business days, with free shipping over 1500 EGP. Step fresh!"*

#### Telemetry Callout for the Committee:
1. Point out the dramatic tone shift: *"Notice the street-smart, energetic tone ('Yo!', 'Step fresh!'), while Lumora was calm and clinical."*
2. Point out the policy shift: Free shipping threshold for UrbanKicks is 1500 EGP (vs 800 EGP for Lumora), and shipping is 50 EGP (vs 45 EGP). Both retrieved from distinct partitioned YAML/RAG configurations.

---

### Scene 7: Egyptian Colloquial Arabic (Order Tracking in Dialect)
* **Active Brand:** `UrbanKicks Egypt`
* **Goal:** Demonstrate understanding of authentic Egyptian Arabic phrasing.
* **Action:** Click sidebar quick-trigger `"Egyptian Arabic (Order Status)"` or type:
  > *"لو سمحت عاوز اعرف طلبي UK-2001 وصل لفين؟"*

#### Expected Agent Response:
> *"طلبك تم تسليمه بنجاح (delivered) للمستلم Nour El-Din عبر رقم التتبع BST-EG-91001. نتمنى لك تجربة رائعة!"*

#### Telemetry Callout for the Committee:
1. Point to **Language Metric:** Detected dialect is `AR_EG`.
2. Point to **Intent Metric:** `order_status` correctly parsed despite colloquial phrasing (`عاوز اعرف`, `وصل لفين`).
3. Point to **Extracted Entities:** `order_id` correctly extracted as `UK-2001`.

---

### Scene 8: Explicit Human Escalation & Legal Threat
* **Active Brand:** `UrbanKicks Egypt`
* **Goal:** Demonstrate immediate compliance with explicit customer escalation demands.
* **Action:** Click sidebar quick-trigger `"Human Escalation (Angry)"` or type:
  > *"This is the third time you delay! I am furious and want to speak with a human supervisor right now!"*

#### Expected Agent Response:
> ⚠️ **Escalated to Human Agent**  
> *"نعتذر منك بشدة يا كوتش على التجربة دي وحقك علينا تماماً. جاري تحويلك حالاً لأحد مشرفي خدمة العملاء في إيربان كيكس عشان يحل مشكلتك في أسرع وقت."*

#### Telemetry Callout for the Committee:
1. Point to **`🛡️ Safety Checklist` Tab:** Trigger #1 (`EXPLICIT_HUMAN_REQUEST`) and Trigger #3 (`COMPLAINT_OR_DISPUTE`) fired.
2. Emphasize: *"BrandMind is honest. It never argues with an angry customer or pretends to be a human. It respects the customer's wish and yields immediately."*

---

## 3. Contingency & Fallback Procedures

| Scenario | Potential Issue | Contingency Action |
| :--- | :--- | :--- |
| **Port 8000 Busy / Blocked** | FastAPI cannot bind to port 8000. | In the Streamlit sidebar, select **"Direct Embedded Agent"**. The app executes the in-memory agent directly without any network dependencies. |
| **ChromaDB Cache Delay** | First turn takes 2 seconds to load embeddings. | The model is pre-loaded during script initialization; subsequent turns execute in < 40 milliseconds. |
| **Typo in Order Number** | Customer enters `LUM-9999` (non-existent). | Agent handles gracefully: Tool returns `found: false`, agent politely requests the customer to verify their receipt. |
| **Complex Jury Question** | Jury asks: *"How does this differ from LangChain / AutoGPT?"* | Answer: *"LangChain allows unconstrained LLM loops where the model decides actions in an open loop. BrandMind is a **constrained agent loop** with a single decision point and a deterministic safety gate. The LLM can never invoke an unauthorized tool or alter policy rules."* |
