# BrandMind: AI-Powered Brand Customer Service Agent

> **“Your brand, always on.”**
>
> DEPI Capstone Project — AI & Data Science / AWS Machine Learning Engineer Track  
> Team Number 4 (Six Members)

---

## 1. Project Overview

**BrandMind** is an enterprise-grade agentic AI customer-service system tailored for online and local e-commerce brands in Egypt and the wider MENA region. Supporting English, Modern Standard Arabic, and Egyptian Arabic, BrandMind automates high-frequency, repetitive customer inquiries (order status, product queries, sizing, return and exchange policies) while cleanly handing off complex, sensitive, or high-value conversations to human agents.

BrandMind is **not** a rigid rule-based decision tree and **not** an unconstrained plain LLM chatbot. It employs a **hybrid architecture** combining:
1. **RAG (Retrieval-Augmented Generation):** Grounded factual recall over chunked brand documents using vector similarity search (ChromaDB + sentence embeddings).
2. **Constrained Agent Loop & Tool Use:** Deterministic execution for querying catalog items, retrieving mocked order statuses, and validating return eligibility.
3. **Deterministic Business Policy Engine:** 100% rule-based, reproducible, unit-tested Python code enforcing return/exchange rules without LLM intervention.
4. **Traditional Machine Learning:** Scikit-learn multi-class intent classifier benchmarked against few-shot LLM intent classification.

---

## 2. The Core Non-Negotiable Principle

> **THE LLM NEVER INVENTS COMPANY POLICY.**

In BrandMind, labor is strictly partitioned:
* **The Two Fuzzy Ends (Handled by LLM):** 
  - *Input:* Intent classification and structured entity/fact extraction from free text (including Egyptian dialect).
  - *Output:* Phrasing the validated response in the distinct, configured brand voice (calm vs. youthful).
* **Factual Grounding (Handled by Retrieval & Tools):** Knowledge chunks retrieved via vector similarity; mocked live data accessed via deterministic tool functions.
* **Operational Decisions (Handled by Pure Python):** Return eligibility, refund validity, and escalation checks are evaluated by deterministic, auditable code.
* **Human Escalation:** Escalation is treated as a first-class success path rather than a system failure.

---

## 3. Synthetic Multi-Tenant Brands

BrandMind is architected as a multi-tenant platform serving multiple brands through external YAML configurations and vector metadata filtering:

1. **Lumora (Main MVP Brand):**
   * *Domain:* Minimalist skincare and beauty rituals.
   * *Tone:* Calm, minimalist, empathetic, reassuring, scientifically mindful.
   * *Policy Nuance:* Strict hygiene standards; opened/unsealed skincare items are non-returnable; allergic reactions trigger immediate human escalation.
2. **UrbanKicks (Demo & Tone-Differentiation Brand):**
   * *Domain:* Youthful streetwear, hype sneakers, and lifestyle apparel.
   * *Tone:* Youthful, energetic, upbeat, confident, street-smart.
   * *Policy Nuance:* Footwear must be unworn outdoors with clean soles and tags intact.

> **DATA NOTICE:** All data across this repository is **100% SYNTHETIC** hand-crafted data created specifically for evaluation purposes. No real customer or order information is used.

---

## 4. Repository Structure

```text
Brand-mind/
├── app/                  # Streamlit Chat UI with real-time intent & escalation side panel
├── config/               # Brand YAML personality and policy profiles (lumora.yaml, urbankicks.yaml)
├── data/
│   ├── raw/              # Markdown/text source policy and FAQ documents per brand
│   ├── processed/        # Cleaned and chunked knowledge chunks with metadata
│   └── structured/       # Synthetic CSVs (products.csv, orders.csv, policies.csv, benchmarks)
├── notebooks/            # Jupyter notebooks for intent classifier training & evaluation
├── src/
│   ├── agents/           # Constrained agent loop, conversation memory, escalation checklist
│   ├── embeddings/       # Embedding generation and ChromaDB indexing
│   ├── retrieval/        # Top-k vector retrieval with brand_id filtering and threshold fallback
│   ├── policies/         # Deterministic return/exchange rule engine
│   ├── tools/            # Validated tool functions (catalog, orders, eligibility, escalation)
│   ├── prompts/          # Jinja2 layered prompts and brand voice formatting
│   └── evaluation/       # Benchmark runner, precision@k, hallucination and latency metrics
├── tests/                # Comprehensive pytest suite (policy engine, tools, endpoints)
├── .env.example          # Environment template for LLM provider and DB keys
├── requirements.txt      # Pinned Python package dependencies
└── README.md             # Project master documentation
```

---

## 5. Quickstart Guide

### Prerequisites
* Python 3.10+
* Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/youssefabady-7/Brand-mind.git
   cd Brand-mind
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your LLM API keys and preferred configuration
   ```

5. **Run Unit Tests:**
   ```bash
   pytest tests/ -v
   ```

6. **Launch the Application:**
   * Backend API:
     ```bash
     uvicorn src.main:app --reload --port 8000
     ```
   * Frontend Streamlit UI:
     ```bash
     streamlit run app/app.py
     ```

---

## 6. Team & Roles (DEPI Capstone Team 4)

* **Youssef Abady (Team Leader) — AI Agent Architect & System Integrator:** Constrained agent loop, session memory, escalation checklist, repository architecture, and live demo integration.
* **Fayrouz Ahmed (Co-Leader) — RAG & Generative AI Lead:** Vector retrieval, ChromaDB metadata filtering, similarity threshold calibration, grounded generation, and AWS architecture design.
* **Dalia Mohamed — Data Engineering & Knowledge Base Lead:** Synthetic brand knowledge authoring, ingestion pipeline, structured CSVs, and labeled Easy/Medium/Hard evaluation dataset.
* **Youssef Wael — Backend, Policy Engine & Tools Developer:** Deterministic policy decision engine, tool functions, FastAPI endpoints, mocked stores, and unit tests.
* **Mariam Osama — NLP, ML & Evaluation Lead:** Intent classification (9 classes), JSON entity extraction, Logistic Regression benchmark, and end-to-end evaluation runner.
* **Mohamed Mostafa — Prompt Engineering, Brand Personality & UI Lead:** Jinja2 prompt layers, Egyptian Arabic dialect tuning, brand YAML configs, and Streamlit UI development.
