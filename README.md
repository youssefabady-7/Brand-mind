# BrandMind: AI-Powered Brand Customer Service Agent

> **“Your brand, always on.”**
>
> **DEPI Capstone Project — AI & Data Science / AWS Machine Learning Engineer Track**  
> **Team Number 4 (Six Members)**  
> **Status:** 100% Complete & Validated | 58/58 Unit & Integration Tests Passing (100%)

---

## 1. Project Overview

**BrandMind** is an enterprise-grade agentic AI customer-service system tailored for online and direct-to-consumer (D2C) brands in Egypt and the wider MENA region. Supporting English, Modern Standard Arabic, and Egyptian Colloquial Arabic (*العامية المصرية*), BrandMind automates high-frequency, repetitive customer inquiries (order status, product discovery, delivery timelines, and return/exchange eligibility) while cleanly and safely handing off complex, sensitive, or high-value conversations to human agents.

BrandMind is **not** a rigid rule-based decision tree and **not** an unconstrained plain LLM chatbot. It employs a **hybrid architecture** combining:
1. **RAG (Retrieval-Augmented Generation):** Factual grounding over chunked brand documents using vector similarity search (ChromaDB + Sentence-Transformers) with cross-lingual thresholding for Egyptian Arabic.
2. **Constrained Agent Loop & Tool Use:** Autonomous single decision point execution for catalog queries, mocked order status tracking, and return eligibility checks.
3. **Deterministic Business Policy Engine:** 100% rule-based, reproducible, unit-tested Python code enforcing return/exchange rules without LLM intervention.
4. **Traditional Machine Learning:** Scikit-learn multi-class intent classifier benchmarked against few-shot LLM intent classification across 540 stratified samples.
5. **Interactive UI & Real-Time Telemetry:** Streamlit dual-column application featuring live telemetry inspection of NLU, RAG grounding, tool execution, and safety gate triggers.

---

## 2. Key Evaluation Results & System Metrics

| Metric | Measured Benchmark | Spec Requirement | Status |
| :--- | :---: | :---: | :---: |
| **Intent Classification Accuracy** | **100.0% (20/20)** | $\ge 85.0\%$ | **EXCEEDS** |
| **Tool / Action Selection Rate** | **100.0% (20/20)** | $\ge 90.0\%$ | **EXCEEDS** |
| **Safety Escalation Precision** | **100.0% (7/7)** | $100.0\%$ | **MEETS** |
| **Policy Hallucination Rate** | **0.0%** | $0.0\%$ | **MEETS** |
| **Deterministic Rule Evaluation** | **100.0% (14/14)** | $100.0\%$ | **MEETS** |
| **Mean Turn Latency** | **1,459.5 ms** | $< 3,000 \text{ ms}$ | **EXCEEDS** |
| **Pytest Unit & Integration Tests** | **58 / 58 Passed** | $100.0\%$ | **MEETS** |

---

## 3. The Core Non-Negotiable Principle

> **THE LLM NEVER INVENTS COMPANY POLICY.**

In BrandMind, labor is strictly partitioned:
* **The Two Fuzzy Ends (Handled by LLM):** 
  - *Input:* Intent classification and structured entity extraction from free text (including Egyptian dialect).
  - *Output:* Phrasing the verified response in the brand's distinct tone of voice (calm vs. youthful).
* **Factual Grounding (Handled by RAG & Tools):** Knowledge chunks retrieved via vector similarity; mocked live data accessed via deterministic tool functions.
* **Operational Decisions (Handled by Pure Python):** Return eligibility, day windows, hygiene rules, and escalation checks are evaluated by deterministic, auditable code.
* **Human Escalation:** Escalation is treated as a first-class success path rather than a system failure.

---

## 4. Multi-Tenant Synthetic Brands

BrandMind is architected as a multi-tenant platform serving multiple brands through external YAML configurations and vector metadata filtering:

1. **Lumora Skincare (Main MVP Brand):**
   * *Domain:* Minimalist skincare and beauty rituals in Egypt.
   * *Tone:* Calm, minimalist, empathetic, reassuring, scientifically mindful.
   * *Policy Nuance:* Strict hygiene standards; opened/unsealed skincare items are non-returnable; allergic reactions trigger immediate human escalation.
2. **UrbanKicks Egypt (Tone-Differentiation & Multi-Tenancy Proof):**
   * *Domain:* Youthful streetwear, hype sneakers, and lifestyle apparel.
   * *Tone:* Youthful, energetic, upbeat, confident, street-smart Egyptian tone.
   * *Policy Nuance:* Footwear must be unworn outdoors with clean soles, original box, and tags intact.

> **DATA NOTICE:** All data across this repository is **100% SYNTHETIC** hand-crafted data created specifically for evaluation purposes. No real customer or order information is used (`data/README.md`).

---

## 5. Complete Deliverables & Documentation Index

All capstone deliverables required by DEPI guidelines are authored and persisted in this repository:

* 📄 **[Final Capstone Academic Report](docs/FINAL_REPORT.md):** Comprehensive report covering problem statement, system architecture, NLU, deterministic engine, ML benchmarks, and ethical AI.
* 📊 **[Evaluation & ML Benchmark Report](docs/EVALUATION_REPORT.md):** Quantitative evaluation report with confusion matrix, precision/recall per class, and KPI analysis.
* ☁️ **[Target AWS Cloud Architecture Specification](docs/AWS_ARCHITECTURE.md):** Serverless AWS production design (API Gateway, Lambda, Bedrock, OpenSearch, Aurora, DynamoDB) with Mermaid diagrams and monthly cost estimation.
* 🖥️ **[Defense Presentation Deck & Script](docs/PRESENTATION_SLIDES.md):** 12-slide presentation deck with slide-by-slide verbatim speaker notes for Team 4.
* 🎬 **[Live Demo Rehearsal Script](docs/DEMO_SCRIPT.md):** Turn-by-turn live demonstration guide with exact input prompts in English and Egyptian Arabic, telemetry inspection callouts, and fallback procedures.
* 📓 **[Intent Classifier Benchmark Notebook](notebooks/intent_classifier_benchmark.ipynb):** Google Colab-ready Jupyter notebook implementing dataset loading, TF-IDF + Logistic Regression training, and confusion matrix plotting.

---

## 6. Repository Architecture

```text
Brand-mind/
├── app/
│   └── app.py            # Streamlit multi-tenant UI with real-time telemetry inspector
├── config/
│   ├── lumora.yaml       # Lumora brand persona, tone, and policy rules
│   └── urbankicks.yaml   # UrbanKicks brand persona, tone, and policy rules
├── data/
│   ├── raw/              # Bilingual Markdown source documents (Lumora & UrbanKicks)
│   ├── processed/        # Cleaned chunks with token counts (chunks.jsonl)
│   ├── structured/       # Synthetic CSVs (products.csv, orders.csv, intent_dataset.csv, evaluation_dataset.csv)
│   ├── evaluation/       # Benchmark JSON outputs and classification reports
│   └── logs/             # JSONL audit logs for return decisions
├── docs/
│   ├── FINAL_REPORT.md       # Final academic capstone report
│   ├── EVALUATION_REPORT.md  # Detailed ML & E2E evaluation report
│   ├── AWS_ARCHITECTURE.md   # Production AWS cloud specification & Mermaid diagram
│   ├── PRESENTATION_SLIDES.md# 12-slide jury defense deck & script
│   ├── DEMO_SCRIPT.md        # Turn-by-turn live demonstration guide
│   └── assets/               # Confusion matrix graphic and architectural assets
├── models/
│   └── intent_classifier.joblib  # Trained scikit-learn intent classifier artifact
├── notebooks/
│   └── intent_classifier_benchmark.ipynb  # Colab-ready ML benchmark notebook
├── src/
│   ├── agents/           # Constrained loop, NLU, language detection, memory, escalation gate
│   ├── embeddings/       # Sentence-transformers embedding & ChromaDB vector indexing
│   ├── retrieval/        # Top-k vector retriever with thresholding & brand isolation
│   ├── policies/         # Deterministic return/exchange rule engine & audit logger
│   ├── tools/            # Validated tool functions (orders, catalog, eligibility, escalation)
│   ├── prompts/          # Jinja2 layered prompts for system, NLU, and grounded replies
│   ├── evaluation/       # ML training pipeline, dataset generator, and E2E benchmark runner
│   └── main.py           # Production FastAPI backend REST API
├── tests/                # 58 unit & integration tests covering 100% of pipeline
├── requirements.txt      # Pinned Python package dependencies
├── .env.example          # Environment variables template
└── README.md             # Master project documentation
```

---

## 7. Quickstart Guide

### Prerequisites
* Python 3.10+
* Git

### Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/youssefabady-7/Brand-mind.git
   cd Brand-mind
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   ```

---

## 8. Running Tests & Evaluations

### Run Full Test Suite (58 Tests)
```bash
pytest tests/ -v
```

### Train Traditional ML Intent Classifier & Generate Confusion Matrix
```bash
python src/evaluation/train_classifier.py
```
*Outputs: `models/intent_classifier.joblib`, `data/evaluation/intent_classification_report.json`, `docs/assets/confusion_matrix.png`.*

### Run End-to-End System Evaluation Benchmark (20 Scenarios)
```bash
python src/evaluation/runner.py
```
*Outputs: `data/evaluation/e2e_evaluation_results.json` and formatted console report.*

---

## 9. Launching the Live Applications

### Option A: Launch Interactive Streamlit UI
```bash
streamlit run app/app.py
```
*Opens `http://localhost:8501`. Features brand switcher (Lumora vs UrbanKicks), interactive chat, demo quick-triggers, and real-time telemetry inspector.*

### Option B: Launch FastAPI Backend Server
```bash
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger API documentation available at `http://127.0.0.1:8000/docs`.*

---

## 10. Team & Roles (DEPI Capstone Team 4)

* **Youssef Abady (Team Leader) — AI Agent Architect & System Integrator:** Constrained agent loop, session memory, escalation checklist, repository architecture, and live demo integration.
* **Fayrouz Ahmed (Co-Leader) — RAG & Generative AI Lead:** Vector retrieval, ChromaDB metadata filtering, similarity threshold calibration, grounded generation, and AWS architecture design.
* **Dalia Mohamed — Data Engineering & Knowledge Base Lead:** Synthetic brand knowledge authoring, ingestion pipeline, structured CSVs, and labeled Easy/Medium/Hard evaluation dataset.
* **Youssef Wael — Backend, Policy Engine & Tools Developer:** Deterministic policy decision engine, tool functions, FastAPI endpoints, mocked stores, and unit tests.
* **Mariam Osama — NLP, ML & Evaluation Lead:** Intent classification (9 classes), JSON entity extraction, Logistic Regression benchmark, and end-to-end evaluation runner.
* **Mohamed Mostafa — Prompt Engineering, Brand Personality & UI Lead:** Jinja2 prompt layers, Egyptian Arabic dialect tuning, brand YAML configs, and Streamlit UI development.
