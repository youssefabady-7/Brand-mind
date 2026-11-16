# SYNTHETIC DATA — not real customers/orders

> **IMPORTANT NOTICE:**
> All data stored in this repository—including customer profiles, order numbers, shipping addresses, phone numbers, brand catalogs, policy documents, and chat utterances—is **100% SYNTHETIC and hand-crafted** for academic evaluation and demonstration within the DEPI Capstone program.
> No real customer data, personal information, or proprietary enterprise records were used or recorded.

## Directory Structure

- `raw/`: Markdown and text source policy and FAQ documents per brand (`lumora/`, `urbankicks/`).
- `processed/`: Validated, chunked text files with embedded metadata tags (`brand_id`, `doc_type`, `section`, `last_updated`).
- `structured/`: Tabular CSV datasets:
  - `products.csv`: Synthetic product catalogs (names, SKUs, pricing in EGP, variants, ingredients/materials, stock).
  - `orders.csv`: Mocked order tracking data (order numbers, customer names, status, purchase dates, items).
  - `policies.csv`: Structured rule conditions (`policy_type`, `allowed_days`, `conditions`, `exceptions`).
  - `intent_dataset.csv`: Labeled intent classification dataset (450–900 rows across 9 intents).
  - `evaluation_dataset.csv`: Gold evaluation benchmark (Easy/Medium/Hard test cases with expected outputs).
