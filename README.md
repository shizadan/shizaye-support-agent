# Shiza — Shizaye Multilinks Support Agent

A single-agent AI customer support application built for Shizaye Multilinks,
a multi-branch telecoms retail business in Kaduna, Nigeria. Shiza answers
customer questions about products, pricing, SIM registration, and warranties,
and can look up live order status — built as a capstone project for the AI &
Data Science bootcamp (ihifix scholarship program).

## 🔗 Live Application
[https://shizaye-support-agent-5eadgqtxtgqxbfr7g3b2fz.streamlit.app/](https://shizaye-support-agent-5eadgqtxtgqxbfr7g3b2fz.streamlit.app/)

## 🎥 Project Video
[Add your Google Drive video URL here]

## Problem It Solves
Shizaye Multilinks handles repetitive customer questions across 8 branches —
opening hours, SIM registration steps, data plans, warranty policy, and order
status. Shiza automates first-line support, freeing staff to focus on
higher-value tasks while giving customers instant, consistent answers.

## Architecture
- **LLM**: `openai/gpt-oss-120b` via Groq's free API (OpenAI-compatible tool-calling) — no cost, no credit card required
- **Agent pattern**: Single agent with native tool-use (function calling)
- **Tools**:
  1. `search_faq` — Retrieval-Augmented Generation over a FAQ/product knowledge
     base, embedded locally with `sentence-transformers` (no embedding API cost)
  2. `lookup_order` — Simulated backend order/account lookup (CSV-based mock DB,
     representing what would connect to Shizaye's real order system)
- **Memory**: Session-based conversation memory via Streamlit `session_state`
- **Deployment**: Streamlit Community Cloud

## How It Works
1. Customer sends a message in the chat UI.
2. The model decides whether the question needs a tool call (FAQ search or order
   lookup) or can be answered directly.
3. If a tool is called, the result is returned to the model, which grounds its
   final answer in that data (no hallucinated prices or statuses).
4. Conversation history persists for the session, so Shiza remembers context
   within a chat.

## Project Structure
```
shizaye-support-agent/
├── app.py                 # Streamlit UI + main loop
├── agent.py                # Agent logic, tool routing, system prompt
├── tools/
│   ├── faq_rag.py          # RAG: FAQ embedding + retrieval
│   └── order_lookup.py     # Mock order/account lookup
├── data/
│   ├── faq_docs.csv        # FAQ/product knowledge base
│   └── orders.csv          # Mock order data
├── requirements.txt
└── README.md
```

## Running Locally / in Colab
```bash
pip install -r requirements.txt
streamlit run app.py
```
You'll be prompted for your Groq API key (or set it as a Streamlit secret
`GROQ_API_KEY` when deploying). Get a free key at console.groq.com — no
credit card required.

## Tech Stack
- Python
- Groq API (free tier) running `openai/gpt-oss-120b`, accessed via the `openai` SDK (OpenAI-compatible endpoint)
- sentence-transformers (local embeddings, free)
- Streamlit
- pandas / numpy

## Known Limitations
- Order data is a mock CSV, not a live database connection — in production this
  would connect to Shizaye's real backend (e.g. Firebase/SimShop Network).
- FAQ knowledge base is a starter set and should be expanded with real,
  up-to-date branch data.
- Currently runs on Groq's free tier for zero-cost development. The agent
  architecture is provider-agnostic — swapping to Claude (Anthropic API) for
  production use would only require changing the client initialization and
  tool-call response parsing in `agent.py`.

## Author
Dan Shizamuayi Shina — Senior Manager, Shizaye Multilinks Limited
[GitHub: shizadan](https://github.com/shizadan)
