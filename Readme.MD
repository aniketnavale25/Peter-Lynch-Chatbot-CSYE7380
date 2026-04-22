---
title: Peter Lynch ChatBot
emoji: 📈
colorFrom: yellow
colorTo: green
sdk: docker
sdk_version: "1.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Peter Lynch ChatBot

A RAG-powered chatbot that answers investment questions in Peter Lynch's voice,
paired with a stock analyzer built on his PEG-ratio philosophy.

## Project Structure

```
lynch_chatbot/
├── app.py            ← Streamlit UI  (Part I chat + Part II stock analyzer)
├── lynch_rag.py      ← RAG backend   (ChromaDB + Groq + reranker)
├── requirements.txt
├── .env              ← Add your GROQ_API_KEY here
└── data/
    └── lynch/        ← Drop your Q&A CSV, PDFs, or TXT files here
        ├── lynch_qa.csv      (must have: questions, answers, labels columns)
        ├── one_up_on_wall_street.pdf
        └── ...
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key
Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get a free key at https://console.groq.com

### 3. Add Lynch's data
Put your hand-crafted Q&A CSV or any PDFs/TXTs into `data/lynch/`.

**CSV format** (required columns):
```
labels,questions,answers
Philosophy,"What is the PEG ratio?","The PEG ratio is P/E divided by..."
Strategy,"How do you find a ten-bagger?","Look for companies in boring industries..."
```

### 4. Run the app
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Architecture

```
User question
     │
     ▼
Query Expansion  (Groq Llama-3 rewrites short queries)
     │
     ▼
Vector Search    (ChromaDB + all-mpnet-base-v2 embeddings, top-20)
     │
     ▼
Reranking        (cross-encoder/ms-marco-MiniLM-L-6-v2, top-4)
     │
     ▼
Answer Generation (Groq Llama-3.1-8b-instant, Lynch persona prompt)
     │
     ▼
Streamlit UI
```

## Features

### Part I — ChatBot
- Peter Lynch persona with first-person RAG responses
- Conversation history (last 3 turns included for context)
- Query expansion for short/vague questions
- Cross-encoder reranking for precision retrieval
- Suggested starter questions

### Part II — Stock Analyzer
- **Fundamentals**: P/E, EPS growth, PEG, ROE, Debt/Equity, Market Cap
- **PEG Verdict**: Lynch-style assessment (bargain / fair / expensive)
- **5-year price chart** with 50-day and 200-day moving averages
- **Buy/Sell signals** based on price-vs-MA (Lynch value zone proxy)
- **Backtest**: Strategy cumulative return vs Buy & Hold
- Supports multiple tickers at once (comma-separated)
- PEG Ratio educational guide built in