---
title: Peter Lynch ChatBot
emoji: 📈
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: "1.56.0"
app_file: app.py
pinned: false
---
new version:
# Peter Lynch ChatBot

**CSYE 7380 — Generative AI in Finance | Northeastern University**
**Team: Peter Lynch Group**

A Streamlit-based investment analysis application that combines a RAG-powered chatbot with quantitative stock screening tools, all built around Peter Lynch's legendary investment philosophy.

---

## Overview

This project delivers three integrated components as required by the course assignment:

1. **RAG Chatbot** — Answers questions in Peter Lynch's voice using a knowledge base built from Lynch's books and curated Q&A pairs
2. **Lynch Stock Analyzer** — Deep-dive fundamental analysis with backtest for any ticker
3. **Financial Ratios Dashboard** — Portfolio-level `fin_data_df` view of key ratios for the Dow Jones 30
4. **K-Means Stock Screener** — Replicates the professor's exact clustering algorithm to generate Long/Short recommendations

---

## Application Structure

| Tab | Description |
|-----|-------------|
| Chat with Peter Lynch | RAG chatbot powered by LangChain + ChromaDB + Groq (Llama 3) |
| Lynch Stock Analyzer | Single-stock deep dive: metrics, scorecard, price chart, backtest |
| Financial Dashboard | `fin_data_df` — key financial ratios for all selected stocks |
| K-Means Stock Screener | Professor's exact K-Means algorithm: Long = cluster 3, Short = cluster 2 |

---

## Tech Stack

### Chatbot (Part I)
- **Framework**: LangChain + ChromaDB (vector store)
- **Embeddings**: `paraphrase-MiniLM-L6-v2` (sentence-transformers)
- **Reranker**: CrossEncoder for result re-ranking
- **LLM**: Groq API with Llama 3 (via OpenAI-compatible interface)
- **Knowledge base**: Peter Lynch Q&A pairs (6 labels, 800+ pairs) + PDF documents

### Stock Analysis (Part II)
- **Data**: `yahooquery` for live financial data
- **Visualization**: Plotly (interactive charts)
- **Backtest**: Lynch-inspired MA200 crossover strategy vs Buy & Hold

### K-Means Screener (Part II — follows professor's notebook)
- **Data**: `yahooquery` `Ticker.financial_data` — same library as professor
- **Algorithm**: `KMeans(n_clusters=4, random_state=100)` with `MinMaxScaler`
- **Signal**: cluster 3 = Long, cluster 2 = Short (professor's exact assignment)
- **Default universe**: Dow Jones 30 stocks

---

## Project Structure

```
Peter-Lynch-Chatbot-CSYE7380/
├── app.py                    # Main Streamlit application
├── lynch_rag.py              # RAG backend (LangChain + ChromaDB + Groq)
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
├── data/
│   └── lynch/
│       ├── peter_lynch_qa_merged.csv     # 800+ Q&A pairs (6 labels)
│       ├── Financial_Ratios_QA.csv       # Financial ratio explanations
│       └── *.pdf                         # Reference documents
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.12
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Install dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# Install packages
pip install -r requirements.txt
pip install yahooquery scikit-learn>=1.3.0
```

### Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_groq_api_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
```

### Run the app

```bash
python -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Knowledge Base

The chatbot's knowledge base is built from CSV files in `data/lynch/`. Each CSV must contain `question`, `answer`, and `label` columns.

| Label | Description | Count |
|-------|-------------|-------|
| Personal Life | Lynch's background and biography | ~329 |
| Psychology | Investor mindset and behavioral principles | ~100 |
| Risk Management | Position sizing and risk control | ~100 |
| Adaptability | Adjusting strategy to market conditions | ~99 |
| Timing | When to buy and sell | ~135 |
| Strategy | Core investment philosophy | ~51 |
| Financial Ratios | PEG ratio, ROE, D/E interpretations | 25 |

To add new documents, place CSV or PDF files in `data/lynch/` and restart the app, then click **Load Lynch Knowledge Base** in the sidebar.

---

## K-Means Algorithm (Professor's Notebook)

The K-Means screener follows the professor's `PeterLynch_Assignment.ipynb` exactly:

```python
# Step 1: Fetch data using yahooquery (same library as professor)
ticker = Ticker(stock_symbol_list)
fin_data_df = pd.DataFrame.from_dict(ticker.financial_data, orient='index').T

# Step 2: Preprocess (professor's exact code)
data = fin_data_df.T.fillna(0)
X = data.drop([non_numeric_cols], axis=1).to_numpy()

# Step 3: Scale and cluster
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
model = KMeans(n_clusters=4, random_state=100)
model.fit(X)
yhat = model.predict(X)

# Step 4: Signal assignment (professor's exact cluster numbers)
Long  = data.index[yhat == 3]
Short = data.index[yhat == 2]
```

---

## Lynch Stock Analyzer — Scorecard Criteria

The analyzer scores each stock against 6 of Lynch's key criteria:

| Criterion | Threshold | Lynch's Rationale |
|-----------|-----------|-------------------|
| PEG Ratio | < 1.0 = bargain | Core Lynch valuation metric |
| EPS Growth | > 15% = fast grower | Primary growth signal |
| Debt/Equity | < 80% = healthy | Debt kills great companies |
| ROE | > 15% = strong | Management efficiency |
| Current Ratio | > 2.0 = safe | Liquidity buffer |
| Insider Ownership | > 5% = skin in game | Management conviction |

---

## Team

| Member | Role |
|--------|------|
| Yingying Zhuang | LSTM modeling, Label 6 (Psychology), UI integration |
| Aniket Vaman Navale | RAG pipeline, deployment |
| Yanqing Lou | Label 1 (Personal Life), data engineering |
| Yizhi Liu | Label 3 (Career Experience) |
| Yu Cao | Label 2 (Investment Insights) |
| Zitiantao Lin | Label 4 (Stock Analysis) |

**Assigned stock**: META

---

## Notes

- The app uses `yahooquery` as the primary data source (same library as professor's notebook). A static dataset of Dow Jones 30 financials is included as fallback when live data is unavailable.
- The backtest strategy is a simplified MA200 crossover approximation of Lynch's value zone approach, not financial advice.
- All financial data is for educational purposes only.