# QuarterLens: Earnings Call Analyst 📈

*An LLM-powered research tool that uses RAG (Retrieval-Augmented Generation) to answer natural-language questions about a company's earnings call history, built entirely on free-tier infrastructure (Groq, Alpha Vantage, and open-source Hugging Face models).*

<p align="center"> <img width="800" src="images/quarterlens_UI.gif"> </p>

## Overview

Reading through two years of quarterly earnings calls to answer a single question, **"Did the company launch any new products?"** or **"How consistent has management been in delivering on past promises?"** is slow and repetitive. QuarterLens automates that process: type a company ticker and a question in plain English, and it searches across the last 8 quarters of that company's earnings calls, retrieves the most relevant passages, and generates a direct, sourced answer.

Under the hood, it combines:

- **Retrieval-Augmented Generation (RAG)** to ground every answer in the actual transcript text, rather than relying on a model's general knowledge

- **Cross-encoder re-ranking** to surface the most relevant passages before generation

- A **fully free inference and data stack**, so the project can be run and demoed at zero cost

## Key Features

- **Any-ticker lookup**, enter any valid stock ticker; not limited to a preset list

- **Preset and custom questions**, choose from common questions (future outlook, product launches, M&A activity, market response) or write your own

- **Management consistency check**, compares past-quarter commitments against later outcomes to flag whether management delivered on stated targets

- **Source-linked answers**, every answer links back to the specific quarter/year it was drawn from, with the full retrieved transcript excerpt available on demand

| Component | Tool |
|---|---|
| LLM (answer generation) | Groq API : `openai/gpt-oss-120b` |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Re-ranking | Hugging Face `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | FAISS (Facebook AI Similarity Search) |
| Transcript data | Alpha Vantage `EARNINGS_CALL_TRANSCRIPT` API |
| Orchestration | LangChain |
| Interface | Streamlit |

## How It Works

1. **Fetch transcripts**, pulls the last 8 quarters of earnings call transcripts for the entered ticker via Alpha Vantage

2. **Chunk and embed**, splits transcripts into ~700-character chunks and embeds them locally using a Hugging Face sentence-transformer model

3. **Store in FAISS**, chunks and their metadata (year, quarter, date) are indexed in a FAISS vector store for similarity search

4. **Retrieve and re-rank**, for each question, the top-25 most similar chunks are retrieved, filtered to the most recent quarters, then re-ranked by relevance using a cross-encoder model

5. **Generate an answer**, the top-ranked chunks are passed as context to the LLM (via Groq), which generates a concise, grounded answer with citations to source quarters

## Installation

### 1. Clone the repository

```bash

git clone <your-repo-url>

cd QuarterLens

```

### 2. Create and activate a virtual environment

```bash

python3.10 -m venv venv

venv\Scripts\activate      # Windows

source venv/bin/activate   # macOS/Linux

```

### 3. Install PyTorch (CPU-only build)

Install this before the rest of the requirements, since `sentence-transformers` depends on it:

```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

```

### 4. Install remaining dependencies
```bash

pip install -r requirements.txt

```

### 5. Get your API keys

- **Groq**, [console.groq.com](https://console.groq.com)

- **Alpha Vantage**, [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

### 6. Set up your `.env` file

Create a `.env` file in the project root:

```bash

GROQ_API_KEY='your_groq_api_key_here'

ALPHA_VANTAGE_API_KEY='your_alpha_vantage_api_key_here'

```
### 7. Run the app

```bash

streamlit run main.py

```

## Project Structure

```

QuarterLens/

├── main.py                # Streamlit application UI and control flow

├── backend_functions.py   # Core logic: transcript fetching, embedding, retrieval, answer generation

├── requirements.txt       # Python dependencies

├── .env                   # API keys (not committed)

```

## Limitations

- **Alpha Vantage transcript coverage** is not universal, smaller or less-covered companies may not have transcript data available, even with a valid ticker.

- **Free-tier rate limits** apply on the standard Alpha Vantage key (25 requests/day); each ticker lookup can use up to 12 requests, so testing several tickers in one session may hit that cap without the educational-tier upgrade.

- This is a proof-of-concept research tool, not a substitute for professional investment research or financial advice.
