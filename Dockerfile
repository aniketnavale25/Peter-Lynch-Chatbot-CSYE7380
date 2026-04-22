FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    langchain \
    langchain-core \
    langchain-community \
    langchain-huggingface \
    langchain-text-splitters \
    chromadb \
    langchain-chroma \
    sentence-transformers \
    torch \
    openai \
    pypdf \
    pandas \
    openpyxl \
    streamlit \
    yfinance \
    plotly \
    numpy \
    python-dotenv

COPY . .

EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]