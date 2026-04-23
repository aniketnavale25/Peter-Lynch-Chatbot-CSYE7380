import json
import math
import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "AMZN", "META", "MSFT", "NFLX", "GOOGL", "TSLA", "NVDA", "JPM", "V"]
START_DATE = "2020-01-01"
END_DATE = "2026-04-22"
CACHE_DIR = "cache"
SLEEP_SECONDS = 8  # slower to reduce rate-limit risk

os.makedirs(CACHE_DIR, exist_ok=True)


def is_json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): is_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [is_json_safe(v) for v in value]
    try:
        return float(value)
    except Exception:
        return str(value)


def clean_info_dict(info: dict) -> dict:
    if not isinstance(info, dict):
        return {}
    return {str(k): is_json_safe(v) for k, v in info.items()}


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def download_history(ticker: str) -> bool:
    hist_path = os.path.join(CACHE_DIR, f"{ticker}_hist.csv")

    # skip existing cache only if it has actual data rows (not just a header)
    if os.path.exists(hist_path):
        try:
            df_check = pd.read_csv(hist_path, nrows=1)
            if len(df_check) > 0:
                print(f"⏭️  History exists, skip -> {hist_path}")
                return True
        except Exception:
            pass

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"⚠️  No history data returned for {ticker}")
            return False

        df.to_csv(hist_path)
        print(f"✅ Saved history -> {hist_path}")
        return True

    except Exception as e:
        print(f"❌ Failed history for {ticker}: {e}")
        return False


def download_fundamentals(ticker: str) -> bool:
    fund_path = os.path.join(CACHE_DIR, f"{ticker}_fund.json")

    # skip existing cache
    if os.path.exists(fund_path) and os.path.getsize(fund_path) > 0:
        print(f"⏭️  Fundamentals exist, skip -> {fund_path}")
        return True

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        if not info or not isinstance(info, dict):
            print(f"⚠️  No fundamentals returned for {ticker}")
            return False

        cleaned = clean_info_dict(info)
        save_json(cleaned, fund_path)
        print(f"✅ Saved fundamentals -> {fund_path}")
        return True

    except Exception as e:
        print(f"❌ Failed fundamentals for {ticker}: {e}")
        return False


def prepare_one_ticker(ticker: str) -> None:
    print(f"\n📥 Processing {ticker} ...")

    hist_ok = download_history(ticker)
    time.sleep(SLEEP_SECONDS)

    fund_ok = download_fundamentals(ticker)
    time.sleep(SLEEP_SECONDS)

    if hist_ok and fund_ok:
        print(f"🎉 {ticker}: history + fundamentals ready")
    elif hist_ok:
        print(f"⚠️  {ticker}: history ready, fundamentals missing")
    elif fund_ok:
        print(f"⚠️  {ticker}: fundamentals ready, history missing")
    else:
        print(f"❌ {ticker}: both history and fundamentals failed")


if __name__ == "__main__":
    print("🚀 Preparing local stock cache...")
    print(f"Tickers: {TICKERS}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Cache dir: {CACHE_DIR}")

    for t in TICKERS:
        prepare_one_ticker(t)

    print("\n✅ Done.")