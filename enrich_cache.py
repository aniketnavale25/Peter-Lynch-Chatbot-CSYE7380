"""
enrich_cache.py — add financial metrics to the raw OHLCV cache CSVs.

For each ticker in cache/{ticker}_hist.csv this module:
  1. Fetches annual financial statements from SEC EDGAR (free, no API key).
  2. Computes the ten metrics required by the Lynch Stock Analyzer:
       Price, P/E Ratio, EPS Growth, ROE, Dividend Yield,
       PEG Ratio, Debt/Equity, P/S Ratio, Current Ratio, Insider Ownership
  3. Appends those metrics as constant columns to the OHLCV data.
  4. Writes cache/{ticker}_enriched.csv.

Run once (or to refresh):
    python enrich_cache.py
"""

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
EDGAR_HEADERS = {"User-Agent": "peter-lynch-chatbot contact@example.com"}
TICKERS = ["AAPL", "AMZN", "META", "MSFT", "NFLX", "GOOGL", "TSLA", "NVDA", "JPM", "V"]
REQUEST_DELAY = 0.5   # seconds between EDGAR calls – SEC fair-use policy


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_cik_map() -> dict[str, str]:
    """Return {TICKER: zero-padded-CIK} for all public companies."""
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=EDGAR_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}


def get_gaap_facts(cik: str) -> dict:
    """Return the us-gaap facts dict for *cik*."""
    r = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=EDGAR_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("facts", {}).get("us-gaap", {})


def _annual_entries(facts: dict, *concepts: str) -> list:
    """Return all 10-K / 20-F annual entries for the first matching concept."""
    for concept in concepts:
        if concept not in facts:
            continue
        for unit_entries in facts[concept]["units"].values():
            annual = [
                e for e in unit_entries
                if e.get("form") in ("10-K", "20-F")
                and e.get("fp") in ("FY", "Q4")
            ]
            if annual:
                return sorted(annual, key=lambda x: x["end"])
    return []


def latest_val(facts: dict, *concepts: str) -> float | None:
    """Most recent annual value across *concepts*."""
    entries = _annual_entries(facts, *concepts)
    return float(entries[-1]["val"]) if entries else None


def two_latest_vals(facts: dict, *concepts: str) -> tuple[float | None, float | None]:
    """Two most recent annual values (latest, one-year-prior)."""
    entries = _annual_entries(facts, *concepts)
    if len(entries) >= 2:
        return float(entries[-1]["val"]), float(entries[-2]["val"])
    if len(entries) == 1:
        return float(entries[-1]["val"]), None
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(ticker: str, facts: dict) -> dict:
    """
    Compute all ten Lynch metrics from EDGAR facts + the local price CSV.
    All percentage metrics are stored as plain percentages (e.g. 14.7 = 14.7 %).
    stock_cache.py divides by 100 where app.py expects a decimal fraction.
    """
    # ── Price from local CSV ─────────────────────────────────────────────────
    hist_path = CACHE_DIR / f"{ticker}_hist.csv"
    price: float | None = None
    if hist_path.exists():
        df = pd.read_csv(hist_path, index_col="Date", parse_dates=True)
        if not df.empty and "Close" in df.columns:
            price = float(df["Close"].dropna().iloc[-1])

    # ── Raw EDGAR values ─────────────────────────────────────────────────────
    eps_cur, eps_prev = two_latest_vals(
        facts,
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    )
    revenue = latest_val(
        facts,
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    )
    net_income = latest_val(facts, "NetIncomeLoss", "ProfitLoss")
    equity = latest_val(
        facts,
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )
    lt_debt = latest_val(
        facts,
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligation",
    ) or 0.0
    st_debt = latest_val(
        facts,
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "NotesPayableCurrent",
        "CommercialPaper",
    ) or 0.0
    current_assets = latest_val(facts, "AssetsCurrent")
    current_liab   = latest_val(facts, "LiabilitiesCurrent")
    shares         = latest_val(
        facts,
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssuedAndOutstanding",
    )
    dividends_paid = latest_val(
        facts,
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
        "DividendsCommonStockCash",
        "DividendsCash",
    )

    # ── Derived metrics ───────────────────────────────────────────────────────

    # P/E Ratio
    pe_ratio: float | None = None
    if price and eps_cur and eps_cur > 0:
        pe_ratio = round(price / eps_cur, 2)

    # EPS Growth  (%)
    eps_growth: float | None = None
    if eps_cur is not None and eps_prev is not None and eps_prev != 0:
        eps_growth = round((eps_cur - eps_prev) / abs(eps_prev) * 100, 2)

    # PEG Ratio
    peg_ratio: float | None = None
    if pe_ratio and eps_growth and eps_growth > 0:
        peg_ratio = round(pe_ratio / eps_growth, 2)

    # ROE  (%)
    roe: float | None = None
    if net_income is not None and equity and equity > 0:
        roe = round(net_income / equity * 100, 2)

    # Dividend Yield  (%)
    dividend_yield: float | None = None
    if dividends_paid and shares and shares > 0 and price:
        dps = abs(dividends_paid) / shares
        dividend_yield = round(dps / price * 100, 4)

    # Debt / Equity  (%)
    debt_equity: float | None = None
    total_debt = lt_debt + st_debt
    if total_debt and equity and equity > 0:
        debt_equity = round(total_debt / equity * 100, 2)

    # P/S Ratio
    ps_ratio: float | None = None
    if price and revenue and revenue > 0 and shares and shares > 0:
        market_cap = price * shares
        ps_ratio = round(market_cap / revenue, 2)

    # Current Ratio
    current_ratio: float | None = None
    if current_assets and current_liab and current_liab > 0:
        current_ratio = round(current_assets / current_liab, 2)

    # Insider Ownership — not available in company-facts API; left as None
    insider_ownership: float | None = None

    return {
        "Price":             price,
        "PE_Ratio":          pe_ratio,
        "EPS_Growth":        eps_growth,
        "ROE":               roe,
        "Dividend_Yield":    dividend_yield,
        "PEG_Ratio":         peg_ratio,
        "Debt_Equity":       debt_equity,
        "PS_Ratio":          ps_ratio,
        "Current_Ratio":     current_ratio,
        "Insider_Ownership": insider_ownership,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_ticker(ticker: str, cik: str) -> bool:
    """
    Load cache/{ticker}_hist.csv, append metric columns, write
    cache/{ticker}_enriched.csv.  Returns True on success.
    """
    hist_path = CACHE_DIR / f"{ticker}_hist.csv"
    out_path  = CACHE_DIR / f"{ticker}_enriched.csv"

    if not hist_path.exists():
        print(f"  [{ticker}] skipped – hist CSV not found")
        return False

    df = pd.read_csv(hist_path, index_col="Date", parse_dates=True)
    if df.empty:
        print(f"  [{ticker}] skipped – hist CSV is empty")
        return False

    print(f"  [{ticker}] fetching EDGAR facts (CIK {cik})…")
    try:
        facts = get_gaap_facts(cik)
    except Exception as exc:
        print(f"  [{ticker}] EDGAR error: {exc}")
        return False

    metrics = compute_metrics(ticker, facts)

    for col, val in metrics.items():
        df[col] = val            # broadcast scalar → all rows

    df.to_csv(out_path)
    computed = {k: v for k, v in metrics.items() if v is not None}
    print(f"  [{ticker}] saved {out_path.name}  "
          f"({len(df)} rows, {len(computed)}/{len(metrics)} metrics computed)")
    return True


def enrich_all(tickers: list[str] = TICKERS) -> None:
    """Enrich every ticker in *tickers* using SEC EDGAR data."""
    print("Fetching CIK map from SEC EDGAR…")
    try:
        cik_map = get_cik_map()
    except Exception as exc:
        print(f"ERROR: could not load CIK map – {exc}")
        return
    time.sleep(REQUEST_DELAY)

    ok = failed = 0
    for ticker in tickers:
        cik = cik_map.get(ticker)
        if not cik:
            print(f"  [{ticker}] CIK not found – skipped")
            failed += 1
            continue
        success = enrich_ticker(ticker, cik)
        ok += success
        failed += not success
        time.sleep(REQUEST_DELAY)

    print(f"\nDone – {ok} enriched, {failed} failed.")


if __name__ == "__main__":
    enrich_all()
