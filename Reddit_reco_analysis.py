# reddit_reco_analysis.py
# Daily recommendation + evidence report (no order execution).

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# CONFIG: map your Reddit feature columns here
# -----------------------------
COLMAP = {
    "date": "date",                 # YYYY-MM-DD
    "ticker": "ticker",
    "mentions": "mentions",
    "attention_z": "attention_z",   # recommended
    "sentiment_net": "sentiment_net",  # recommended
    "bull_ratio": "bull_ratio",     # optional fallback
    "score_sum": "score_sum",       # optional
    "unique_authors": "unique_authors",  # optional
    "unique_threads": "unique_threads",  # optional
}

BENCHMARK = "SPY"
DEFAULT_FEATURES_PATH = Path("output") / "RC_2025-10" / "ticker_features_daily.csv"


@dataclass
class Params:
    # universe screens
    min_mentions: int = 5
    min_price: float = 2.0
    liquidity_lookback: int = 20
    min_avg_dollar_vol: float = 5e6

    # recommendation construction
    long_short: bool = True
    top_n: int = 10

    # signal definition
    w_attention: float = 1.0
    w_sentiment: float = 0.5

    # evidence / validation
    horizon_days: int = 1          # daily trading basis
    analog_quantile: float = 0.95  # similar-signal analog threshold (per-ticker percentile)
    analog_min_obs: int = 25       # require enough analog observations
    train_ratio: float = 0.6       # walk-forward split

    # cost model for validation (bps per unit turnover)
    transaction_cost_bps: float = 10.0


# -----------------------------
# Loading + normalization
# -----------------------------
def load_reddit_features(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        alt = Path(__file__).resolve().parent / path
        if alt.exists():
            p = alt
        else:
            default_path = Path(__file__).resolve().parent / DEFAULT_FEATURES_PATH
            if p.name == "ticker_features_daily.csv" and default_path.exists():
                p = default_path
            else:
                output_dir = Path(__file__).resolve().parent / "output"
                matches = list(output_dir.rglob(p.name)) if output_dir.exists() else []
                if len(matches) == 1:
                    p = matches[0]
                elif len(matches) > 1:
                    raise FileNotFoundError(
                        f"Multiple matches for '{path}' under {output_dir}: {matches}"
                    )
                else:
                    raise FileNotFoundError(
                        f"Could not find '{path}'. Try passing the full path, e.g. "
                        f"output\\RC_2025-10\\{p.name}"
                    )

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Rename to standardized internal names
    rename = {}
    for internal, src in COLMAP.items():
        if src in df.columns:
            rename[src] = internal
    df = df.rename(columns=rename)

    required = ["date", "ticker", "mentions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. "
                         f"Available columns: {list(df.columns)[:80]}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce").fillna(0).astype(int)

    for c in ["attention_z", "sentiment_net", "bull_ratio", "score_sum", "unique_authors", "unique_threads"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_prices_from_csv(path: str) -> pd.DataFrame:
    """
    Expect columns at minimum: date, ticker, close, volume
    Optional: open, high, low, adj_close
    """
    px = pd.read_csv(path)
    px.columns = [c.strip().lower() for c in px.columns]

    required = {"date", "ticker", "close", "volume"}
    if not required.issubset(set(px.columns)):
        raise ValueError(f"prices_csv must include columns {required}. Got: {set(px.columns)}")

    px["date"] = pd.to_datetime(px["date"]).dt.date
    px["ticker"] = px["ticker"].astype(str).str.upper().str.strip()

    # Fill optional columns
    for c in ["open", "high", "low", "adj_close"]:
        if c not in px.columns:
            px[c] = np.nan

    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["volume"] = pd.to_numeric(px["volume"], errors="coerce")
    return px


def _date_only(value: str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def download_prices_yf(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Install with: pip install yfinance")

    tickers = sorted(set(tickers))
    data = yf.download(
        tickers=tickers,
        start=_date_only(start),
        end=_date_only(end),
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )

    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            tmp = data[t].copy()
            tmp["ticker"] = t
            tmp = tmp.reset_index().rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            })
            frames.append(tmp)
    else:
        tmp = data.copy()
        tmp["ticker"] = tickers[0]
        tmp = tmp.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        frames.append(tmp)

    px = pd.concat(frames, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"]).dt.date
    px["ticker"] = px["ticker"].astype(str).str.upper().str.strip()
    return px


# -----------------------------
# Feature engineering
# -----------------------------
def compute_liquidity(px: pd.DataFrame, lookback: int) -> pd.DataFrame:
    px = px.sort_values(["ticker", "date"]).copy()
    px["dollar_vol"] = px["close"] * px["volume"]
    px["avg_dollar_vol"] = (
        px.groupby("ticker")["dollar_vol"]
          .transform(lambda s: s.rolling(lookback, min_periods=max(5, lookback // 4)).mean())
    )
    return px


def compute_forward_returns(px: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    px = px.sort_values(["ticker", "date"]).copy()
    px["close_fwd"] = px.groupby("ticker")["close"].shift(-horizon_days)
    px[f"ret_fwd_{horizon_days}d"] = px["close_fwd"] / px["close"] - 1.0
    return px


def build_signal(rf: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = rf.copy()

    if "attention_z" not in df.columns:
        df["attention_z"] = np.log1p(df["mentions"]).astype(float)

    if "sentiment_net" not in df.columns:
        if "bull_ratio" in df.columns:
            df["sentiment_net"] = (2.0 * df["bull_ratio"].fillna(0.5) - 1.0).astype(float)
        else:
            df["sentiment_net"] = 0.0

    df["score"] = p.w_attention * df["attention_z"].fillna(0.0) + p.w_sentiment * df["sentiment_net"].fillna(0.0)

    # basic reddit universe constraint
    df = df[df["mentions"] >= p.min_mentions].copy()
    return df


def merge_reddit_prices(sig: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "ticker", "close", "volume", "avg_dollar_vol"]
    out = sig.merge(px[keep], on=["date", "ticker"], how="inner")
    return out


# -----------------------------
# Recommendation logic
# -----------------------------
def recommend_for_date(sig_px: pd.DataFrame, asof_date: str, p: Params) -> pd.DataFrame:
    dt = pd.to_datetime(asof_date).date()
    d = sig_px[sig_px["date"] == dt].copy()

    # screens
    d = d[(d["close"] >= p.min_price)]
    d = d[(d["avg_dollar_vol"].fillna(0.0) >= p.min_avg_dollar_vol)]

    if d.empty:
        return d

    d = d.sort_values("score", ascending=False)

    longs = d.head(p.top_n).copy()
    longs["side"] = "LONG"

    if p.long_short:
        shorts = d.tail(p.top_n).copy()
        shorts = shorts[~shorts["ticker"].isin(longs["ticker"])]
        shorts["side"] = "SHORT"
        rec = pd.concat([longs, shorts], ignore_index=True)
    else:
        rec = longs

    # add rank
    rec["rank_today"] = rec["score"].rank(ascending=False, method="min").astype(int)
    cols = ["date", "ticker", "side", "rank_today", "score", "mentions", "attention_z", "sentiment_net"]
    extra = [c for c in ["unique_authors", "unique_threads", "score_sum"] if c in rec.columns]
    cols += extra
    return rec[cols].sort_values(["side", "rank_today"], ascending=[True, True])


# -----------------------------
# Evidence: historical analogs
# -----------------------------
def analog_evidence_for_ticker(
    ticker: str,
    sig_px: pd.DataFrame,
    px_ret: pd.DataFrame,
    asof_date: str,
    p: Params
) -> Dict[str, float]:
    """
    Evidence based on ticker's own history:
      - compute ticker's score percentile threshold (e.g. 95th)
      - filter past dates with score >= threshold (excluding asof_date and last horizon days)
      - compute forward return distribution stats
    """
    dt = pd.to_datetime(asof_date).date()

    tdf = sig_px[sig_px["ticker"] == ticker].copy()
    if tdf.empty:
        return {"n_analogs": 0}

    # percentile threshold on *past* scores only
    tdf_past = tdf[tdf["date"] < dt].copy()
    if len(tdf_past) < p.analog_min_obs:
        return {"n_analogs": 0}

    thr = float(np.nanquantile(tdf_past["score"].values, p.analog_quantile))

    analog_dates = tdf_past[tdf_past["score"] >= thr]["date"].unique().tolist()
    if not analog_dates:
        return {"n_analogs": 0, "score_thr": thr}

    # pull forward returns from px_ret
    rcol = f"ret_fwd_{p.horizon_days}d"
    r = (
        px_ret[(px_ret["ticker"] == ticker) & (px_ret["date"].isin(analog_dates))][rcol]
        .dropna()
        .astype(float)
        .values
    )

    n = int(len(r))
    if n < max(10, p.analog_min_obs // 3):
        return {"n_analogs": n, "score_thr": thr}

    mean = float(np.mean(r))
    med = float(np.median(r))
    hit = float(np.mean(r > 0.0))
    std = float(np.std(r, ddof=1)) if n > 1 else np.nan
    tstat = float(mean / (std / math.sqrt(n))) if (std is not None and std > 1e-12) else np.nan

    return {
        "n_analogs": n,
        "score_thr": thr,
        "analog_mean_ret": mean,
        "analog_median_ret": med,
        "analog_hit_rate": hit,
        "analog_tstat": tstat,
    }


def context_stats(px: pd.DataFrame, ticker: str, asof_date: str) -> Dict[str, float]:
    """
    Simple context: recent momentum + volatility.
    """
    dt = pd.to_datetime(asof_date).date()
    tpx = px[(px["ticker"] == ticker) & (px["date"] <= dt)].sort_values("date").copy()
    if len(tpx) < 30:
        return {}

    # daily returns
    r = tpx["close"].pct_change().dropna().values
    last_5 = float(np.prod(1.0 + r[-5:]) - 1.0) if len(r) >= 5 else np.nan
    last_20 = float(np.prod(1.0 + r[-20:]) - 1.0) if len(r) >= 20 else np.nan
    vol_20 = float(np.std(r[-20:], ddof=1)) if len(r) >= 20 else np.nan

    return {"ret_5d": last_5, "ret_20d": last_20, "vol_20d": vol_20}


# -----------------------------
# Validation: walk-forward ba
#
# (Walk-forward backtest stub removed in this script; see reddit_daily_trade_validate.py)


# -----------------------------
# Recommendation report (print + optional CSV)
# -----------------------------
def build_reco_report(
    rec: pd.DataFrame,
    sig_px: pd.DataFrame,
    px_ret: pd.DataFrame,
    px: pd.DataFrame,
    asof_date: str,
    p: Params,
) -> pd.DataFrame:
    rows = []
    for row in rec.itertuples(index=False):
        ticker = row.ticker
        evidence = analog_evidence_for_ticker(ticker, sig_px, px_ret, asof_date, p)
        ctx = context_stats(px, ticker, asof_date)
        payload = {
            "date": row.date,
            "ticker": ticker,
            "side": row.side,
            "rank_today": row.rank_today,
            "score": row.score,
            "mentions": row.mentions,
            "attention_z": row.attention_z,
            "sentiment_net": row.sentiment_net,
        }
        for c in ["unique_authors", "unique_threads", "score_sum"]:
            if hasattr(row, c):
                payload[c] = getattr(row, c)
        # add realized forward return (if available) and net returns after cost
        rcol = f"ret_fwd_{p.horizon_days}d"
        try:
            val = px_ret.loc[(px_ret["ticker"] == ticker) & (px_ret["date"] == row.date), rcol]
            realized = float(val.iloc[0]) if not val.empty else float("nan")
        except Exception:
            realized = float("nan")

        payload["realized_ret"] = realized
        # net returns after single-sided and round-trip transaction costs (bps -> decimal)
        tc = float(p.transaction_cost_bps) / 10000.0
        payload["net_ret_after_cost"] = realized - tc if not math.isnan(realized) else float("nan")
        payload["net_ret_after_roundtrip_cost"] = realized - 2.0 * tc if not math.isnan(realized) else float("nan")

        payload.update(evidence)
        payload.update(ctx)

        # replace 5d/20d nets with 1-day net return (round-trip costs)
        # compute net_1day from realized forward return minus round-trip transaction costs
        try:
            net_1day = payload.get("net_ret_after_roundtrip_cost", float("nan"))
        except Exception:
            net_1day = float("nan")
        payload["net_1day"] = net_1day

        # remove ret_5d and ret_20d if present
        if "ret_5d" in payload:
            payload.pop("ret_5d", None)
        if "ret_20d" in payload:
            payload.pop("ret_20d", None)

        rows.append(payload)

    if not rows:
        return rec.head(0)
    return pd.DataFrame(rows)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate daily Reddit-based trade recommendations with evidence.")
    ap.add_argument(
        "--reddit_features",
        default=str(DEFAULT_FEATURES_PATH),
        help="Path to ticker_features_daily.csv or .parquet",
    )
    ap.add_argument("--asof_date", required=True, help="Date to recommend for (YYYY-MM-DD)")
    ap.add_argument("--top_n", type=int, default=10, help="Top N longs (and shorts if long_short)")
    ap.add_argument("--long_only", action="store_true", help="Only recommend longs")

    ap.add_argument("--min_mentions", type=int, default=5)
    ap.add_argument("--min_price", type=float, default=2.0)
    ap.add_argument("--min_avg_dollar_vol", type=float, default=5e6)
    ap.add_argument("--liquidity_lookback", type=int, default=20)

    ap.add_argument("--w_attention", type=float, default=1.0)
    ap.add_argument("--w_sentiment", type=float, default=0.5)
    ap.add_argument("--horizon_days", type=int, default=1)
    ap.add_argument("--analog_quantile", type=float, default=0.95)
    ap.add_argument("--analog_min_obs", type=int, default=25)

    ap.add_argument("--use_yfinance", action="store_true", help="Download prices from Yahoo via yfinance")
    ap.add_argument("--prices_csv", default="", help="CSV with price history (date,ticker,close,volume)")
    ap.add_argument("--out_csv", default="", help="Optional output CSV for recommendations + evidence")

    args = ap.parse_args()

    params = Params(
        min_mentions=args.min_mentions,
        min_price=args.min_price,
        liquidity_lookback=args.liquidity_lookback,
        min_avg_dollar_vol=args.min_avg_dollar_vol,
        long_short=(not args.long_only),
        top_n=args.top_n,
        w_attention=args.w_attention,
        w_sentiment=args.w_sentiment,
        horizon_days=args.horizon_days,
        analog_quantile=args.analog_quantile,
        analog_min_obs=args.analog_min_obs,
    )

    rf = load_reddit_features(args.reddit_features)
    sig = build_signal(rf, params)
    if sig.empty:
        raise RuntimeError("No rows after Reddit filters. Lower --min_mentions or check input data.")

    min_dt = min(sig["date"])
    max_dt = max(sig["date"])
    dt = pd.to_datetime(args.asof_date).date()
    if dt < min_dt or dt > max_dt:
        print(f"asof_date {dt} not in Reddit data range [{min_dt}, {max_dt}].")
        return

    tickers = sorted(sig["ticker"].unique().tolist())
    if args.use_yfinance:
        start = (pd.to_datetime(min_dt) - pd.Timedelta(days=params.liquidity_lookback * 2)).strftime("%Y-%m-%d")
        end = (pd.to_datetime(max_dt) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        px = download_prices_yf(tickers, start=start, end=end)
    else:
        if not args.prices_csv:
            raise ValueError("Provide --prices_csv or use --use_yfinance.")
        px = load_prices_from_csv(args.prices_csv)

    px = compute_liquidity(px, lookback=params.liquidity_lookback)
    sig_px = merge_reddit_prices(sig, px)
    if sig_px.empty:
        raise RuntimeError("No overlap between Reddit data and price data after joins/filters.")

    rec = recommend_for_date(sig_px, args.asof_date, params)
    if rec.empty:
        print("No recommendations after screens for the given date.")
        return

    px_ret = compute_forward_returns(px, params.horizon_days)
    report = build_reco_report(rec, sig_px, px_ret, px, args.asof_date, params)

    print(f"\nRecommendations for {args.asof_date} (top_n={args.top_n}, long_short={params.long_short})")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(report)

    if args.out_csv:
        report.to_csv(args.out_csv, index=False)
        print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()
