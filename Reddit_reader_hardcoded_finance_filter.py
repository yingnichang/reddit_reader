from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import polars as pl


# -----------------------------
# Finance-only filtering (subreddits)
# -----------------------------
DEFAULT_FINANCE_SUBREDDITS = {
    "wallstreetbets",
    "stocks",
    "investing",
    "stockmarket",
    "options",
    "thetagang",
    "daytrading",
    "algotrading",
    "securityanalysis",
    "valueinvesting",
    "pennystocks",
    "smallstreetbets",
    "superstonk",
    "spacs",
    # Crypto-market sentiment (optional; keep/remove as needed)
    "cryptocurrency",
    "bitcoinmarkets",
    "ethtrader",
}

# Hardcoded finance filtering behavior
# Set ENABLE_FINANCE_SUBREDDIT_FILTER = False if you want to process all subreddits.
ENABLE_FINANCE_SUBREDDIT_FILTER = True
FINANCE_SUBREDDIT_ALLOWLIST = set(DEFAULT_FINANCE_SUBREDDITS)

# Optional: additional keyword-based finance filter (kept OFF by default).
# If enabled, a row must contain at least one keyword in (title + body/selftext).
ENABLE_FINANCE_KEYWORD_FILTER = False
FINANCE_KEYWORDS = [
    "stock", "stocks", "share", "shares", "equity", "etf", "index", "earnings",
    "guidance", "dividend", "buy", "sell", "long", "short", "call", "put",
    "options", "gamma", "iv", "vol", "volume", "market cap", "valuation",
    "crypto", "bitcoin", "btc", "ethereum", "eth", "token", "defi",
]

def _build_keyword_regex(keywords: list[str]) -> str:
    """Build a single case-insensitive regex for keyword matching in Polars."""
    parts: list[str] = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        # Escape regex, then allow flexible whitespace for multi-word terms
        esc = re.escape(kw.upper())
        esc = esc.replace(r"\ ", r"\\s+")
        parts.append(esc)
    if not parts:
        return r"$^"  # match nothing
    return r"(?:" + "|".join(parts) + r")"

# Uppercased content is matched against this pattern when ENABLE_FINANCE_KEYWORD_FILTER=True
FINANCE_KEYWORD_REGEX = _build_keyword_regex(FINANCE_KEYWORDS)


# -----------------------------
# Ticker extraction & filters
# -----------------------------
# Require 3+ letters for bare tickers to cut noise like "IF" while still
# allowing cashtags such as $TSLA.
TICKER_REGEX = r"\b[A-Z]{3,5}\b"
CASHTAG_REGEX = r"\$[A-Z]{2,5}\b"

# Finance-context gate to reduce false positives when a bare ticker appears.
FINANCE_CONTEXT_REGEX = (
    r"\b(STOCK|SHARE|SHARES|BUY|SELL|EARNINGS|CALL|PUT|OPTION|OPTIONS|HOLD|LONG|"
    r"SHORT|BULL|BEAR|PRICE|PT|TARGET|DIVIDEND|GUIDANCE|EPS|IPO|NASDAQ|NYSE|"
    r"MARKET|VOLUME|FLOAT|BAGS|BAGHOLD|DIP|PUMP|DUMP|CATALYST|FDA|APPROVAL|"
    r"UPGRADE|DOWNGRADE|RATING|GUIDE|GUIDANCE|BEAT|MISS)\b"
)

# Lightweight stance proxy (lexical). This is NOT a full sentiment model,
# but it is fast and usually adds value beyond raw mention counts.
BULLISH_REGEX = (
    r"\b(BUY|BUYING|LONG|BULL|BULLISH|CALLS?|MOON|RIP|BREAKOUT|PUMP|SQUEEZE|"
    r"UPSIDE|UNDERVALUED|BEAT|CRUSH|STRONG|GUIDANCE RAISE|RAISED|UPGRADE)\b"
)
BEARISH_REGEX = (
    r"\b(SELL|SELLING|SHORT|BEAR|BEARISH|PUTS?|DUMP|RUG|CRASH|DOWNSIDE|"
    r"OVERVALUED|MISS|WEAK|GUIDANCE CUT|CUT|DOWNGRADE|FRAUD|BAGHOLD)\b"
)

STOPWORDS = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "FOR", "FROM", "I",
    "IN", "IS", "IT", "NO", "NOT", "OF", "ON", "OR", "THE", "THIS",
    "THAT", "TO", "US", "WE", "WITH", "YOU",
    "IF", "AM", "PM", "DO", "GO", "ME", "MY", "HE", "SHE", "HIM", "HER",
    "SO", "UP", "TV", "OK", "BUY", "BUT", "JUST", "ITS", "ANY", "ALL",
    "ONE", "ARE", "NEW", "NOW", "CAN", "OUT", "FOR",
}
# Even if a symbol exists, these are often ambiguous in text; blacklist them.
AMBIGUOUS_TICKERS = {
    "ALL", "ONE", "ARE", "NOW", "NEW", "FOR", "OUT", "CAN",
    "BUY", "BUT", "JUST", "ITS", "ANY",
}

YAHOO_SCREENER_URL = (
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    "?scrIds=most_actives&start={start}&count={count}"
)


def fetch_yahoo_most_actives(
    limit: int,
    cache_path: Path,
    max_age_hours: int = 24,
    force_refresh: bool = False,
) -> list[str]:
    """
    Pull an allowlist of liquid names to reduce ticker false positives.
    Cached on disk to avoid repeated network requests.
    """
    if cache_path.exists() and not force_refresh:
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds <= max_age_hours * 3600:
            return [
                line.strip().upper()
                for line in cache_path.read_text(encoding="utf8").splitlines()
                if line.strip()
            ]

    tickers: list[str] = []
    start = 0
    page_size = 100
    headers = {"User-Agent": "Mozilla/5.0"}

    while len(tickers) < limit:
        count = min(page_size, limit - start)
        url = YAHOO_SCREENER_URL.format(start=start, count=count)
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print("Failed to fetch Yahoo Finance most actives:", repr(exc))
            break

        results = payload.get("finance", {}).get("result", [])
        quotes = results[0].get("quotes", []) if results else []
        if not quotes:
            break

        for q in quotes:
            symbol = (q.get("symbol") or "").strip()
            if symbol:
                tickers.append(symbol.upper())

        start += page_size

    if tickers:
        cache_path.write_text("\n".join(tickers), encoding="utf8")
    return tickers


# -----------------------------
# SQLite helpers
# -----------------------------
def init_db(conn: sqlite3.Connection, track_unique_authors: bool, track_unique_threads: bool) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS counts (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            mentions INTEGER NOT NULL,
            score_sum INTEGER NOT NULL,
            bull INTEGER NOT NULL,
            bear INTEGER NOT NULL,
            PRIMARY KEY(date, ticker)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_files (
            path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            processed_at REAL NOT NULL
        )
        """
    )
    if track_unique_authors:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS authors (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                author TEXT NOT NULL,
                PRIMARY KEY(date, ticker, author)
            )
            """
        )
    if track_unique_threads:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                PRIMARY KEY(date, ticker, thread_id)
            )
            """
        )
    conn.commit()


def file_already_processed(conn: sqlite3.Connection, path: Path) -> bool:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT mtime FROM processed_files WHERE path = ?",
        (str(path),),
    ).fetchone()
    if not row:
        return False
    prev_mtime = float(row[0])
    return abs(prev_mtime - path.stat().st_mtime) < 1e-6


def mark_file_processed(conn: sqlite3.Connection, path: Path) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO processed_files(path, mtime, processed_at)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            mtime = excluded.mtime,
            processed_at = excluded.processed_at
        """,
        (str(path), float(path.stat().st_mtime), time.time()),
    )
    conn.commit()


# -----------------------------
# Core extraction pipeline (per file)
# -----------------------------
def process_parquet_file(
    parquet_path: Path,
    conn: sqlite3.Connection,
    allowlist: set[str],
    use_allowlist: bool,
    track_unique_authors: bool,
    track_unique_threads: bool,
    subreddit_filter: set[str] | None,
) -> None:
    """
    Extract per-(date,ticker) aggregates from one parquet file and upsert into SQLite.
    If track_unique_authors/threads are enabled, exact unique sets are stored via
    INSERT OR IGNORE (can be large; but accurate).
    """
    # Build a lazy pipeline per-file and only collect small aggregated results
    lf = pl.scan_parquet(str(parquet_path))
    schema = lf.schema

    # Unify content: title + text (submissions), or text only (comments)
    title_expr = pl.col("title") if "title" in schema else pl.lit("")
    text_expr = pl.col("text") if "text" in schema else pl.lit("")
    lf = lf.with_columns(
        title_nc=title_expr.cast(pl.Utf8, strict=False).fill_null(""),
        text_nc=text_expr.cast(pl.Utf8, strict=False).fill_null(""),
    ).with_columns(
        content=pl.concat_str([pl.col("title_nc"), pl.lit("\n"), pl.col("text_nc")]),
        content_upper=pl.concat_str([pl.col("title_nc"), pl.lit("\n"), pl.col("text_nc")]).str.to_uppercase(),
        date=pl.from_epoch(pl.col("created_utc"), time_unit="s").dt.date(),
        score_i=pl.col("score").cast(pl.Int64, strict=False).fill_null(0),
        author_s=pl.col("author").cast(pl.Utf8, strict=False).fill_null(""),
        thread_id=pl.when(pl.col("type") == "comment")
        .then(pl.col("link_id").cast(pl.Utf8, strict=False))
        .otherwise(pl.col("id").cast(pl.Utf8, strict=False))
        .fill_null(""),
    )

    # Basic hygiene
    lf = lf.filter(
        (pl.col("content").str.strip_chars().str.len_chars() > 0)
        & (pl.col("content").str.strip_chars() != "[removed]")
        & (pl.col("content").str.strip_chars() != "[deleted]")
    )

    # Optional: restrict to finance-focused subreddits (recommended if your dumps include many topics)
    if subreddit_filter:
        lf = (
            lf.with_columns(
                subreddit_l=pl.col("subreddit").cast(pl.Utf8, strict=False).fill_null("").str.to_lowercase()
            )
            .filter(pl.col("subreddit_l").is_in(list(subreddit_filter)))
            .drop("subreddit_l")
        )

    # Optional: keyword-based finance filter (in addition to subreddit filtering)
    if ENABLE_FINANCE_KEYWORD_FILTER:
        lf = lf.filter(pl.col("content_upper").str.contains(FINANCE_KEYWORD_REGEX))

    # Context & stance flags
    lf = lf.with_columns(
        has_fin_context=pl.col("content_upper").str.contains(FINANCE_CONTEXT_REGEX),
        is_bullish=pl.col("content_upper").str.contains(BULLISH_REGEX),
        is_bearish=pl.col("content_upper").str.contains(BEARISH_REGEX),
        tickers=pl.concat_list(
            [
                pl.col("content_upper").str.extract_all(TICKER_REGEX),
                pl.col("content_upper").str.extract_all(CASHTAG_REGEX),
            ]
        ).list.unique(),
    )

    # Explode tickers and keep a cashtag flag for gating
    lf = lf.explode("tickers").filter(pl.col("tickers").is_not_null() & (pl.col("tickers").str.len_chars() > 0))
    lf = lf.with_columns(
        is_cashtag=pl.col("tickers").str.starts_with("$"),
        ticker=pl.col("tickers").str.replace(r"^\$", ""),
    ).drop("tickers")

    # Finance gating for bare tickers: either cashtag, or finance context in the same row
    lf = lf.filter(pl.col("is_cashtag") | pl.col("has_fin_context"))

    # Remove common English tokens and ambiguous tickers
    lf = lf.filter(~pl.col("ticker").is_in(list(STOPWORDS)))
    lf = lf.filter(~pl.col("ticker").is_in(list(AMBIGUOUS_TICKERS)))

    # Optional allowlist filter; if allowlist is empty, we fallback to "no allowlist"
    if use_allowlist and allowlist:
        lf = lf.filter(pl.col("ticker").is_in(list(allowlist)))

    # Aggregate additive metrics
    grp = (
        lf.group_by(["date", "ticker"])
        .agg(
            [
                pl.len().alias("mentions"),
                pl.col("score_i").sum().alias("score_sum"),
                pl.col("is_bullish").cast(pl.Int64).sum().alias("bull"),
                pl.col("is_bearish").cast(pl.Int64).sum().alias("bear"),
            ]
        )
        .collect()
    )

    if not grp.is_empty():
        cur = conn.cursor()
        rows = [(str(r[0]), r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in grp.iter_rows()]
        cur.executemany(
            """
            INSERT INTO counts(date, ticker, mentions, score_sum, bull, bear)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, ticker) DO UPDATE SET
                mentions = mentions + excluded.mentions,
                score_sum = score_sum + excluded.score_sum,
                bull = bull + excluded.bull,
                bear = bear + excluded.bear
            """,
            rows,
        )
        conn.commit()

    # Track unique authors / threads (exact, via de-dup tables)
    if track_unique_authors:
        authors_df = lf.select(["date", "ticker", "author_s"]).unique().collect()
        if not authors_df.is_empty():
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR IGNORE INTO authors(date, ticker, author) VALUES (?, ?, ?)",
                [(str(r[0]), r[1], r[2]) for r in authors_df.iter_rows()],
            )
            conn.commit()

    if track_unique_threads:
        threads_df = lf.select(["date", "ticker", "thread_id"]).filter(pl.col("thread_id").str.len_chars() > 0).unique().collect()
        if not threads_df.is_empty():
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR IGNORE INTO threads(date, ticker, thread_id) VALUES (?, ?, ?)",
                [(str(r[0]), r[1], r[2]) for r in threads_df.iter_rows()],
            )
            conn.commit()


# -----------------------------
# Export features and z-scores
# -----------------------------
def export_features(
    conn: sqlite3.Connection,
    out_csv: Path,
    out_parquet: Path,
    rolling_window: int,
    min_periods: int,
    plot_top10: bool,
    plot_path: Path | None,
) -> None:
    """
    Build a daily feature table:
      - mentions, score_sum
      - bullish/bearish counts + net/ratio
      - optional unique_authors/unique_threads (if tracked)
      - rolling z-score of mentions as an "attention surprise" metric
    """
    # Pull base table
    base = pl.read_database(
        query="SELECT date, ticker, mentions, score_sum, bull, bear FROM counts",
        connection=conn,
    )

    # Join unique author/thread counts if available
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "authors" in tables:
        au = pl.read_database(
            query="SELECT date, ticker, COUNT(*) AS unique_authors FROM authors GROUP BY date, ticker",
            connection=conn,
        )
        base = base.join(au, on=["date", "ticker"], how="left")
    else:
        base = base.with_columns(unique_authors=pl.lit(None, dtype=pl.Int64))

    if "threads" in tables:
        th = pl.read_database(
            query="SELECT date, ticker, COUNT(*) AS unique_threads FROM threads GROUP BY date, ticker",
            connection=conn,
        )
        base = base.join(th, on=["date", "ticker"], how="left")
    else:
        base = base.with_columns(unique_threads=pl.lit(None, dtype=pl.Int64))

    # Derived metrics + rolling attention z-score
    df = (
        base.with_columns(
            date=pl.col("date").cast(pl.Utf8, strict=False).str.to_date(),
            bull=pl.col("bull").cast(pl.Int64),
            bear=pl.col("bear").cast(pl.Int64),
            mentions=pl.col("mentions").cast(pl.Int64),
            score_sum=pl.col("score_sum").cast(pl.Int64),
        )
        .with_columns(
            sentiment_net=(pl.col("bull") - pl.col("bear")).cast(pl.Int64),
            bull_ratio=pl.when((pl.col("bull") + pl.col("bear")) > 0)
            .then(pl.col("bull") / (pl.col("bull") + pl.col("bear")))
            .otherwise(None),
        )
        .sort(["ticker", "date"])
        .with_columns(
            mentions_roll_mean=pl.col("mentions").rolling_mean(rolling_window, min_periods=min_periods).over("ticker"),
            mentions_roll_std=pl.col("mentions").rolling_std(rolling_window, min_periods=min_periods).over("ticker"),
        )
        .with_columns(
            attention_z=pl.when(pl.col("mentions_roll_std") > 0)
            .then((pl.col("mentions") - pl.col("mentions_roll_mean")) / pl.col("mentions_roll_std"))
            .otherwise(None),
        )
        .sort(["date", "mentions"], descending=[False, True])
    )

    df.write_csv(out_csv)
    df.write_parquet(out_parquet)

    print("Wrote daily ticker features CSV:", out_csv)
    print("Wrote daily ticker features Parquet:", out_parquet)
    print("Preview (top 25 rows by date then mentions):")
    print(df.head(25))

    if plot_top10:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipping top-10 chart.")
            return

        top10 = (
            df.group_by("ticker")
            .agg(pl.col("mentions").sum().alias("total_mentions"))
            .sort("total_mentions", descending=True)
            .head(10)
            .get_column("ticker")
            .to_list()
        )
        if not top10:
            print("No tickers available for top-10 chart.")
            return

        plot_df = (
            df.filter(pl.col("ticker").is_in(top10))
            .select(["date", "ticker", "mentions"])
            .sort(["date", "ticker"])
        )

        try:
            import pandas as pd
        except ImportError:
            print("pandas is not installed; skipping top-10 chart.")
            return

        pdf = plot_df.to_pandas()
        pivot = (
            pdf.pivot_table(index="date", columns="ticker", values="mentions", aggfunc="sum")
            .fillna(0)
            .sort_index()
        )

        fig_path = plot_path if plot_path else out_csv.with_suffix(".top10.png")
        plt.figure(figsize=(12, 6))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], label=col)
        plt.title("Top 10 Tickers by Daily Mentions")
        plt.xlabel("Date")
        plt.ylabel("Mentions")
        plt.legend(ncol=5, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("Wrote top-10 chart:", fig_path)


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(
        description="Aggregate Reddit Parquet dumps into daily ticker features for predictive research."
    )
    p.add_argument("--input_dir", required=True, help="Folder containing parquet files (recursively searched)")
    p.add_argument("--glob", default="*.parquet", help="Glob for parquet discovery (default: *.parquet)")
    p.add_argument("--db", default="", help="SQLite DB path (default: <input_dir>/ticker_features.db)")
    p.add_argument("--out_csv", default="", help="Output CSV path (default: <input_dir>/ticker_features_daily.csv)")
    p.add_argument("--out_parquet", default="", help="Output Parquet path (default: <input_dir>/ticker_features_daily.parquet)")
    p.add_argument("--reset_db", action="store_true", help="Delete existing DB and rebuild")
    p.add_argument("--max_files", type=int, default=0, help="Process only first N files (0 = all)")
    p.add_argument("--track_unique_authors", action="store_true", help="Track exact unique authors per (date,ticker)")
    p.add_argument("--track_unique_threads", action="store_true", help="Track exact unique threads per (date,ticker)")

    # Allowlist controls
    p.add_argument("--use_allowlist", action="store_true", help="Filter tickers to an allowlist (recommended)")
    p.add_argument("--allowlist_limit", type=int, default=1500, help="Yahoo most-actives size")
    p.add_argument("--allowlist_cache", default="", help="Cache path for allowlist (default: <input_dir>/yahoo_most_actives_<limit>.txt)")
    p.add_argument("--force_refresh_allowlist", action="store_true", help="Force refresh allowlist from network")
    p.add_argument("--allowlist_file", default="", help="Optional newline-separated tickers to union into allowlist")

    # Rolling features
    p.add_argument("--rolling_window", type=int, default=30, help="Rolling window (days) for attention_z")
    p.add_argument("--min_periods", type=int, default=10, help="Min periods before attention_z is computed")
    p.add_argument("--plot_top10", action="store_true", help="Write a line chart for top 10 tickers by mentions")
    p.add_argument("--plot_path", default="", help="Chart output path (default: <out_csv>.top10.png)")

    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {in_dir}")

    db_path = Path(args.db) if args.db else (in_dir / "ticker_features.db")
    out_csv = Path(args.out_csv) if args.out_csv else (in_dir / "ticker_features_daily.csv")
    out_parquet = Path(args.out_parquet) if args.out_parquet else (in_dir / "ticker_features_daily.parquet")

    # Hardcoded finance-only filter
    subreddit_filter = FINANCE_SUBREDDIT_ALLOWLIST if ENABLE_FINANCE_SUBREDDIT_FILTER else None


    if args.reset_db and db_path.exists():
        db_path.unlink()

    files = sorted(in_dir.rglob(args.glob))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    print(f"Discovered {len(files)} parquet files under {in_dir}")

    # Build allowlist
    allowlist: set[str] = set()
    if args.use_allowlist:
        cache_path = Path(args.allowlist_cache) if args.allowlist_cache else (in_dir / f"yahoo_most_actives_{args.allowlist_limit}.txt")
        allowlist = set(
            fetch_yahoo_most_actives(
                limit=args.allowlist_limit,
                cache_path=cache_path,
                force_refresh=args.force_refresh_allowlist,
            )
        )
        if args.allowlist_file:
            pth = Path(args.allowlist_file)
            if pth.exists():
                extra = {line.strip().upper() for line in pth.read_text(encoding="utf8").splitlines() if line.strip()}
                allowlist |= extra

        if allowlist:
            print(f"Allowlist enabled: {len(allowlist)} tickers")
        else:
            print("Allowlist enabled but empty (network/cache failed). Falling back to no allowlist filter.")

    conn = sqlite3.connect(str(db_path))
    init_db(conn, track_unique_authors=args.track_unique_authors, track_unique_threads=args.track_unique_threads)

    # Process files
    processed = 0
    for i, fp in enumerate(files, start=1):
        if not args.reset_db and file_already_processed(conn, fp):
            continue

        print(f"[{i}/{len(files)}] Processing: {fp}")
        try:
            process_parquet_file(
                parquet_path=fp,
                conn=conn,
                allowlist=allowlist,
                use_allowlist=args.use_allowlist,
                track_unique_authors=args.track_unique_authors,
                track_unique_threads=args.track_unique_threads,
                subreddit_filter=subreddit_filter,
            )
            mark_file_processed(conn, fp)
            processed += 1
        except Exception as exc:
            print("Failed:", fp, "=>", repr(exc))

    print(f"Processed {processed} new parquet files (skipped already-processed files).")

    # Export final features
    export_features(
        conn=conn,
        out_csv=out_csv,
        out_parquet=out_parquet,
        rolling_window=args.rolling_window,
        min_periods=args.min_periods,
        plot_top10=args.plot_top10,
        plot_path=Path(args.plot_path) if args.plot_path else None,
    )
    conn.close()


if __name__ == "__main__":
    main()
