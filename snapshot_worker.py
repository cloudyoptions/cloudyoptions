#!/usr/bin/env python3
"""
Cloud Options Snapshot & Volatility Dashboard Builder
"""

import html as html_lib
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    sys.exit("Missing dependency. Run: pip install -r requirements.txt")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("Missing dependency. Run: pip install -r requirements.txt")


SPREAD_SAMPLE_INTERVAL_SECONDS = 60
RETRY_DELAY_SECONDS = 5

MIN_OI_FOR_RATIO = 25
STALE_QUOTE_DAYS = 3
ARBITRAGE_TOLERANCE = 0.01
SNAPSHOT_NO_BID_WARN_PCT = 40.0
SNAPSHOT_OI_ZERO_WARN_PCT = 40.0
SNAPSHOT_IV_UNRELIABLE_WARN_PCT = 40.0
RISK_FREE_RATE = 0.045


def get_scan_config():
    """Reads up to 50 watchlist tickers and optional MAX_PREMIUM from environment variables."""
    raw_tickers = os.getenv(
        "TICKERS",
        "SPY, QQQ, XLE, GOOGL, AAPL, MSFT, NVDA, AMZN, META, TSLA"
    ).strip()
    if not raw_tickers:
        sys.exit("No ticker(s) entered in TICKERS.")

    tickers, seen = [], set()
    for t in raw_tickers.replace("\n", ",").split(","):
        t = t.strip().upper()
        if t and t not in seen:
            tickers.append(t)
            seen.add(t)
    if not tickers:
        sys.exit("No valid ticker(s) entered in TICKERS.")
    if len(tickers) > 50:
        dropped, tickers = tickers[50:], tickers[:50]
        print(f"Only the first 50 tickers will be scanned: {', '.join(tickers)}")
        print(f"(dropped: {', '.join(dropped)})")

    max_premium = None
    raw_max = os.getenv("MAX_PREMIUM", "").strip()
    if raw_max:
        try:
            max_premium = float(raw_max)
        except ValueError:
            sys.exit(f"'{raw_max}' isn't a valid number.")

    return SimpleNamespace(
        tickers=tickers,
        max_premium=max_premium,
        min_premium=0.06,
        min_dte=None,
        max_dte=None,
        max_expirations=None,
        min_strike=None,
        max_strike=None,
        max_spread_pct=None,
        min_volume=0,
        calls_only=False,
        puts_only=False,
        sort_by="volume",
        top=20,
        delay=0.35,
        cache_minutes=5.0,
        no_cache=True,
        out_dir=os.getenv("OUT_DIR", "docs"),
    )


def get_current_price(tk):
    try:
        fast = tk.fast_info
        price = fast.get("lastPrice") or fast.get("last_price")
        if price is not None and math.isfinite(float(price)) and float(price) > 0:
            return float(price)
    except Exception:
        pass
    try:
        hist = tk.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            close_series = hist["Close"].dropna()
            if not close_series.empty:
                return float(close_series.iloc[-1])
    except Exception:
        pass
    return math.nan


def compute_map_bias_flag(df):
    result = {
        "bias": "Neutral",
        "score": 0.0,
        "summary": "Insufficient liquidity or IV data to establish directional bias.",
        "call_vol_share": None,
        "call_oi_share": None,
        "vol_tilt": 0.0,
        "oi_tilt": 0.0,
        "skew_diff_pct": None,
        "iv_slope_pct": None,
        "total_weighted_vol": 0.0,
    }
    if df is None or df.empty:
        return result

    score = 0.0
    notes = []

    liq_df = df.copy()
    spread_col = "avgSpreadPct" if "avgSpreadPct" in liq_df.columns else "spreadPct"
    spread_vals = pd.to_numeric(liq_df.get(spread_col), errors="coerce").fillna(100.0).clip(lower=0.0, upper=200.0)
    liq_weight = 1.0 / (1.0 + spread_vals / 100.0)

    vol_series = pd.to_numeric(liq_df.get("volume"), errors="coerce").fillna(0.0) * liq_weight
    oi_series = pd.to_numeric(liq_df.get("openInterest"), errors="coerce").fillna(0.0) * liq_weight

    call_mask = liq_df["type"] == "Call"
    put_mask = liq_df["type"] == "Put"

    call_vol = float(vol_series[call_mask].sum())
    put_vol = float(vol_series[put_mask].sum())
    tot_vol = call_vol + put_vol
    result["total_weighted_vol"] = round(tot_vol, 1)

    call_oi = float(oi_series[call_mask].sum())
    put_oi = float(oi_series[put_mask].sum())
    tot_oi = call_oi + put_oi

    if tot_vol > 0:
        call_vol_share = call_vol / tot_vol
        vol_tilt = (call_vol_share - 0.5) * 2.0
        result["call_vol_share"] = round(call_vol_share * 100.0, 2)
        result["vol_tilt"] = round(vol_tilt, 3)
        score += vol_tilt * 1.2
        if abs(vol_tilt) >= 0.12:
            side = "Call" if vol_tilt > 0 else "Put"
            notes.append(f"Liquidity vol leans {side} ({call_vol_share * 100:.0f}% calls)")

    if tot_oi > 0:
        call_oi_share = call_oi / tot_oi
        oi_tilt = (call_oi_share - 0.5) * 2.0
        result["call_oi_share"] = round(call_oi_share * 100.0, 2)
        result["oi_tilt"] = round(oi_tilt, 3)
        score += oi_tilt * 1.0
        if abs(oi_tilt) >= 0.12:
            side = "Call" if oi_tilt > 0 else "Put"
            notes.append(f"OI leans {side} ({call_oi_share * 100:.0f}% calls)")

    iv_reliable_mask = df["ivReliable"] if "ivReliable" in df.columns else pd.Series(True, index=df.index)
    iv_series = pd.to_numeric(df.get("impliedVolatility"), errors="coerce")
    iv_df = df[iv_reliable_mask.fillna(True) & iv_series.notna() & (iv_series > 0)].copy()
    iv_df["impliedVolatility"] = iv_series.loc[iv_df.index]

    if not iv_df.empty and "daysToExpiration" in iv_df.columns:
        exp_iv = (
            iv_df.groupby(["expiration", "daysToExpiration", "type"], as_index=False)["impliedVolatility"]
            .median()
        )
        if not exp_iv.empty:
            piv = exp_iv.pivot_table(index=["expiration", "daysToExpiration"], columns="type", values="impliedVolatility")
            if "Call" in piv.columns and "Put" in piv.columns:
                paired = piv.dropna(subset=["Call", "Put"])
                if not paired.empty:
                    skew_diff = float((paired["Put"] - paired["Call"]).mean())
                    result["skew_diff_pct"] = round(skew_diff * 100.0, 2)
                    skew_tilt = max(-1.0, min(1.0, -skew_diff / 0.10))
                    score += skew_tilt * 0.9
                    if abs(skew_diff) >= 0.02:
                        if skew_diff > 0:
                            notes.append(f"Put IV > Call IV by {skew_diff * 100:.1f}% (downside skew)")
                        else:
                            notes.append(f"Call IV > Put IV by {abs(skew_diff) * 100:.1f}% (upside skew)")

            by_exp = exp_iv.groupby(["expiration", "daysToExpiration"], as_index=False)["impliedVolatility"].median()
            by_exp = by_exp.sort_values("daysToExpiration")
            if len(by_exp) >= 2:
                front_iv = float(by_exp.iloc[0]["impliedVolatility"])
                back_iv = float(by_exp.iloc[-1]["impliedVolatility"])
                if back_iv > 0:
                    slope = (front_iv - back_iv) / back_iv
                    result["iv_slope_pct"] = round(slope * 100.0, 2)
                    if abs(slope) >= 0.08:
                        if notes and "downside skew" in " ".join(notes) and slope > 0:
                            score -= 0.4
                            notes.append("Front-loaded IV curve")
                        elif notes and "upside skew" in " ".join(notes) and slope > 0:
                            score += 0.4
                            notes.append("Front-loaded IV curve")
                        else:
                            notes.append("Inverted IV curve" if slope > 0 else "Contango IV curve")

    if score >= 0.35:
        bias = "Bullish"
    elif score <= -0.35:
        bias = "Bearish"
    else:
        bias = "Neutral"

    summary = "; ".join(notes) if notes else "Balanced call/put liquidity and flat IV structure."
    result["bias"] = bias
    result["score"] = round(score, 2)
    result["summary"] = summary
    return result


def get_volatility_insights(tk, full_df, underlying_price):
    map_bias = compute_map_bias_flag(full_df)
    insights = {
        "current_30d_hv": None,
        "hv_52w_high": None,
        "hv_52w_low": None,
        "hv_rank": None,
        "atm_iv": None,
        "iv_hv_ratio": None,
        "pricing_bias": "N/A",
        "map_bias": map_bias["bias"],
        "map_bias_score": map_bias["score"],
        "map_bias_summary": map_bias["summary"],
        "call_vol_share": map_bias["call_vol_share"],
        "call_oi_share": map_bias["call_oi_share"],
        "vol_tilt": map_bias["vol_tilt"],
        "oi_tilt": map_bias["oi_tilt"],
        "skew_diff_pct": map_bias["skew_diff_pct"],
        "iv_slope_pct": map_bias["iv_slope_pct"],
        "total_weighted_vol": map_bias["total_weighted_vol"],
    }

    try:
        hist = tk.history(period="1y")
        if hist is not None and not hist.empty and len(hist) >= 30 and "Close" in hist.columns:
            close_vals = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            if len(close_vals) >= 30:
                log_returns = np.log(close_vals / close_vals.shift(1))
                rolling_30d_hv = log_returns.rolling(window=30).std() * np.sqrt(252) * 100.0
                hv_series = rolling_30d_hv.dropna()

                if not hv_series.empty:
                    current_hv = float(hv_series.iloc[-1])
                    hv_high = float(hv_series.max())
                    hv_low = float(hv_series.min())

                    hv_rank = 0.0
                    if hv_high > hv_low:
                        hv_rank = ((current_hv - hv_low) / (hv_high - hv_low)) * 100.0

                    insights["current_30d_hv"] = round(current_hv, 2)
                    insights["hv_52w_high"] = round(hv_high, 2)
                    insights["hv_52w_low"] = round(hv_low, 2)
                    insights["hv_rank"] = round(hv_rank, 1)
    except Exception:
        pass

    try:
        if full_df is not None and not full_df.empty and underlying_price and not math.isnan(underlying_price):
            iv_series = pd.to_numeric(full_df.get("impliedVolatility"), errors="coerce")
            strike_series = pd.to_numeric(full_df.get("strike"), errors="coerce")
            reliable_mask = full_df["ivReliable"] if "ivReliable" in full_df.columns else pd.Series(True, index=full_df.index)

            valid_ivs = full_df[reliable_mask.fillna(True) & iv_series.notna() & (iv_series > 0) & strike_series.notna()].copy()
            if not valid_ivs.empty:
                valid_ivs["impliedVolatility"] = iv_series.loc[valid_ivs.index]
                valid_ivs["strike"] = strike_series.loc[valid_ivs.index]
                valid_ivs["strike_diff"] = (valid_ivs["strike"] - underlying_price).abs()
                atm_contracts = valid_ivs.nsmallest(6, "strike_diff")
                mean_atm_iv = float(atm_contracts["impliedVolatility"].mean()) * 100.0
                insights["atm_iv"] = round(mean_atm_iv, 2)

                if insights["current_30d_hv"] and insights["current_30d_hv"] > 0:
                    ratio = mean_atm_iv / insights["current_30d_hv"]
                    insights["iv_hv_ratio"] = round(ratio, 2)
                    if ratio > 1.10:
                        insights["pricing_bias"] = "Overpriced (Sell Bias)"
                    elif ratio < 0.90:
                        insights["pricing_bias"] = "Underpriced (Buy Bias)"
                    else:
                        insights["pricing_bias"] = "Fairly Valued"
    except Exception:
        pass

    return insights


def fetch_one_expiration_raw(ticker_obj, expiration):
    for attempt in range(3):
        try:
            chain = ticker_obj.option_chain(expiration)
            if chain is not None and hasattr(chain, "calls") and hasattr(chain, "puts"):
                return chain
        except YFRateLimitError:
            wait_s = 15 * (attempt + 1)
            print(f"    Rate-limited by Yahoo on {expiration}; waiting {wait_s}s before retry...")
            time.sleep(wait_s)
        except Exception as exc:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"    Failed fetching expiration {expiration}: {exc}")
    return None


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(option_type, S, K, T_years, sigma, r=RISK_FREE_RATE):
    nan_result = {"delta": math.nan, "gamma": math.nan, "theta": math.nan, "vega": math.nan}
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (S, K, T_years, sigma)):
        return nan_result
    if S <= 0 or K <= 0 or T_years <= 0 or sigma <= 0:
        return nan_result
    try:
        sqrt_T = math.sqrt(T_years)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)
        vega = S * _norm_pdf(d1) * sqrt_T / 100.0
        if option_type == "Call":
            delta = _norm_cdf(d1)
            theta_annual = (
                -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
                - r * K * math.exp(-r * T_years) * _norm_cdf(d2)
            )
        else:
            delta = _norm_cdf(d1) - 1.0
            theta_annual = (
                -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
                + r * K * math.exp(-r * T_years) * _norm_cdf(-d2)
            )
        return {"delta": delta, "gamma": gamma, "theta": theta_annual / 365.0, "vega": vega}
    except (ValueError, ZeroDivisionError):
        return nan_result


def make_contract_key(row):
    symbol = row.get("contractSymbol")
    if symbol is not None and not pd.isna(symbol) and str(symbol).strip():
        return str(symbol).strip()
    return f"{row.get('type', '')}|{row.get('expiration', '')}|{row.get('strike', '')}"


def add_computed_metrics(df, option_type, expiration, underlying_price, fetch_time):
    df = df.copy() if df is not None else pd.DataFrame()
    df["type"] = option_type
    df["expiration"] = expiration

    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    days_to_expiration = (exp_date - date.today()).days
    df["daysToExpiration"] = days_to_expiration

    for col in ["bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "strike"]:
        if col not in df.columns:
            df[col] = float("nan")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    df["openInterest"] = df["openInterest"].fillna(0)

    bid = df["bid"].fillna(0)
    ask = df["ask"].fillna(0)
    mid = (bid + ask) / 2.0
    df["midPrice"] = mid
    spread = ask - bid
    df["spreadDollar"] = spread

    has_two_sided_market = (bid > 0) & (ask > 0)
    crossed_market = (bid > 0) & (ask > 0) & (ask < bid)
    valid_spread = has_two_sided_market & ~crossed_market
    df["spreadPct"] = spread.where(valid_spread) / mid.where(valid_spread) * 100.0

    volume = df["volume"].fillna(0)
    open_interest = df["openInterest"].fillna(0)
    reliable_oi = open_interest >= MIN_OI_FOR_RATIO
    df["volumeOiRatio"] = volume.where(reliable_oi) / open_interest.where(reliable_oi)

    if underlying_price and not math.isnan(underlying_price):
        df["moneynessPct"] = (df["strike"] - underlying_price) / underlying_price * 100.0
        df["inTheMoney"] = (df["strike"] < underlying_price) if option_type == "Call" else (df["strike"] > underlying_price)
    else:
        df["moneynessPct"] = math.nan
        df["inTheMoney"] = None

    df["noBid"] = bid <= 0
    df["crossedMarket"] = crossed_market
    df["lowOiConfidence"] = ~reliable_oi

    if underlying_price and not math.isnan(underlying_price):
        if option_type == "Call":
            intrinsic = (underlying_price - df["strike"]).clip(lower=0)
        else:
            intrinsic = (df["strike"] - underlying_price).clip(lower=0)
        df["arbitrageViolation"] = (bid > 0) & (bid < intrinsic - ARBITRAGE_TOLERANCE)
    else:
        df["arbitrageViolation"] = False

    if "lastTradeDate" in df.columns:
        last_trade = pd.to_datetime(df["lastTradeDate"], errors="coerce", utc=True)
        fetch_ts = pd.Timestamp(fetch_time)
        fetch_ts = fetch_ts.tz_localize("UTC") if fetch_ts.tzinfo is None else fetch_ts.tz_convert("UTC")
        quote_age = (fetch_ts - last_trade).dt.total_seconds() / 86400.0
        df["quoteAgeDays"] = quote_age.round(2)
        df["staleQuote"] = quote_age > STALE_QUOTE_DAYS
    else:
        df["quoteAgeDays"] = math.nan
        df["staleQuote"] = None

    df["ivReliable"] = has_two_sided_market & ~df["arbitrageViolation"] & ~df["crossedMarket"]

    T_years = days_to_expiration / 365.0
    deltas, gammas, thetas, vegas = [], [], [], []
    for strike, sigma in zip(df["strike"], df["impliedVolatility"]):
        g = black_scholes_greeks(option_type, underlying_price, strike, T_years, sigma)
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        thetas.append(g["theta"])
        vegas.append(g["vega"])
    df["delta"], df["gamma"], df["theta"], df["vega"] = deltas, gammas, thetas, vegas
    unreliable = ~df["ivReliable"]
    df.loc[unreliable, ["delta", "gamma", "theta", "vega"]] = math.nan

    df["contractKey"] = [make_contract_key(r) for _, r in df.iterrows()]
    return df


def attach_average_spread(df, state, ticker):
    df = df.copy()
    if state is None:
        df["avgSpreadPct"] = df["spreadPct"]
        df["spreadSampleCount"] = df["spreadPct"].notna().astype(int)
        return df

    avg_vals = []
    sample_counts = []
    for _, row in df.iterrows():
        key = row.get("contractKey") or make_contract_key(row)
        val = row.get("spreadPct")
        samples = state.record_spread_sample(ticker, key, val)
        if samples:
            avg_vals.append(float(np.mean(samples)))
            sample_counts.append(len(samples))
        else:
            avg_vals.append(math.nan)
            sample_counts.append(0)

    df["avgSpreadPct"] = avg_vals
    df["spreadSampleCount"] = sample_counts
    return df


def apply_filters(df, args):
    out = df
    if args.calls_only:
        out = out[out["type"] == "Call"]
    if args.puts_only:
        out = out[out["type"] == "Put"]
    if args.min_strike is not None:
        out = out[out["strike"] >= args.min_strike]
    if args.max_strike is not None:
        out = out[out["strike"] <= args.max_strike]
    if args.min_premium is not None:
        out = out[out["lastPrice"] >= args.min_premium]
    if args.max_premium is not None:
        out = out[out["lastPrice"] <= args.max_premium]
    if getattr(args, "min_dte", None) is not None:
        out = out[out["daysToExpiration"] >= args.min_dte]
    if getattr(args, "max_dte", None) is not None:
        out = out[out["daysToExpiration"] <= args.max_dte]
    spread_col = "avgSpreadPct" if "avgSpreadPct" in out.columns else "spreadPct"
    if args.max_spread_pct is not None:
        out = out[out[spread_col].notna() & (out[spread_col] <= args.max_spread_pct + 1e-9) & ~out["crossedMarket"]]
    if args.min_volume:
        out = out[out["volume"].fillna(0) >= args.min_volume]
    return out


def assess_snapshot_quality(full_df):
    warnings = []
    if full_df.empty:
        return warnings
    n = len(full_df)
    pct_no_bid = 100.0 * full_df["noBid"].sum() / n
    pct_oi_zero = 100.0 * (full_df["openInterest"].fillna(0) == 0).sum() / n
    pct_iv_unreliable = 100.0 * (~full_df["ivReliable"]).sum() / n
    n_arbitrage = int(full_df["arbitrageViolation"].sum())

    if pct_no_bid >= SNAPSHOT_NO_BID_WARN_PCT:
        warnings.append(
            f"{pct_no_bid:.0f}% of contracts have NO BID at all -- far above normal "
            f"illiquidity. Treat lastPrice/spread/volume conclusions with heavy skepticism."
        )
    if pct_oi_zero >= SNAPSHOT_OI_ZERO_WARN_PCT:
        warnings.append(
            f"{pct_oi_zero:.0f}% of contracts show ZERO open interest -- suggests "
            f"the field didn't populate correctly this fetch."
        )
    if pct_iv_unreliable >= SNAPSHOT_IV_UNRELIABLE_WARN_PCT:
        warnings.append(
            f"{pct_iv_unreliable:.0f}% of contracts have no reliable implied volatility."
        )
    if n_arbitrage > 0:
        warnings.append(
            f"{n_arbitrage} contract(s) bid below intrinsic value -- quotes are stale/broken."
        )
    return warnings


DASHBOARD_CSS = """
    body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 24px 32px 60px;
           background: #f7f8fa; color: #1a1a1a; }
    .watchlist-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
                     background: white; border: 1px solid #ccd2da; border-radius: 8px;
                     padding: 10px 14px; margin-bottom: 14px; }
    .watchlist-bar label { font-size: 0.78em; font-weight: 700; color: #374151;
                          text-transform: uppercase; letter-spacing: 0.03em; }
    .watchlist-bar input[list="watchlist-tickers"] { padding: 6px 10px; border: 1px solid #ccd2da;
                          border-radius: 6px; font-size: 0.9em; width: 170px; text-transform: uppercase; }
    .watchlist-bar select { padding: 6px 10px; border: 1px solid #ccd2da; border-radius: 6px;
                          font-size: 0.9em; background: white; }
    .watchlist-go-btn { padding: 6px 14px; border: 1px solid #2563eb; border-radius: 6px;
                        background: #2563eb; color: white; font-size: 0.88em; font-weight: 600;
                        cursor: pointer; }
    .watchlist-go-btn:hover { background: #1d4ed8; }
    h1 { font-size: 1.6em; margin-bottom: 4px; }
    .header-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    h2 { font-size: 1.15em; margin-top: 28px; border-bottom: 2px solid #e2e4e8; padding-bottom: 6px; }
    .meta { color: #555; font-size: 0.92em; line-height: 1.6; }
    .warning-banner { background: #fff4e5; border: 1px solid #f0b429; border-radius: 8px; padding: 14px 18px; margin: 16px 0; }
    .warning-banner h2 { border: none; margin: 0 0 8px 0; color: #92400e; }
    .warning-banner ul { margin: 4px 0 0 0; padding-left: 20px; }
    .stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 10px 0 4px; }
    .stat-box { background: white; border: 1px solid #e2e4e8; border-radius: 8px; padding: 10px 16px; min-width: 140px; }
    .stat-box.highlight { border-color: #2563eb; background: #f0f5ff; }
    .stat-box.bias-bullish { border-color: #16a34a; background: #f0fdf4; }
    .stat-box.bias-bullish .value, .stat-box.bias-bullish .sub { color: #15803d; }
    .stat-box.bias-bearish { border-color: #dc2626; background: #fef2f2; }
    .stat-box.bias-bearish .value, .stat-box.bias-bearish .sub { color: #b91c1c; }
    .stat-box.bias-neutral { border-color: #64748b; background: #f8fafc; }
    .stat-box.bias-neutral .value, .stat-box.bias-neutral .sub { color: #475569; }
    .stat-box .label { font-size: 0.72em; color: #666; text-transform: uppercase; letter-spacing: 0.03em; font-weight: 600; }
    .stat-box .value { font-size: 1.25em; font-weight: 700; color: #111827; margin-top: 2px; }
    .stat-box .sub { font-size: 0.75em; color: #2563eb; font-weight: 600; margin-top: 2px; }
    .bias-note { font-size: 0.88em; color: #374151; background: white; border: 1px solid #e2e4e8;
                 border-radius: 8px; padding: 10px 14px; margin-top: 10px; line-height: 1.5; }
    .story-banner { font-size: 0.88em; color: #1e293b; background: #f8fafc; border-left: 4px solid #2563eb;
                    border-top: 1px solid #e2e4e8; border-right: 1px solid #e2e4e8; border-bottom: 1px solid #e2e4e8;
                    border-radius: 6px; padding: 10px 14px; margin: 10px 0 8px; line-height: 1.5; }
    .chart-box { background: white; border: 1px solid #e2e4e8; border-radius: 8px; padding: 8px; margin: 14px 0; }
    table.data-table { border-collapse: collapse; width: 100%; background: white; font-size: 0.85em; }
    table.data-table th, table.data-table td { border: 1px solid #e2e4e8; padding: 6px 10px; text-align: right; white-space: nowrap; }
    table.data-table th:first-child, table.data-table td:first-child { text-align: left; }
    table.data-table th { background: #eef1f5; cursor: pointer; user-select: none; position: sticky; top: 0; }
    table.data-table th:hover { background: #dfe4ea; }
    table.data-table tbody tr:nth-child(even) { background: #fafbfc; }
    table.data-table tbody tr.flagged-row { background: #fdecea; color: #9a3412; }
    table.data-table tbody tr:hover { background: #dbeafe !important; }
    .note { color: #666; font-size: 0.88em; font-style: italic; margin: 4px 0 10px; }
    .table-wrap { max-height: 600px; overflow: auto; border: 1px solid #e2e4e8; border-radius: 8px; }
    .filter-bar { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-end; background: white;
                  border: 1px solid #e2e4e8; border-radius: 8px; padding: 12px 16px; margin: 10px 0 14px; }
    .filter-field { display: flex; flex-direction: column; gap: 4px; }
    .filter-field label { font-size: 0.72em; color: #777; text-transform: uppercase; letter-spacing: 0.03em; }
    .filter-field select, .filter-field input[type="number"] { padding: 5px 8px; border: 1px solid #ccd2da; border-radius: 6px;
                  font-size: 0.9em; }
    .filter-field input[type="number"] { width: 75px; }
    .slider-group { display: flex; align-items: center; gap: 8px; }
    .filter-field input[type="range"] { accent-color: #2563eb; cursor: pointer; width: 110px; }
    .filter-reset { padding: 6px 14px; border: 1px solid #ccd2da; border-radius: 6px; background: #f1f3f6;
                    cursor: pointer; font-size: 0.85em; height: 30px; }
    .filter-reset:hover { background: #e2e6eb; }
    .filter-count { font-size: 0.85em; color: #555; margin-left: auto; align-self: center; }
"""

SORT_JS = """
function goToWatchlistTicker(rawVal) {
    const clean = (rawVal || '').trim().toUpperCase();
    if (!clean) return;
    window.location.href = clean + '.html';
}

function handleWatchlistSubmit(event) {
    event.preventDefault();
    const inputEl = document.getElementById('watchlist-input');
    const selectEl = document.getElementById('watchlist-select');
    const chosen = (inputEl && inputEl.value.trim()) ? inputEl.value.trim() : (selectEl ? selectEl.value : '');
    goToWatchlistTicker(chosen);
    return false;
}

function sortTable(tableId, colIdx) {
    const table = document.getElementById(tableId);
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const asc = table.dataset.sortCol == colIdx ? table.dataset.sortDir !== "asc" : true;
    rows.sort((a, b) => {
        const xText = a.cells[colIdx].innerText.trim();
        const yText = b.cells[colIdx].innerText.trim();
        const xNum = parseFloat(xText.replace(/[^0-9.\-]/g, ""));
        const yNum = parseFloat(yText.replace(/[^0-9.\-]/g, ""));
        const bothNumeric = !isNaN(xNum) && !isNaN(yNum) && xText.match(/^-?[\d.,]+%?$/) && yText.match(/^-?[\d.,]+%?$/);
        if (bothNumeric) {
            return asc ? xNum - yNum : yNum - xNum;
        }
        return asc ? xText.localeCompare(yText) : yText.localeCompare(xText);
    });
    table.dataset.sortCol = colIdx;
    table.dataset.sortDir = asc ? "asc" : "desc";
    rows.forEach(r => tbody.appendChild(r));
}

function getTickerStorageKey() {
    const parts = window.location.pathname.split('/');
    const last = parts[parts.length - 1] || 'DEFAULT.html';
    const ticker = last.replace(/\.html$/i, '').toUpperCase();
    return 'options_filter_state_' + ticker;
}

function saveFilterState() {
    const state = {
        type: document.getElementById('filter-type').value,
        dte: document.getElementById('filter-dte-num').value,
        maxDte: document.getElementById('filter-max-dte-num').value,
        price: document.getElementById('filter-price-num').value,
        vol: document.getElementById('filter-vol-num').value,
        spread: document.getElementById('filter-spread-num').value,
        oi: document.getElementById('filter-oi-num').value
    };
    localStorage.setItem(getTickerStorageKey(), JSON.stringify(state));
}

function loadFilterState() {
    const saved = localStorage.getItem(getTickerStorageKey());
    if (!saved) return;
    try {
        const state = JSON.parse(saved);
        if (state.type !== undefined) document.getElementById('filter-type').value = state.type;

        const pairs = [
            ['filter-dte-slider', 'filter-dte-num', state.dte],
            ['filter-max-dte-slider', 'filter-max-dte-num', state.maxDte],
            ['filter-price-slider', 'filter-price-num', state.price],
            ['filter-vol-slider', 'filter-vol-num', state.vol],
            ['filter-spread-slider', 'filter-spread-num', state.spread],
            ['filter-oi-slider', 'filter-oi-num', state.oi]
        ];

        pairs.forEach(([sId, nId, val]) => {
            if (val !== undefined && val !== "") {
                const slider = document.getElementById(sId);
                const num = document.getElementById(nId);
                if (slider) slider.value = val;
                if (num) num.value = val;
            }
        });
    } catch (e) {
        console.error('Failed to restore filter state', e);
    }
}

function syncInputs(sliderId, numId) {
    const slider = document.getElementById(sliderId);
    const num = document.getElementById(numId);
    if (!slider || !num) return;
    slider.addEventListener('input', () => {
        num.value = slider.value;
        saveFilterState();
        filterTable();
    });
    num.addEventListener('input', () => {
        slider.value = num.value;
        saveFilterState();
        filterTable();
    });
}

function filterTable() {
    const typeVal = document.getElementById('filter-type').value;
    const minDteRaw = parseFloat(document.getElementById('filter-dte-num').value);
    const minDte = isNaN(minDteRaw) ? 0 : minDteRaw;
    const maxDteRaw = parseFloat(document.getElementById('filter-max-dte-num').value);
    const maxDte = isNaN(maxDteRaw) ? Infinity : maxDteRaw;
    const maxPriceRaw = parseFloat(document.getElementById('filter-price-num').value);
    const maxPrice = isNaN(maxPriceRaw) ? Infinity : maxPriceRaw;
    const minVolRaw = parseFloat(document.getElementById('filter-vol-num').value);
    const minVol = isNaN(minVolRaw) ? 0 : minVolRaw;
    const maxSpreadRaw = parseFloat(document.getElementById('filter-spread-num').value);
    const maxSpread = isNaN(maxSpreadRaw) ? Infinity : maxSpreadRaw;
    const minOiRaw = parseFloat(document.getElementById('filter-oi-num').value);
    const minOi = isNaN(minOiRaw) ? 0 : minOiRaw;

    const tbody = document.querySelector('#scan-results-table tbody');
    if (!tbody) return;
    const rows = tbody.rows;
    let visibleCount = 0;

    for (let r of rows) {
        const type = r.cells[0].innerText.trim();
        const dte = parseFloat(r.cells[2].innerText) || 0;
        const last = parseFloat(r.cells[5].innerText) || 0;
        const avgSpreadText = r.cells[9].innerText.trim();
        const avgSpread = avgSpreadText === '-' ? Infinity : parseFloat(avgSpreadText);
        const vol = parseFloat(r.cells[10].innerText) || 0;
        const oi = parseFloat(r.cells[11].innerText) || 0;

        let match = true;
        if (typeVal !== 'ALL' && type !== typeVal) match = false;
        if (dte < minDte) match = false;
        if (dte > maxDte) match = false;
        if (last > maxPrice) match = false;
        if (vol < minVol) match = false;
        if (avgSpread > maxSpread) match = false;
        if (oi < minOi) match = false;

        r.style.display = match ? '' : 'none';
        if (match) visibleCount++;
    }
    const countEl = document.getElementById('filter-count');
    if (countEl) countEl.innerText = visibleCount + ' matching contracts';
}

function resetFilters() {
    localStorage.removeItem(getTickerStorageKey());

    document.getElementById('filter-type').value = 'ALL';
    const resets = [
        ['filter-dte-slider', 'filter-dte-num', 0],
        ['filter-max-dte-slider', 'filter-max-dte-num', 365],
        ['filter-price-slider', 'filter-price-num', 999],
        ['filter-vol-slider', 'filter-vol-num', 0],
        ['filter-spread-slider', 'filter-spread-num', 100],
        ['filter-oi-slider', 'filter-oi-num', 0]
    ];
    resets.forEach(([s, n, v]) => {
        const slider = document.getElementById(s);
        const num = document.getElementById(n);
        if (slider) slider.value = v;
        if (num) num.value = v;
    });
    filterTable();
}

document.addEventListener('DOMContentLoaded', () => {
    syncInputs('filter-dte-slider', 'filter-dte-num');
    syncInputs('filter-max-dte-slider', 'filter-max-dte-num');
    syncInputs('filter-price-slider', 'filter-price-num');
    syncInputs('filter-vol-slider', 'filter-vol-num');
    syncInputs('filter-spread-slider', 'filter-spread-num');
    syncInputs('filter-oi-slider', 'filter-oi-num');

    loadFilterState();
    filterTable();
});
"""


def summarize_order_flow_narrative(history):
    if not history:
        return "Awaiting minutely snapshots to build the order-flow narrative."

    first = history[0]
    latest = history[-1]
    n = len(history)

    if n == 1:
        vol_txt = (
            f"Call volume share is {latest['call_vol_share']:.1f}%"
            if latest.get("call_vol_share") is not None
            else "Call/put volume share is balanced"
        )
        skew_txt = (
            f"Put-Call IV skew stands at {latest['skew_diff_pct']:+.2f}%"
            if latest.get("skew_diff_pct") is not None
            else "IV skew is flat"
        )
        return (
            f"Initial snapshot (#{latest['round']} at {latest['timestamp']}): "
            f"Bias is <strong>{latest['bias']}</strong> (score {latest['score']:+.2f}; yellow neutral zone is -0.35 to +0.35). "
            f"{vol_txt}; {skew_txt}. Additional scheduled ticks will chart directional momentum."
        )

    score_delta = latest["score"] - first["score"]
    if score_delta >= 0.15:
        trend_phrase = f"strengthened bullishly ({first['score']:+.2f} &rarr; {latest['score']:+.2f})"
    elif score_delta <= -0.15:
        trend_phrase = f"shifted bearishly ({first['score']:+.2f} &rarr; {latest['score']:+.2f})"
    else:
        trend_phrase = f"held steady ({first['score']:+.2f} &rarr; {latest['score']:+.2f})"

    flips = []
    for i in range(1, n):
        if history[i]["bias"] != history[i - 1]["bias"]:
            flips.append(f"{history[i - 1]['bias']}&rarr;{history[i]['bias']} @ {history[i]['timestamp']}")

    flip_txt = f" Regime shifts: {', '.join(flips)}." if flips else f" Regime remained <strong>{latest['bias']}</strong> across all {n} samples."

    vol_shift_txt = ""
    if first.get("call_vol_share") is not None and latest.get("call_vol_share") is not None:
        dv = latest["call_vol_share"] - first["call_vol_share"]
        vol_shift_txt = f" Call volume flow moved {first['call_vol_share']:.1f}% &rarr; {latest['call_vol_share']:.1f}% ({dv:+.1f}%)."

    price_txt = ""
    if first.get("spot") is not None and latest.get("spot") is not None and first["spot"] > 0:
        dp = latest["spot"] - first["spot"]
        dp_pct = (dp / first["spot"]) * 100.0
        price_txt = f" Underlying moved ${first['spot']:.2f} &rarr; ${latest['spot']:.2f} ({dp_pct:+.2f}%)."

    return (
        f"Across <strong>{n}</strong> snapshots ({first['timestamp']} &ndash; {latest['timestamp']}), "
        f"composite order-flow bias {trend_phrase}.{flip_txt}{vol_shift_txt}{price_txt}"
    )


def _compute_sensitive_flow_deltas(values):
    base = next((float(v) for v in values if v is not None and math.isfinite(float(v))), None)
    if base is None:
        return [None for _ in values]
    deltas = []
    for v in values:
        if v is None or not math.isfinite(float(v)):
            deltas.append(None)
        else:
            deltas.append(round(float(v) - base, 2))
    return deltas


def _symmetric_zoom_range(series_list, min_span=6.0, pad_ratio=0.25):
    nums = [
        abs(float(val))
        for s in series_list
        for val in s
        if val is not None and math.isfinite(float(val))
    ]
    peak = max(nums) if nums else 0.0
    bound = max(min_span, peak * (1.0 + pad_ratio))
    return [-round(bound, 2), round(bound, 2)]


def build_order_flow_story_chart(ticker, history):
    if not history:
        return "<p>No market interpretation history recorded yet.</p>"

    times = [h["timestamp"] for h in history]
    rounds = [h["round"] for h in history]
    scores = [h["score"] for h in history]
    biases = [h["bias"] for h in history]
    summaries = [h["summary"] for h in history]
    spots = [h["spot"] for h in history]
    call_vols = [h["call_vol_share"] for h in history]
    call_ois = [h["call_oi_share"] for h in history]
    d_call_vols = _compute_sensitive_flow_deltas(call_vols)
    d_call_ois = _compute_sensitive_flow_deltas(call_ois)
    skews = [h["skew_diff_pct"] for h in history]
    slopes = [h["iv_slope_pct"] for h in history]
    atm_ivs = [h["atm_iv"] for h in history]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.29, 0.29],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=(
            f"{ticker} Composite Order-Flow Bias Score vs. Spot Price (Yellow Strip = -0.35 to +0.35)",
            "Liquidity-Weighted Order Flow Shift (Δ vs. First Sample — High Sensitivity)",
            "Volatility Structure (Put-Call IV Skew %, Term Slope % & ATM IV %)",
        ),
    )

    fig.add_hrect(y0=0.35, y1=2.4, fillcolor="rgba(22, 163, 74, 0.07)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=-0.35, y1=0.35, fillcolor="rgba(250, 204, 21, 0.28)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=-2.4, y1=-0.35, fillcolor="rgba(220, 38, 38, 0.07)", line_width=0, row=1, col=1)

    fig.add_hline(y=0.35, line_dash="dash", line_color="#eab308", line_width=1.2, row=1, col=1)
    fig.add_hline(y=-0.35, line_dash="dash", line_color="#eab308", line_width=1.2, row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="#94a3b8", line_width=1, row=1, col=1)

    marker_colors = [
        "#16a34a" if b == "Bullish" else ("#dc2626" if b == "Bearish" else "#ca8a04")
        for b in biases
    ]
    hover_story = [
        f"<b>Sample #{r} ({t})</b><br>"
        f"Regime: <b>{b}</b> (Score: {s:+.2f})<br>"
        f"Drivers: {summ}"
        for r, t, b, s, summ in zip(rounds, times, biases, scores, summaries)
    ]

    fig.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            name="Bias Score",
            mode="lines+markers",
            line=dict(color="#1d4ed8", width=2.5),
            marker=dict(size=9, color=marker_colors, line=dict(width=1.5, color="white")),
            text=hover_story,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    for i in range(1, len(history)):
        if biases[i] != biases[i - 1]:
            flip_color = "#15803d" if biases[i] == "Bullish" else ("#b91c1c" if biases[i] == "Bearish" else "#a16207")
            fig.add_annotation(
                x=times[i],
                y=scores[i],
                text=f"Flip: {biases[i]}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-26 if scores[i] >= 0 else 26,
                font=dict(size=11, color=flip_color),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=flip_color,
                borderwidth=1,
                row=1,
                col=1,
            )

    if any(sp is not None for sp in spots):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=spots,
                name="Spot Price ($)",
                mode="lines+markers",
                line=dict(color="#f59e0b", width=2, dash="dash"),
                marker=dict(size=6, color="#f59e0b"),
                hovertemplate="Spot: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    fig.add_hline(y=0.0, line_dash="dot", line_color="#64748b", line_width=1.2, row=2, col=1)
    if any(v is not None for v in d_call_vols):
        vol_hover = [
            f"Δ Call Vol: {dv:+.2f} pts (Share: {cv:.1f}%)" if (dv is not None and cv is not None) else "Δ Call Vol: N/A"
            for dv, cv in zip(d_call_vols, call_vols)
        ]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=d_call_vols,
                name="Δ Call Vol Share (pts)",
                mode="lines+markers",
                line=dict(color="#0d9488", width=2.4),
                marker=dict(size=8, color="#0d9488", line=dict(width=1.2, color="white")),
                text=vol_hover,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if any(o is not None for o in d_call_ois):
        oi_hover = [
            f"Δ Call OI: {do:+.2f} pts (Share: {co:.1f}%)" if (do is not None and co is not None) else "Δ Call OI: N/A"
            for do, co in zip(d_call_ois, call_ois)
        ]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=d_call_ois,
                name="Δ Call OI Share (pts)",
                mode="lines+markers",
                line=dict(color="#6366f1", width=2.2, dash="dash"),
                marker=dict(size=7, color="#6366f1", line=dict(width=1.2, color="white")),
                text=oi_hover,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    flow_y_range = _symmetric_zoom_range([d_call_vols, d_call_ois], min_span=5.0, pad_ratio=0.30)

    fig.add_hline(y=0.0, line_dash="dot", line_color="#94a3b8", line_width=1, row=3, col=1, secondary_y=False)
    if any(sk is not None for sk in skews):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=skews,
                name="Put-Call IV Skew %",
                mode="lines+markers",
                line=dict(color="#e11d48", width=2),
                marker=dict(size=6),
                hovertemplate="Put-Call Skew: %{y:+.2f}%<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    if any(sl is not None for sl in slopes):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=slopes,
                name="IV Term Slope %",
                mode="lines+markers",
                line=dict(color="#8b5cf6", width=1.8, dash="dash"),
                marker=dict(size=6),
                hovertemplate="IV Term Slope: %{y:+.2f}%<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    if any(iv is not None for iv in atm_ivs):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=atm_ivs,
                name="ATM IV %",
                mode="lines+markers",
                line=dict(color="#0284c7", width=2),
                marker=dict(size=6),
                hovertemplate="ATM IV: %{y:.2f}%<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

    fig.update_yaxes(title_text="Bias Score", range=[-2.4, 2.4], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Spot ($)", showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Δ Flow (pts)", range=flow_y_range, zeroline=False, row=2, col=1)
    fig.update_yaxes(title_text="Skew / Slope %", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ATM IV %", showgrid=False, row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Sample Timestamp", row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        height=760,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=75, b=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_interactive_html(
    ticker,
    underlying_price,
    full_df,
    filtered_df,
    scan_warnings,
    vol_insights,
    max_premium=None,
    sample_rounds=1,
    market_history=None,
    watchlist_tickers=None,
):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    market_history = market_history or []

    wl_list = list(watchlist_tickers) if watchlist_tickers else [ticker]
    if ticker not in wl_list:
        wl_list = [ticker] + wl_list
    datalist_opts = "\n".join(f'<option value="{html_lib.escape(t)}">' for t in wl_list)
    select_opts = "\n".join(
        f'<option value="{html_lib.escape(t)}"{" selected" if t == ticker else ""}>{html_lib.escape(t)}</option>'
        for t in wl_list
    )
    watchlist_bar_html = f"""
    <form class="watchlist-bar" onsubmit="return handleWatchlistSubmit(event);">
        <label for="watchlist-input">Watchlist Ticker ({len(wl_list)}):</label>
        <input id="watchlist-input" list="watchlist-tickers" placeholder="Type ticker (e.g. XLE)" autocomplete="off">
        <datalist id="watchlist-tickers">
            {datalist_opts}
        </datalist>
        <select id="watchlist-select" onchange="goToWatchlistTicker(this.value)">
            {select_opts}
        </select>
        <button type="submit" class="watchlist-go-btn">View Ticker</button>
    </form>
    """

    warn_html = ""
    if scan_warnings:
        items = "".join(f"<li>{html_lib.escape(str(w))}</li>" for w in scan_warnings)
        warn_html = f"""
        <div class="warning-banner">
            <h2>Data Quality Warnings</h2>
            <ul>{items}</ul>
        </div>
        """

    total_contracts = len(full_df)
    filtered_count = len(filtered_df)
    call_count = len(filtered_df[filtered_df["type"] == "Call"]) if not filtered_df.empty else 0
    put_count = len(filtered_df[filtered_df["type"] == "Put"]) if not filtered_df.empty else 0

    hv_str = f"{vol_insights['current_30d_hv']:.1f}%" if vol_insights.get("current_30d_hv") is not None else "-"
    iv_str = f"{vol_insights['atm_iv']:.1f}%" if vol_insights.get("atm_iv") is not None else "-"
    ratio_str = f"{vol_insights['iv_hv_ratio']:.2f}" if vol_insights.get("iv_hv_ratio") is not None else "-"
    rank_str = f"{vol_insights['hv_rank']:.1f}%" if vol_insights.get("hv_rank") is not None else "-"
    bias_str = html_lib.escape(str(vol_insights.get("pricing_bias", "N/A")))

    map_bias = vol_insights.get("map_bias", "Neutral")
    map_bias_score = vol_insights.get("map_bias_score", 0.0)
    map_bias_summary = html_lib.escape(str(vol_insights.get("map_bias_summary", "")))
    bias_css_cls = {
        "Bullish": "bias-bullish",
        "Bearish": "bias-bearish",
    }.get(map_bias, "bias-neutral")

    story_narrative_html = summarize_order_flow_narrative(market_history)
    chart_story_html = build_order_flow_story_chart(ticker, market_history)

    chart_liquidity_html = "<p>No data to display liquidity map.</p>"
    if not filtered_df.empty:
        plot_data = filtered_df.copy()
        spread_series = plot_data["avgSpreadPct"] if "avgSpreadPct" in plot_data.columns else plot_data["spreadPct"]
        plot_data["spreadPct_clean"] = pd.to_numeric(spread_series, errors="coerce").fillna(100.0)
        plot_data["openInterest_clean"] = pd.to_numeric(plot_data["openInterest"], errors="coerce").fillna(1.0).clip(lower=1.0)
        plot_data["volume_clean"] = pd.to_numeric(plot_data["volume"], errors="coerce").fillna(0.0)
        fig_liq = px.scatter(
            plot_data,
            x="volume_clean",
            y="spreadPct_clean",
            size="openInterest_clean",
            color="type",
            hover_data=["strike", "expiration", "daysToExpiration", "lastPrice", "bid", "ask"],
            labels={"volume_clean": "Volume", "spreadPct_clean": "Avg Spread %", "type": "Option Type"},
            title=f"{ticker} Liquidity Map (Volume vs Avg Spread %)",
        )
        fig_liq.update_layout(template="plotly_white", height=420)
        chart_liquidity_html = fig_liq.to_html(full_html=False, include_plotlyjs=False)

    chart_iv_html = "<p>No reliable IV data available for smile chart.</p>"
    iv_df = filtered_df[filtered_df["ivReliable"] & filtered_df["impliedVolatility"].notna()] if not filtered_df.empty else pd.DataFrame()
    if not iv_df.empty:
        fig_iv = px.line(
            iv_df.sort_values(["daysToExpiration", "strike"]),
            x="strike",
            y="impliedVolatility",
            color="expiration",
            line_dash="type",
            hover_data=["daysToExpiration"],
            title=f"{ticker} Implied Volatility Smile by Expiration",
            labels={"strike": "Strike Price", "impliedVolatility": "Implied Volatility"},
        )
        fig_iv.update_layout(template="plotly_white", height=420)
        chart_iv_html = fig_iv.to_html(full_html=False, include_plotlyjs=False)

    max_price_init = max_premium if max_premium is not None else 999.0
    price_display = f"${underlying_price:.2f}" if underlying_price is not None and not math.isnan(underlying_price) else "N/A"

    table_rows = []
    for _, row in filtered_df.iterrows():
        is_flagged = bool(row.get("noBid", False) or row.get("staleQuote", False) or row.get("arbitrageViolation", False))
        row_cls = ' class="flagged-row"' if is_flagged else ""

        def fmt_val(v, is_pct=False, decimals=2):
            if v is None or pd.isna(v) or not math.isfinite(float(v)):
                return "-"
            num = float(v)
            return f"{num:.{decimals}f}%" if is_pct else f"{num:.{decimals}f}"

        avg_spread_val = row.get("avgSpreadPct", row.get("spreadPct"))
        sample_cnt = int(row.get("spreadSampleCount", 0) or 0)
        avg_spread_title = html_lib.escape(f"Averaged across {sample_cnt} sample(s)")

        cells = [
            html_lib.escape(str(row.get("type", "-"))),
            html_lib.escape(str(row.get("expiration", "-"))),
            str(int(pd.to_numeric(row.get("daysToExpiration", 0), errors="coerce") or 0)),
            fmt_val(row.get("strike")),
            fmt_val(row.get("moneynessPct"), is_pct=True),
            fmt_val(row.get("lastPrice")),
            fmt_val(row.get("bid")),
            fmt_val(row.get("ask")),
            fmt_val(row.get("midPrice")),
            fmt_val(avg_spread_val, is_pct=True),
            str(int(pd.to_numeric(row.get("volume", 0), errors="coerce") or 0)),
            str(int(pd.to_numeric(row.get("openInterest", 0), errors="coerce") or 0)),
            fmt_val(row.get("volumeOiRatio")),
            fmt_val(row.get("impliedVolatility"), is_pct=True),
            fmt_val(row.get("delta"), decimals=3),
            fmt_val(row.get("theta"), decimals=3),
            "YES" if row.get("noBid") else "No",
            "YES" if row.get("staleQuote") else "No",
        ]
        rendered_cells = []
        for idx, c in enumerate(cells):
            if idx == 9:
                rendered_cells.append(f'<td title="{avg_spread_title}">{c}</td>')
            else:
                rendered_cells.append(f"<td>{c}</td>")
        table_rows.append(f"<tr{row_cls}>" + "".join(rendered_cells) + "</tr>")

    table_body = "\n".join(table_rows)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{ticker} Options Scan Dashboard</title>
    <style>{DASHBOARD_CSS}</style>
    <script>{SORT_JS}</script>
</head>
<body>
    {watchlist_bar_html}
    <div class="header-row">
        <div>
            <h1>{ticker} Options Scanner Dashboard</h1>
            <div class="meta">Underlying Price: <strong>{price_display}</strong> | Fetched: {now_str} | Spread Samples: <strong>{sample_rounds}</strong> (every {SPREAD_SAMPLE_INTERVAL_SECONDS}s)</div>
        </div>
    </div>

    {warn_html}

    <h2>Volatility, Map Bias & Pricing Overview</h2>
    <div class="stat-row">
        <div class="stat-box {bias_css_cls}">
            <div class="label">Underlying Bias (Maps)</div>
            <div class="value">{map_bias}</div>
            <div class="sub">Score: {map_bias_score:+.2f}</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">30D Realized HV</div>
            <div class="value">{hv_str}</div>
            <div class="sub">Past Movement</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">ATM Implied Vol (IV)</div>
            <div class="value">{iv_str}</div>
            <div class="sub">Market Expectation</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">IV / HV Ratio</div>
            <div class="value">{ratio_str}</div>
            <div class="sub">{bias_str}</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">HV Rank (1Y)</div>
            <div class="value">{rank_str}</div>
            <div class="sub">Range Relative</div>
        </div>
    </div>
    <div class="bias-note"><strong>Latest Map Interpretation ({map_bias}):</strong> {map_bias_summary}</div>

    <div class="stat-row" style="margin-top: 12px;">
        <div class="stat-box"><div class="label">Total Chain Contracts</div><div class="value">{total_contracts}</div></div>
        <div class="stat-box"><div class="label">Matching Filter</div><div class="value">{filtered_count}</div></div>
        <div class="stat-box"><div class="label">Calls</div><div class="value">{call_count}</div></div>
        <div class="stat-box"><div class="label">Puts</div><div class="value">{put_count}</div></div>
    </div>

    <h2>Order Flow Story & Market Interpretation Trend</h2>
    <div class="story-banner"><strong>Order Flow Story:</strong> {story_narrative_html}</div>
    <div class="chart-box">{chart_story_html}</div>

    <h2>Liquidity Map & Implied Volatility by Expiration</h2>
    <div class="chart-box">{chart_liquidity_html}</div>
    <div class="chart-box">{chart_iv_html}</div>

    <h2>Filtered Option Contracts</h2>
    <div class="filter-bar">
        <div class="filter-field">
            <label>Type</label>
            <select id="filter-type" onchange="saveFilterState(); filterTable()">
                <option value="ALL">All</option>
                <option value="Call">Call</option>
                <option value="Put">Put</option>
            </select>
        </div>
        <div class="filter-field">
            <label>Min DTE</label>
            <div class="slider-group">
                <input type="range" id="filter-dte-slider" min="0" max="365" value="0">
                <input type="number" id="filter-dte-num" min="0" max="365" value="0">
            </div>
        </div>
        <div class="filter-field">
            <label>Max DTE</label>
            <div class="slider-group">
                <input type="range" id="filter-max-dte-slider" min="0" max="365" value="365">
                <input type="number" id="filter-max-dte-num" min="0" max="365" value="365">
            </div>
        </div>
        <div class="filter-field">
            <label>Max Option Price ($)</label>
            <div class="slider-group">
                <input type="range" id="filter-price-slider" min="0" max="1000" value="{max_price_init}">
                <input type="number" id="filter-price-num" min="0" max="1000" value="{max_price_init}">
            </div>
        </div>
        <div class="filter-field">
            <label>Min Volume</label>
            <div class="slider-group">
                <input type="range" id="filter-vol-slider" min="0" max="5000" value="0">
                <input type="number" id="filter-vol-num" min="0" max="100000" value="0">
            </div>
        </div>
        <div class="filter-field">
            <label>Max Avg Spread %</label>
            <div class="slider-group">
                <input type="range" id="filter-spread-slider" min="0" max="100" value="100">
                <input type="number" id="filter-spread-num" min="0" max="100" value="100">
            </div>
        </div>
        <div class="filter-field">
            <label>Min OI</label>
            <div class="slider-group">
                <input type="range" id="filter-oi-slider" min="0" max="5000" value="0">
                <input type="number" id="filter-oi-num" min="0" max="100000" value="0">
            </div>
        </div>
        <button class="filter-reset" onclick="resetFilters()">Reset Filters</button>
        <div class="filter-count" id="filter-count">{filtered_count} matching contracts</div>
    </div>

    <div class="note">Click any table header to sort. Avg Spread% averages valid spread% samples across snapshots (hover cell for sample count). Rows with data-quality issues are highlighted in red.</div>
    <div class="table-wrap">
        <table class="data-table" id="scan-results-table" data-sort-col="" data-sort-dir="">
            <thead>
                <tr>
                    <th onclick="sortTable('scan-results-table', 0)">Type</th>
                    <th onclick="sortTable('scan-results-table', 1)">Exp</th>
                    <th onclick="sortTable('scan-results-table', 2)">DTE</th>
                    <th onclick="sortTable('scan-results-table', 3)">Strike</th>
                    <th onclick="sortTable('scan-results-table', 4)">Moneyness%</th>
                    <th onclick="sortTable('scan-results-table', 5)">Last</th>
                    <th onclick="sortTable('scan-results-table', 6)">Bid</th>
                    <th onclick="sortTable('scan-results-table', 7)">Ask</th>
                    <th onclick="sortTable('scan-results-table', 8)">Mid</th>
                    <th onclick="sortTable('scan-results-table', 9)">Avg Spread%</th>
                    <th onclick="sortTable('scan-results-table', 10)">Vol</th>
                    <th onclick="sortTable('scan-results-table', 11)">OI</th>
                    <th onclick="sortTable('scan-results-table', 12)">Vol/OI</th>
                    <th onclick="sortTable('scan-results-table', 13)">IV</th>
                    <th onclick="sortTable('scan-results-table', 14)">Delta</th>
                    <th onclick="sortTable('scan-results-table', 15)">Theta</th>
                    <th onclick="sortTable('scan-results-table', 16)">NoBid</th>
                    <th onclick="sortTable('scan-results-table', 17)">Stale</th>
                </tr>
            </thead>
            <tbody>
                {table_body}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


STATE_PATH = Path(os.getenv("STATE_FILE", "docs/data/state.json"))


class DashboardState:
    def __init__(self, config, state_path: Path = STATE_PATH):
        self.config = config
        self.state_path = state_path
        self.dashboards = {}
        self.spread_history = {}
        self.market_history = {}
        self.sample_rounds = {}
        self._load()

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.spread_history = data.get("spread_history", {})
            self.market_history = data.get("market_history", {})
            self.sample_rounds = data.get("sample_rounds", {})
            print(f"[State] Loaded persisted history from {self.state_path}.")
        except Exception as exc:
            print(f"[State] Could not load {self.state_path} ({exc}); starting fresh.")

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_rounds": self.sample_rounds,
            "market_history": self.market_history,
            "spread_history": self.spread_history,
        }
        tmp_path = self.state_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(self.state_path)
        print(f"[State] Saved persisted history to {self.state_path}.")

    def record_spread_sample(self, ticker, contract_key, spread_pct):
        ticker_hist = self.spread_history.setdefault(ticker, {})
        samples = ticker_hist.setdefault(contract_key, [])
        if spread_pct is not None and not pd.isna(spread_pct) and math.isfinite(float(spread_pct)):
            samples.append(float(spread_pct))
        return list(samples)

    def record_market_snapshot(self, ticker, sample_round, underlying_price, vol_insights, timestamp_str=None):
        ts = timestamp_str or datetime.now().strftime("%H:%M:%S")
        spot_val = (
            round(float(underlying_price), 2)
            if underlying_price is not None and not math.isnan(underlying_price)
            else None
        )
        entry = {
            "round": sample_round,
            "timestamp": ts,
            "spot": spot_val,
            "bias": vol_insights.get("map_bias", "Neutral"),
            "score": float(vol_insights.get("map_bias_score", 0.0) or 0.0),
            "summary": vol_insights.get("map_bias_summary", ""),
            "call_vol_share": vol_insights.get("call_vol_share"),
            "call_oi_share": vol_insights.get("call_oi_share"),
            "vol_tilt": vol_insights.get("vol_tilt", 0.0),
            "oi_tilt": vol_insights.get("oi_tilt", 0.0),
            "skew_diff_pct": vol_insights.get("skew_diff_pct"),
            "iv_slope_pct": vol_insights.get("iv_slope_pct"),
            "atm_iv": vol_insights.get("atm_iv"),
            "current_30d_hv": vol_insights.get("current_30d_hv"),
            "iv_hv_ratio": vol_insights.get("iv_hv_ratio"),
            "total_weighted_vol": vol_insights.get("total_weighted_vol", 0.0),
        }
        hist = self.market_history.setdefault(ticker, [])
        hist.append(entry)
        return list(hist)

    def increment_sample_round(self, ticker):
        self.sample_rounds[ticker] = self.sample_rounds.get(ticker, 0) + 1
        return self.sample_rounds[ticker]

    def set_dashboard(self, ticker, html):
        self.dashboards[ticker] = html


GLOBAL_STATE = None


def print_minutely_data(ticker, underlying_price, full_df, filtered_df, vol_insights, sample_round):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    px_str = f"${underlying_price:.2f}" if underlying_price is not None and not math.isnan(underlying_price) else "N/A"
    hv = vol_insights.get("current_30d_hv")
    iv = vol_insights.get("atm_iv")
    ratio = vol_insights.get("iv_hv_ratio")
    bias = vol_insights.get("map_bias", "Neutral")
    score = vol_insights.get("map_bias_score", 0.0)
    call_vol = vol_insights.get("call_vol_share")

    spread_col = "avgSpreadPct" if "avgSpreadPct" in full_df.columns else "spreadPct"
    valid_spreads = pd.to_numeric(full_df.get(spread_col), errors="coerce").dropna()
    mean_spread = f"{valid_spreads.mean():.2f}%" if not valid_spreads.empty else "N/A"

    hv_str = f"{hv:.1f}%" if hv is not None else "N/A"
    iv_str = f"{iv:.1f}%" if iv is not None else "N/A"
    ratio_str = f"{ratio:.2f}" if ratio is not None else "N/A"
    cvol_str = f"{call_vol:.1f}%" if call_vol is not None else "N/A"

    print(
        f"  [Snapshot | #{sample_round} @ {ts}] {ticker}: "
        f"Spot={px_str} | Contracts={len(full_df)} (Filtered={len(filtered_df)}) | "
        f"AvgSpread={mean_spread} | CallVol={cvol_str} | 30dHV={hv_str} | ATM_IV={iv_str} | "
        f"IV/HV={ratio_str} | Bias={bias} ({score:+.2f})"
    )


def scan_and_cache_ticker(ticker, config, watchlist_tickers=None, force_refresh=False):
    tk = yf.Ticker(ticker)
    underlying_price = get_current_price(tk)
    expirations = tk.options
    if not expirations:
        raise RuntimeError(f"No options expirations returned for {ticker}.")

    cutoff = date.today() + timedelta(days=365)
    valid_exps = []
    for e in expirations:
        try:
            d = datetime.strptime(str(e), "%Y-%m-%d").date()
            if d <= cutoff:
                valid_exps.append(str(e))
        except ValueError:
            continue

    if not valid_exps:
        raise RuntimeError(f"No expirations within 365 days for {ticker}.")

    all_dfs = []
    now_ts = datetime.now()

    for exp in valid_exps:
        chain = fetch_one_expiration_raw(tk, exp)
        if chain is None:
            continue

        calls_df = add_computed_metrics(chain.calls, "Call", exp, underlying_price, now_ts)
        puts_df = add_computed_metrics(chain.puts, "Put", exp, underlying_price, now_ts)
        exp_df = pd.concat([calls_df, puts_df], ignore_index=True)

        if not exp_df.empty:
            all_dfs.append(exp_df)

        if config.delay and config.delay > 0:
            time.sleep(config.delay)

    if not all_dfs:
        raise RuntimeError(f"No option chain rows retrieved across expirations for {ticker}.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    if full_df.empty:
        raise RuntimeError(f"Combined option chain is empty for {ticker}.")

    full_df = attach_average_spread(full_df, GLOBAL_STATE, ticker)
    sample_round = GLOBAL_STATE.increment_sample_round(ticker) if GLOBAL_STATE else 1
    filtered_df = apply_filters(full_df, config)
    warnings = assess_snapshot_quality(full_df)
    vol_insights = get_volatility_insights(tk, full_df, underlying_price)

    market_history = (
        GLOBAL_STATE.record_market_snapshot(
            ticker,
            sample_round,
            underlying_price,
            vol_insights,
            timestamp_str=now_ts.strftime("%H:%M:%S"),
        )
        if GLOBAL_STATE
        else []
    )

    html = build_interactive_html(
        ticker,
        underlying_price,
        full_df,
        filtered_df,
        warnings,
        vol_insights,
        config.max_premium,
        sample_rounds=sample_round,
        market_history=market_history,
        watchlist_tickers=watchlist_tickers or config.tickers,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ticker}.html"
    out_file.write_text(html, encoding="utf-8")

    if GLOBAL_STATE:
        GLOBAL_STATE.set_dashboard(ticker, html)

    print_minutely_data(ticker, underlying_price, full_df, filtered_df, vol_insights, sample_round)
    return True


def refresh_ticker_with_retry(ticker, config, watchlist_tickers=None, force_refresh=True, max_attempts_before_log=3):
    attempt = 0
    while True:
        attempt += 1
        try:
            scan_and_cache_ticker(ticker, config, watchlist_tickers=watchlist_tickers, force_refresh=force_refresh)
            return True
        except Exception as exc:
            print(
                f"  [Retry #{attempt}] Data refresh failed for {ticker}: {exc}. "
                f"Retrying in {RETRY_DELAY_SECONDS}s..."
            )
            if max_attempts_before_log is not None and attempt >= max_attempts_before_log:
                return False
            time.sleep(RETRY_DELAY_SECONDS)


def main():
    config = get_scan_config()
    global GLOBAL_STATE
    GLOBAL_STATE = DashboardState(config)

    for ticker in config.tickers:
        print(f"\nProcessing {ticker}...")
        refresh_ticker_with_retry(
            ticker,
            config,
            watchlist_tickers=config.tickers,
            force_refresh=True,
            max_attempts_before_log=3,
        )

    GLOBAL_STATE.save()
    print("\n50-ticker snapshot generation complete.")


if __name__ == "__main__":
    main()
