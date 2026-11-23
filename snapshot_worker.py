#!/usr/bin/env python3
"""
Cloud Options Snapshot & Volatility Dashboard Builder (Single-Page Watchlist Edition)
Includes data-integrity guards:
1. Failed or incomplete ticker pulls preserve prior valid snapshot & history entries.
2. Broken/unreliable quotes (no-bid, crossed markets, below-intrinsic arbitrage, or tiny volume/OI)
   are excluded from spread averages, implied volatility calculations, and bias scoring.
3. Rolling history caps keep trend charts clean, bounded, and responsive.
"""

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


SPREAD_SAMPLE_INTERVAL_SECONDS = 60
BACKGROUND_REFRESH_INTERVAL_MINUTES = 15
RETRY_DELAY_SECONDS = 5

MIN_OI_FOR_RATIO = 25
MIN_VOLUME_FOR_FLOW = 10
MAX_SPREAD_SAMPLES_PER_CONTRACT = 80
MAX_GREEK_CALC_SCALE = 80
MAX_MARKET_HISTORY_SAMPLES = 80
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
        "AAL, AAPL, AMZN, AVGO, BAC, CLF, CLSK, COIN, CSCO, F, GOOGL, IBIT, JD, KO, MARA, NVDA, ONDS, OSCR, PEP, PFE, PG, PLTR, QQQ, RIOT, RIVN, RKLB, S, SBUX, SLV, SOFI, SPCX, SPY, SQQQ, T, TSLA, UBER, WMT, WULF, XLE, XLF, XLU, XLP"
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

    min_premium = 0.04
    raw_min = os.getenv("MIN_PREMIUM", "").strip()
    if raw_min:
        try:
            min_premium = float(raw_min)
        except ValueError:
            sys.exit(f"'{raw_min}' isn't a valid number.")

    max_premium = 4.0
    raw_max = os.getenv("MAX_PREMIUM", "").strip()
    if raw_max:
        try:
            max_premium = float(raw_max)
        except ValueError:
            sys.exit(f"'{raw_max}' isn't a valid number.")

    return SimpleNamespace(
        tickers=tickers,
        max_premium=max_premium,
        min_premium=min_premium,
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
                val = float(close_series.iloc[-1])
                if math.isfinite(val) and val > 0:
                    return val
    except Exception:
        pass
    return math.nan


def _clean_num(val, decimals=2):
    if val is None or (isinstance(val, float) and (math.isnan(val) or not math.isfinite(val))):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, decimals)


def compute_map_bias_flag(df):
    result = {
        "bias": "Neutral",
        "score": 0.0,
        "summary": "Insufficient clean liquidity or IV data to establish directional bias.",
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

    # Only include contracts with clean, non-crossed, non-arbitrage quotes for flow/bias signals
    clean_mask = pd.Series(True, index=df.index)
    if "noBid" in df.columns:
        clean_mask &= ~df["noBid"].fillna(False)
    if "crossedMarket" in df.columns:
        clean_mask &= ~df["crossedMarket"].fillna(False)
    if "arbitrageViolation" in df.columns:
        clean_mask &= ~df["arbitrageViolation"].fillna(False)

    liq_df = df[clean_mask].copy()
    if liq_df.empty:
        return result

    spread_col = "avgSpreadPct" if "avgSpreadPct" in liq_df.columns else "spreadPct"
    spread_vals = pd.to_numeric(liq_df.get(spread_col), errors="coerce").fillna(50.0).clip(lower=0.0, upper=100.0)
    liq_weight = 1.0 / (1.0 + spread_vals / 50.0)

    raw_vol = pd.to_numeric(liq_df.get("volume"), errors="coerce").fillna(0.0)
    active_vol = raw_vol.where(raw_vol >= MIN_VOLUME_FOR_FLOW, 0.0)
    vol_series = active_vol * liq_weight

    raw_oi = pd.to_numeric(liq_df.get("openInterest"), errors="coerce").fillna(0.0)
    active_oi = raw_oi.where(raw_oi >= MIN_OI_FOR_RATIO, 0.0)
    oi_series = active_oi * liq_weight

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

    iv_reliable_mask = liq_df["ivReliable"] if "ivReliable" in liq_df.columns else pd.Series(True, index=liq_df.index)
    iv_series = pd.to_numeric(liq_df.get("impliedVolatility"), errors="coerce")
    # Sanity-bound IV to 1%..400% so corrupted Yahoo IV spikes do not distort skew or slope
    valid_iv_range = iv_series.notna() & (iv_series > 0.01) & (iv_series < 4.0)
    iv_df = liq_df[iv_reliable_mask.fillna(True) & valid_iv_range].copy()
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

    score = max(-2.4, min(2.4, score))
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
            close_vals = close_vals[close_vals > 0]
            if len(close_vals) >= 30:
                log_returns = np.log(close_vals / close_vals.shift(1))
                rolling_30d_hv = log_returns.rolling(window=30).std() * np.sqrt(252) * 100.0
                hv_series = rolling_30d_hv.dropna()
                hv_series = hv_series[np.isfinite(hv_series) & (hv_series > 0) & (hv_series < 300)]

                if not hv_series.empty:
                    current_hv = float(hv_series.iloc[-1])
                    hv_high = float(hv_series.max())
                    hv_low = float(hv_series.min())

                    hv_rank = 0.0
                    if hv_high > hv_low:
                        hv_rank = ((current_hv - hv_low) / (hv_high - hv_low)) * 100.0
                        hv_rank = max(0.0, min(100.0, hv_rank))

                    insights["current_30d_hv"] = round(current_hv, 2)
                    insights["hv_52w_high"] = round(hv_high, 2)
                    insights["hv_52w_low"] = round(hv_low, 2)
                    insights["hv_rank"] = round(hv_rank, 1)
    except Exception:
        pass

    try:
        if full_df is not None and not full_df.empty and underlying_price and math.isfinite(underlying_price) and underlying_price > 0:
            iv_series = pd.to_numeric(full_df.get("impliedVolatility"), errors="coerce")
            strike_series = pd.to_numeric(full_df.get("strike"), errors="coerce")
            reliable_mask = full_df["ivReliable"] if "ivReliable" in full_df.columns else pd.Series(True, index=full_df.index)
            no_bid_mask = full_df["noBid"] if "noBid" in full_df.columns else pd.Series(False, index=full_df.index)

            valid_ivs = full_df[
                reliable_mask.fillna(True)
                & (~no_bid_mask.fillna(False))
                & iv_series.notna()
                & (iv_series > 0.01)
                & (iv_series < 4.0)
                & strike_series.notna()
                & (strike_series > 0)
            ].copy()
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

    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    df["openInterest"] = df["openInterest"].fillna(0).clip(lower=0)

    bid = df["bid"].fillna(0)
    ask = df["ask"].fillna(0)
    mid = (bid + ask) / 2.0
    df["midPrice"] = mid
    spread = ask - bid
    df["spreadDollar"] = spread

    has_two_sided_market = (bid > 0) & (ask > 0)
    crossed_market = (bid > 0) & (ask > 0) & (ask < bid)
    valid_spread = has_two_sided_market & ~crossed_market & (mid > 0)

    if underlying_price and not math.isnan(underlying_price) and underlying_price > 0:
        if option_type == "Call":
            intrinsic = (underlying_price - df["strike"]).clip(lower=0)
        else:
            intrinsic = (df["strike"] - underlying_price).clip(lower=0)
        df["arbitrageViolation"] = (bid > 0) & (bid < intrinsic - ARBITRAGE_TOLERANCE)
    else:
        df["arbitrageViolation"] = False

    clean_spread_mask = valid_spread & ~df["arbitrageViolation"]
    raw_spread_pct = spread.where(clean_spread_mask) / mid.where(clean_spread_mask) * 100.0
    # Bound extreme bid/ask glitches above 250% spread so single-quote anomalies don't skew averages
    df["spreadPct"] = raw_spread_pct.where(raw_spread_pct.isna() | ((raw_spread_pct >= 0) & (raw_spread_pct <= 250.0)))

    volume = df["volume"].fillna(0)
    open_interest = df["openInterest"].fillna(0)
    reliable_oi = open_interest >= MIN_OI_FOR_RATIO
    df["volumeOiRatio"] = volume.where(reliable_oi) / open_interest.where(reliable_oi)

    if underlying_price and not math.isnan(underlying_price) and underlying_price > 0:
        df["moneynessPct"] = (df["strike"] - underlying_price) / underlying_price * 100.0
        df["inTheMoney"] = (df["strike"] < underlying_price) if option_type == "Call" else (df["strike"] > underlying_price)
    else:
        df["moneynessPct"] = math.nan
        df["inTheMoney"] = None

    df["noBid"] = bid <= 0
    df["crossedMarket"] = crossed_market
    df["lowOiConfidence"] = ~reliable_oi

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

    df["ivReliable"] = clean_spread_mask

    # Prefer Yahoo Finance Greeks when provided; fall back to Black-Scholes if unavailable
    yf_greek_aliases = {
        "delta": ["delta", "Delta"],
        "gamma": ["gamma", "Gamma"],
        "theta": ["theta", "Theta"],
        "vega": ["vega", "Vega"],
    }
    yf_cols = {}
    for target_col, candidates in yf_greek_aliases.items():
        matched_series = None
        for cand in candidates:
            if cand in df.columns:
                s = pd.to_numeric(df[cand], errors="coerce")
                if s.notna().any():
                    matched_series = s
                    break
        yf_cols[target_col] = matched_series if matched_series is not None else pd.Series(math.nan, index=df.index)

    T_years = days_to_expiration / 365.0
    deltas, gammas, thetas, vegas, computed_flags = [], [], [], [], []
    for idx, (strike, sigma, is_rel) in enumerate(zip(df["strike"], df["impliedVolatility"], df["ivReliable"])):
        d_val = yf_cols["delta"].iloc[idx]
        g_val = yf_cols["gamma"].iloc[idx]
        t_val = yf_cols["theta"].iloc[idx]
        v_val = yf_cols["vega"].iloc[idx]

        need_calc = (
            pd.isna(d_val) or not math.isfinite(float(d_val))
            or pd.isna(t_val) or not math.isfinite(float(t_val))
        )
        if need_calc and is_rel and sigma is not None and not pd.isna(sigma) and (0.01 < float(sigma) < 4.0):
            bs = black_scholes_greeks(option_type, underlying_price, strike, T_years, float(sigma))
            used_any_calc = False
            if (pd.isna(d_val) or not math.isfinite(float(d_val))) and math.isfinite(bs["delta"]):
                d_val = bs["delta"]
                used_any_calc = True
            if (pd.isna(g_val) or not math.isfinite(float(g_val))) and math.isfinite(bs["gamma"]):
                g_val = bs["gamma"]
                used_any_calc = True
            if (pd.isna(t_val) or not math.isfinite(float(t_val))) and math.isfinite(bs["theta"]):
                t_val = bs["theta"]
                used_any_calc = True
            if (pd.isna(v_val) or not math.isfinite(float(v_val))) and math.isfinite(bs["vega"]):
                v_val = bs["vega"]
                used_any_calc = True
            computed_flags.append(used_any_calc)
        else:
            computed_flags.append(False)

        deltas.append(d_val)
        gammas.append(g_val)
        thetas.append(t_val)
        vegas.append(v_val)

    df["delta"], df["gamma"], df["theta"], df["vega"] = deltas, gammas, thetas, vegas
    df["greekComputed"] = computed_flags

    df["contractKey"] = [make_contract_key(r) for _, r in df.iterrows()]
    return df


def attach_average_spread(df, state, ticker):
    df = df.copy()
    if state is None:
        df["avgSpreadPct"] = df["spreadPct"]
        df["spreadSampleCount"] = df["spreadPct"].notna().astype(int)
        df["greekCalcCount"] = df.get("greekComputed", pd.Series(False, index=df.index)).astype(int)
        return df

    avg_vals = []
    sample_counts = []
    greek_calc_counts = []
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

        was_computed = bool(row.get("greekComputed", False))
        calc_count = state.record_greek_calc(ticker, key, was_computed)
        greek_calc_counts.append(calc_count)

    df["avgSpreadPct"] = avg_vals
    df["spreadSampleCount"] = sample_counts
    df["greekCalcCount"] = greek_calc_counts
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


def serialize_ticker_snapshot(ticker, underlying_price, full_df, filtered_df, warnings, vol_insights, sample_round, fetched_at):
    contracts = []
    for _, row in filtered_df.iterrows():
        iv_raw = row.get("impliedVolatility")
        iv_reliable = bool(row.get("ivReliable", False))
        iv_clean = None
        if iv_reliable and iv_raw is not None and not pd.isna(iv_raw) and math.isfinite(float(iv_raw)):
            iv_val = float(iv_raw)
            if 0.01 < iv_val < 4.0:
                iv_clean = _clean_num(iv_val * 100.0, 2)

        contracts.append({
            "type": str(row.get("type", "-")),
            "exp": str(row.get("expiration", "-")),
            "dte": int(pd.to_numeric(row.get("daysToExpiration", 0), errors="coerce") or 0),
            "strike": _clean_num(row.get("strike"), 2),
            "moneyness": _clean_num(row.get("moneynessPct"), 2),
            "last": _clean_num(row.get("lastPrice"), 2),
            "bid": _clean_num(row.get("bid"), 2),
            "ask": _clean_num(row.get("ask"), 2),
            "mid": _clean_num(row.get("midPrice"), 2),
            "avgSpread": _clean_num(row.get("avgSpreadPct", row.get("spreadPct")), 2),
            "samples": int(pd.to_numeric(row.get("spreadSampleCount", 0), errors="coerce") or 0),
            "vol": int(pd.to_numeric(row.get("volume", 0), errors="coerce") or 0),
            "oi": int(pd.to_numeric(row.get("openInterest", 0), errors="coerce") or 0),
            "volOi": _clean_num(row.get("volumeOiRatio"), 2),
            "iv": iv_clean,
            "delta": _clean_num(row.get("delta"), 3),
            "theta": _clean_num(row.get("theta"), 3),
            "greekComputed": bool(row.get("greekComputed", False)),
            "greekCalcCount": int(pd.to_numeric(row.get("greekCalcCount", 0), errors="coerce") or 0),
            "noBid": bool(row.get("noBid", False)),
            "stale": bool(row.get("staleQuote", False)),
            "arb": bool(row.get("arbitrageViolation", False)),
        })

    call_count = sum(1 for c in contracts if c["type"] == "Call")
    put_count = sum(1 for c in contracts if c["type"] == "Put")

    return {
        "ticker": ticker,
        "spot": _clean_num(underlying_price, 2),
        "fetched_at": fetched_at,
        "sample_rounds": sample_round,
        "refresh_count": sample_round,
        "refresh_interval_minutes": BACKGROUND_REFRESH_INTERVAL_MINUTES,
        "warnings": warnings,
        "insights": vol_insights,
        "counts": {
            "total": int(len(full_df)),
            "filtered": int(len(filtered_df)),
            "calls": int(call_count),
            "puts": int(put_count),
        },
        "contracts": contracts,
    }


STATE_PATH = Path(os.getenv("STATE_FILE", "docs/data/state.json"))
SNAPSHOTS_PATH = Path(os.getenv("SNAPSHOTS_FILE", "docs/data/snapshots.json"))


class DashboardState:
    def __init__(self, config, state_path: Path = STATE_PATH, snapshots_path: Path = SNAPSHOTS_PATH):
        self.config = config
        self.state_path = state_path
        self.snapshots_path = snapshots_path
        self.spread_history = {}
        self.market_history = {}
        self.greek_calc_counts = {}
        self.sample_rounds = {}
        self.snapshots = {}
        self._load()

    def _load(self):
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.spread_history = data.get("spread_history", {})
                self.market_history = data.get("market_history", {})
                self.greek_calc_counts = data.get("greek_calc_counts", {})
                self.sample_rounds = data.get("sample_rounds", {})
                print(f"[State] Loaded persisted history from {self.state_path}.")
            except Exception as exc:
                print(f"[State] Could not load {self.state_path} ({exc}); starting fresh.")

        if self.snapshots_path.exists():
            try:
                with open(self.snapshots_path, "r", encoding="utf-8") as f:
                    snap_data = json.load(f)
                if isinstance(snap_data, dict):
                    self.snapshots = snap_data.get("tickers", snap_data)
                print(f"[Snapshots] Loaded existing ticker snapshots from {self.snapshots_path}.")
            except Exception as exc:
                print(f"[Snapshots] Could not load {self.snapshots_path} ({exc}); starting fresh.")

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_rounds": self.sample_rounds,
            "market_history": self.market_history,
            "greek_calc_counts": self.greek_calc_counts,
            "spread_history": self.spread_history,
        }
        tmp_state = self.state_path.with_suffix(".tmp")
        with open(tmp_state, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_state.replace(self.state_path)
        print(f"[State] Saved persisted history to {self.state_path}.")

        self.snapshots_path.parent.mkdir(parents=True, exist_ok=True)
        snap_payload = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "refresh_interval_minutes": BACKGROUND_REFRESH_INTERVAL_MINUTES,
            "watchlist": self.config.tickers,
            "tickers": self.snapshots,
        }
        tmp_snap = self.snapshots_path.with_suffix(".tmp")
        with open(tmp_snap, "w", encoding="utf-8") as f:
            json.dump(snap_payload, f, separators=(",", ":"))
        tmp_snap.replace(self.snapshots_path)
        print(f"[Snapshots] Saved {len(self.snapshots)} ticker snapshots to {self.snapshots_path}.")

    def record_greek_calc(self, ticker, contract_key, was_computed: bool):
        ticker_counts = self.greek_calc_counts.setdefault(ticker, {})
        current = int(ticker_counts.get(contract_key, 0) or 0)
        if was_computed:
            current = min(MAX_GREEK_CALC_SCALE, current + 1)
        ticker_counts[contract_key] = current
        return current

    def record_spread_sample(self, ticker, contract_key, spread_pct):
        ticker_hist = self.spread_history.setdefault(ticker, {})
        samples = ticker_hist.setdefault(contract_key, [])
        if spread_pct is not None and not pd.isna(spread_pct):
            fval = float(spread_pct)
            if math.isfinite(fval) and 0.0 <= fval <= 250.0:
                samples.append(round(fval, 4))
                if len(samples) > MAX_SPREAD_SAMPLES_PER_CONTRACT:
                    del samples[:-MAX_SPREAD_SAMPLES_PER_CONTRACT]
        return list(samples)

    def record_market_snapshot(self, ticker, sample_round, underlying_price, vol_insights, timestamp_str=None):
        ts = timestamp_str or datetime.now().strftime("%H:%M:%S")
        spot_val = (
            round(float(underlying_price), 2)
            if underlying_price is not None and math.isfinite(underlying_price) and underlying_price > 0
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
        if len(hist) > MAX_MARKET_HISTORY_SAMPLES:
            del hist[:-MAX_MARKET_HISTORY_SAMPLES]
        return list(hist)

    def increment_sample_round(self, ticker):
        self.sample_rounds[ticker] = self.sample_rounds.get(ticker, 0) + 1
        return self.sample_rounds[ticker]

    def set_snapshot(self, ticker, snapshot_obj):
        self.snapshots[ticker] = snapshot_obj


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


def scan_and_cache_ticker(ticker, config):
    tk = yf.Ticker(ticker)
    underlying_price = get_current_price(tk)
    if underlying_price is None or not math.isfinite(underlying_price) or underlying_price <= 0:
        raise RuntimeError(f"Invalid or missing underlying price for {ticker}.")

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
            bid_s = pd.to_numeric(exp_df.get("bid"), errors="coerce")
            ask_s = pd.to_numeric(exp_df.get("ask"), errors="coerce")
            last_s = pd.to_numeric(exp_df.get("lastPrice"), errors="coerce")
            two_sided = bid_s.notna() & ask_s.notna() & (bid_s > 0) & (ask_s > 0) & (ask_s >= bid_s)
            ref_price = ((bid_s + ask_s) / 2.0).where(two_sided, last_s)
            price_ok = ref_price.notna()
            if config.min_premium is not None:
                price_ok &= ref_price >= config.min_premium
            if config.max_premium is not None:
                price_ok &= ref_price <= config.max_premium
            exp_df = exp_df[price_ok].copy()

        if not exp_df.empty:
            all_dfs.append(exp_df)

        if config.delay and config.delay > 0:
            time.sleep(config.delay)

    if not all_dfs:
        raise RuntimeError(f"No option chain rows retrieved across expirations for {ticker}.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    if full_df.empty:
        raise RuntimeError(f"Combined option chain is empty for {ticker}.")

    # Reject corrupted chain fetches where every single contract has zero bid or non-positive strike
    valid_strikes = pd.to_numeric(full_df.get("strike"), errors="coerce")
    if (valid_strikes > 0).sum() == 0:
        raise RuntimeError(f"All option strikes invalid for {ticker}.")

    # Compute preliminary metrics & average spreads only after chain sanity check passes
    full_df = attach_average_spread(full_df, GLOBAL_STATE, ticker)
    filtered_df = apply_filters(full_df, config)
    warnings = assess_snapshot_quality(full_df)
    vol_insights = get_volatility_insights(tk, full_df, underlying_price)

    # Commit round, market_history, and snapshots.json atomically only after full calculation succeeds
    sample_round = GLOBAL_STATE.increment_sample_round(ticker) if GLOBAL_STATE else 1
    if GLOBAL_STATE:
        GLOBAL_STATE.record_market_snapshot(
            ticker,
            sample_round,
            underlying_price,
            vol_insights,
            timestamp_str=now_ts.strftime("%H:%M:%S"),
        )
        snap_obj = serialize_ticker_snapshot(
            ticker,
            underlying_price,
            full_df,
            filtered_df,
            warnings,
            vol_insights,
            sample_round,
            now_ts.strftime("%Y-%m-%d %H:%M:%S"),
        )
        GLOBAL_STATE.set_snapshot(ticker, snap_obj)

    print_minutely_data(ticker, underlying_price, full_df, filtered_df, vol_insights, sample_round)
    return True


def refresh_ticker_with_retry(ticker, config, max_attempts_before_log=3):
    attempt = 0
    while True:
        attempt += 1
        try:
            scan_and_cache_ticker(ticker, config)
            return True
        except Exception as exc:
            print(
                f"  [Retry #{attempt}] Data refresh failed for {ticker}: {exc}. "
                f"Retrying in {RETRY_DELAY_SECONDS}s..."
            )
            if max_attempts_before_log is not None and attempt >= max_attempts_before_log:
                print(
                    f"  [Preserved] Keeping previous valid snapshot & history for {ticker} "
                    f"after {attempt} failed attempt(s)."
                )
                return False
            time.sleep(RETRY_DELAY_SECONDS)


def build_index_page(out_dir: Path, watchlist: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Options Watchlist Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            margin: 0;
            padding: 24px 32px 60px;
            background: #f7f8fa;
            color: #1a1a1a;
        }
        .watchlist-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            background: white;
            border: 1px solid #e2e4e8;
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 18px;
        }
        .watchlist-bar label {
            font-size: 0.75em;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }
        .watchlist-bar input[type="text"],
        .watchlist-bar select {
            padding: 6px 10px;
            border: 1px solid #ccd2da;
            border-radius: 6px;
            font-size: 0.9em;
            background: #fff;
            color: #111827;
            text-transform: uppercase;
        }
        .watchlist-bar input[type="text"] {
            width: 210px;
        }
        .watchlist-bar select {
            min-width: 110px;
        }
        .view-ticker-btn {
            padding: 7px 15px;
            border: 1px solid #2563eb;
            border-radius: 6px;
            background: #2563eb;
            color: white;
            font-size: 0.88em;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
        }
        .view-ticker-btn:hover {
            background: #1d4ed8;
        }
        .status-pill {
            margin-left: auto;
            font-size: 0.82em;
            color: #475569;
        }
        h1 {
            font-size: 1.6em;
            margin-bottom: 4px;
            margin-top: 0;
        }
        h2 {
            font-size: 1.15em;
            margin-top: 28px;
            border-bottom: 2px solid #e2e4e8;
            padding-bottom: 6px;
        }
        .meta {
            color: #555;
            font-size: 0.92em;
            line-height: 1.6;
        }
        .warning-banner {
            display: none;
            background: #fff4e5;
            border: 1px solid #f0b429;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 16px 0;
        }
        .warning-banner.visible {
            display: block;
        }
        .warning-banner h2 {
            border: none;
            margin: 0 0 8px 0;
            color: #92400e;
        }
        .warning-banner ul {
            margin: 4px 0 0 0;
            padding-left: 20px;
        }
        .stat-row {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            margin: 10px 0 4px;
        }
        .stat-box {
            background: white;
            border: 1px solid #e2e4e8;
            border-radius: 8px;
            padding: 10px 16px;
            min-width: 140px;
        }
        .stat-box.highlight {
            border-color: #2563eb;
            background: #f0f5ff;
        }
        .stat-box.bias-bullish {
            border-color: #16a34a;
            background: #f0fdf4;
        }
        .stat-box.bias-bullish .value,
        .stat-box.bias-bullish .sub {
            color: #15803d;
        }
        .stat-box.bias-bearish {
            border-color: #dc2626;
            background: #fef2f2;
        }
        .stat-box.bias-bearish .value,
        .stat-box.bias-bearish .sub {
            color: #b91c1c;
        }
        .stat-box.bias-neutral {
            border-color: #64748b;
            background: #f8fafc;
        }
        .stat-box.bias-neutral .value,
        .stat-box.bias-neutral .sub {
            color: #475569;
        }
        .stat-box .label {
            font-size: 0.72em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            font-weight: 600;
        }
        .stat-box .value {
            font-size: 1.25em;
            font-weight: 700;
            color: #111827;
            margin-top: 2px;
        }
        .stat-box .sub {
            font-size: 0.75em;
            color: #2563eb;
            font-weight: 600;
            margin-top: 2px;
        }
        .bias-note {
            font-size: 0.88em;
            color: #374151;
            background: white;
            border: 1px solid #e2e4e8;
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 10px;
            line-height: 1.5;
        }
        .story-banner {
            font-size: 0.88em;
            color: #1e293b;
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            border-top: 1px solid #e2e4e8;
            border-right: 1px solid #e2e4e8;
            border-bottom: 1px solid #e2e4e8;
            border-radius: 6px;
            padding: 10px 14px;
            margin: 10px 0 8px;
            line-height: 1.5;
        }
        .chart-box {
            background: white;
            border: 1px solid #e2e4e8;
            border-radius: 8px;
            padding: 8px;
            margin: 14px 0;
            overflow-x: auto;
        }
        table.data-table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            font-size: 0.85em;
        }
        table.data-table th,
        table.data-table td {
            border: 1px solid #e2e4e8;
            padding: 6px 10px;
            text-align: right;
            white-space: nowrap;
        }
        table.data-table th:first-child,
        table.data-table td:first-child {
            text-align: left;
        }
        table.data-table th {
            background: #eef1f5;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
            z-index: 2;
        }
        table.data-table th:hover {
            background: #dfe4ea;
        }
        table.data-table tbody tr:nth-child(even) {
            background: #fafbfc;
        }
        table.data-table tbody tr.flagged-row {
            background: #fdecea;
            color: #9a3412;
        }
        table.data-table tbody tr:hover {
            background: #dbeafe !important;
        }
        .note {
            color: #666;
            font-size: 0.88em;
            font-style: italic;
            margin: 4px 0 10px;
        }
        .table-wrap {
            max-height: 600px;
            overflow: auto;
            border: 1px solid #e2e4e8;
            border-radius: 8px;
        }
        .filter-bar {
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
            align-items: flex-end;
            background: white;
            border: 1px solid #e2e4e8;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 10px 0 14px;
        }
        .filter-field {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .filter-field label {
            font-size: 0.72em;
            color: #777;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .filter-field select,
        .filter-field input[type="number"] {
            padding: 5px 8px;
            border: 1px solid #ccd2da;
            border-radius: 6px;
            font-size: 0.9em;
        }
        .filter-field input[type="number"] {
            width: 75px;
        }
        .slider-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .filter-field input[type="range"] {
            accent-color: #2563eb;
            cursor: pointer;
            width: 110px;
        }
        .filter-reset {
            padding: 6px 14px;
            border: 1px solid #ccd2da;
            border-radius: 6px;
            background: #f1f3f6;
            cursor: pointer;
            font-size: 0.85em;
            height: 30px;
        }
        .filter-reset:hover {
            background: #e2e6eb;
        }
        .filter-count {
            font-size: 0.85em;
            color: #555;
            margin-left: auto;
            align-self: center;
        }
        .notice-bar {
            display: none;
            background: #fff4e5;
            border: 1px solid #f0b429;
            color: #92400e;
            border-radius: 8px;
            padding: 10px 14px;
            margin: 12px 0;
            font-size: 0.9em;
        }
        .notice-bar.visible {
            display: block;
        }
    </style>
</head>
<body>
    <div class="watchlist-bar">
        <label for="watchlist-input" id="watchlist-label">WATCHLIST TICKER:</label>
        <input
            type="text"
            id="watchlist-input"
            list="watchlist-tickers"
            placeholder="Type ticker (e.g. XLE)"
            autocomplete="off"
            oninput="syncWatchlistFromInput()"
            onkeydown="if (event.key === 'Enter') { event.preventDefault(); applySelectedTicker(); }"
        >
        <datalist id="watchlist-tickers"></datalist>
        <select id="watchlist-select" onchange="syncWatchlistFromSelect()" aria-label="Select watchlist ticker"></select>
        <button type="button" class="view-ticker-btn" onclick="applySelectedTicker()">View Ticker</button>
        <span class="status-pill" id="batch-status">Loading latest snapshots...</span>
    </div>

    <div class="header-row">
        <div>
            <h1 id="dashboard-title">Options Scanner Dashboard</h1>
            <div class="meta" id="dashboard-meta">Loading latest background snapshot...</div>
        </div>
    </div>

    <div id="ticker-notice" class="notice-bar"></div>

    <div id="warning-banner" class="warning-banner">
        <h2>Data Quality Warnings</h2>
        <ul id="warning-list"></ul>
    </div>

    <h2>Volatility, Map Bias &amp; Pricing Overview</h2>
    <div class="stat-row">
        <div class="stat-box bias-neutral" id="bias-box">
            <div class="label">Underlying Bias (Maps)</div>
            <div class="value" id="stat-bias">-</div>
            <div class="sub" id="stat-bias-score">Score: -</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">30D Realized HV</div>
            <div class="value" id="stat-hv">-</div>
            <div class="sub">Past Movement</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">ATM Implied Vol (IV)</div>
            <div class="value" id="stat-iv">-</div>
            <div class="sub">Market Expectation</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">IV / HV Ratio</div>
            <div class="value" id="stat-ratio">-</div>
            <div class="sub" id="stat-pricing-bias">-</div>
        </div>
        <div class="stat-box highlight">
            <div class="label">HV Rank (1Y)</div>
            <div class="value" id="stat-hv-rank">-</div>
            <div class="sub">Range Relative</div>
        </div>
    </div>
    <div class="bias-note" id="bias-note"><strong>Latest Map Interpretation:</strong> Waiting for snapshot data...</div>

    <div class="stat-row" style="margin-top: 12px;">
        <div class="stat-box"><div class="label">Total Chain Contracts</div><div class="value" id="stat-total-contracts">0</div></div>
        <div class="stat-box"><div class="label">Matching Filter</div><div class="value" id="stat-matching-count">0</div></div>
        <div class="stat-box"><div class="label">Calls</div><div class="value" id="stat-calls">0</div></div>
        <div class="stat-box"><div class="label">Puts</div><div class="value" id="stat-puts">0</div></div>
    </div>

    <h2>Order Flow Story &amp; Market Interpretation Trend</h2>
    <div class="story-banner" id="story-banner"><strong>Order Flow Story:</strong> Waiting for persisted history...</div>
    <div class="chart-box">
        <div id="chart-story" style="width: 100%; height: 760px;"></div>
    </div>

    <h2>Liquidity Map &amp; Implied Volatility by Expiration</h2>
    <div class="chart-box">
        <div id="chart-liquidity" style="width: 100%; height: 420px;"></div>
    </div>
    <div class="chart-box">
        <div id="chart-iv" style="width: 100%; height: 420px;"></div>
    </div>

    <h2>Filtered Option Contracts</h2>
    <div class="filter-bar">
        <div class="filter-field">
            <label for="filter-type">Type</label>
            <select id="filter-type" onchange="saveFilterState(); filterTable()">
                <option value="ALL">All</option>
                <option value="Call">Call</option>
                <option value="Put">Put</option>
            </select>
        </div>
        <div class="filter-field">
            <label for="filter-dte-num">Min DTE</label>
            <div class="slider-group">
                <input type="range" id="filter-dte-slider" min="0" max="365" value="0">
                <input type="number" id="filter-dte-num" min="0" max="365" value="0">
            </div>
        </div>
        <div class="filter-field">
            <label for="filter-max-dte-num">Max DTE</label>
            <div class="slider-group">
                <input type="range" id="filter-max-dte-slider" min="0" max="365" value="365">
                <input type="number" id="filter-max-dte-num" min="0" max="365" value="365">
            </div>
        </div>
        <div class="filter-field">
            <label for="filter-price-num">Max Option Price ($)</label>
            <div class="slider-group">
                <input type="range" id="filter-price-slider" min="0" max="1000" value="999">
                <input type="number" id="filter-price-num" min="0" max="1000" value="999">
            </div>
        </div>
        <div class="filter-field">
            <label for="filter-vol-num">Min Volume</label>
            <div class="slider-group">
                <input type="range" id="filter-vol-slider" min="0" max="5000" value="0">
                <input type="number" id="filter-vol-num" min="0" max="100000" value="0">
            </div>
        </div>
        <div class="filter-field">
            <label for="filter-spread-num">Max Avg Spread %</label>
            <div class="slider-group">
                <input type="range" id="filter-spread-slider" min="0" max="100" value="100">
                <input type="number" id="filter-spread-num" min="0" max="100" value="100">
            </div>
        </div>
        <div class="filter-field">
            <label for="filter-oi-num">Min OI</label>
            <div class="slider-group">
                <input type="range" id="filter-oi-slider" min="0" max="5000" value="0">
                <input type="number" id="filter-oi-num" min="0" max="100000" value="0">
            </div>
        </div>
        <button type="button" class="filter-reset" onclick="resetFilters()">Reset Filters</button>
        <div class="filter-count" id="filter-count">0 matching contracts</div>
    </div>

    <div class="note">Click any table header to sort. Avg Spread% averages clean spread% samples across snapshots (hover cell for sample count). Delta &amp; Theta show Yahoo Greeks when available or compute them when missing, shading mildly more orange on a 0&ndash;80 scale each time Greeks are calculated while keeping text legible. Red rows highlight no-bid, stale, or below-intrinsic quotes.</div>
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
            <tbody id="contracts-tbody"></tbody>
        </table>
    </div>

    <script>
        let SNAPSHOT_PAYLOAD = { updated_at: null, watchlist: [], tickers: {} };
        let STATE_PAYLOAD = { saved_at: null, sample_rounds: {}, market_history: {}, spread_history: {} };
        let ACTIVE_TICKER = null;

        function getWatchlist() {
            const fromSnap = Array.isArray(SNAPSHOT_PAYLOAD.watchlist) ? SNAPSHOT_PAYLOAD.watchlist : [];
            const fromKeys = Object.keys(SNAPSHOT_PAYLOAD.tickers || {});
            const merged = [];
            const seen = new Set();
            for (const t of [...fromSnap, ...fromKeys]) {
                const up = String(t || "").trim().toUpperCase();
                if (up && !seen.has(up)) {
                    merged.push(up);
                    seen.add(up);
                }
            }
            return merged;
        }

        function populateWatchlistControls(selectedTicker) {
            const wl = getWatchlist();
            const lbl = document.getElementById("watchlist-label");
            if (lbl) {
                lbl.textContent = `WATCHLIST TICKER (${wl.length}):`;
            }
            const datalist = document.getElementById("watchlist-tickers");
            if (datalist) {
                datalist.innerHTML = wl.map(t => `<option value="${t}"></option>`).join("");
            }
            const select = document.getElementById("watchlist-select");
            if (select) {
                select.innerHTML = wl.map(
                    t => `<option value="${t}"${t === selectedTicker ? " selected" : ""}>${t}</option>`
                ).join("");
            }
            const input = document.getElementById("watchlist-input");
            if (input && selectedTicker) {
                input.value = selectedTicker;
            }
        }

        function syncWatchlistFromInput() {
            const input = document.getElementById("watchlist-input");
            const select = document.getElementById("watchlist-select");
            if (!input || !select) return;
            const val = input.value.trim().toUpperCase();
            const match = Array.from(select.options).find(o => o.value === val);
            if (match) {
                select.value = val;
            }
        }

        function syncWatchlistFromSelect() {
            const input = document.getElementById("watchlist-input");
            const select = document.getElementById("watchlist-select");
            if (!input || !select) return;
            input.value = select.value;
            applySelectedTicker();
        }

        function applySelectedTicker() {
            const input = document.getElementById("watchlist-input");
            const select = document.getElementById("watchlist-select");
            const raw = (input && input.value.trim()) ? input.value.trim() : (select ? select.value : "");
            const clean = raw.toUpperCase().replace(/[^A-Z0-9.\-]/g, "");
            if (!clean) return;
            renderTickerView(clean);
        }

        function escapeHtml(str) {
            return String(str ?? "")
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;");
        }

        function fmtNum(v, pct = false, dec = 2) {
            if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
            const n = Number(v);
            if (!Number.isFinite(n)) return "-";
            return pct ? `${n.toFixed(dec)}%` : n.toFixed(dec);
        }

        function summarizeNarrative(history) {
            if (!history || history.length === 0) {
                return "Awaiting snapshots to build the order-flow narrative.";
            }
            const first = history[0];
            const latest = history[history.length - 1];
            const n = history.length;

            if (n === 1) {
                const volTxt = (latest.call_vol_share !== null && latest.call_vol_share !== undefined)
                    ? `Call volume share is ${Number(latest.call_vol_share).toFixed(1)}%`
                    : "Call/put volume share is balanced";
                const skewTxt = (latest.skew_diff_pct !== null && latest.skew_diff_pct !== undefined)
                    ? `Put-Call IV skew stands at ${Number(latest.skew_diff_pct) >= 0 ? "+" : ""}${Number(latest.skew_diff_pct).toFixed(2)}%`
                    : "IV skew is flat";
                return `Initial snapshot (#${latest.round} at ${latest.timestamp}): Bias is <strong>${escapeHtml(latest.bias)}</strong> (score ${Number(latest.score) >= 0 ? "+" : ""}${Number(latest.score).toFixed(2)}; yellow neutral zone is -0.35 to +0.35). ${volTxt}; ${skewTxt}. Additional scheduled background runs will chart directional momentum.`;
            }

            const delta = Number(latest.score) - Number(first.score);
            let trend = `held steady (${Number(first.score) >= 0 ? "+" : ""}${Number(first.score).toFixed(2)} &rarr; ${Number(latest.score) >= 0 ? "+" : ""}${Number(latest.score).toFixed(2)})`;
            if (delta >= 0.15) {
                trend = `strengthened bullishly (${Number(first.score) >= 0 ? "+" : ""}${Number(first.score).toFixed(2)} &rarr; ${Number(latest.score) >= 0 ? "+" : ""}${Number(latest.score).toFixed(2)})`;
            } else if (delta <= -0.15) {
                trend = `shifted bearishly (${Number(first.score) >= 0 ? "+" : ""}${Number(first.score).toFixed(2)} &rarr; ${Number(latest.score) >= 0 ? "+" : ""}${Number(latest.score).toFixed(2)})`;
            }

            const flips = [];
            for (let i = 1; i < n; i++) {
                if (history[i].bias !== history[i - 1].bias) {
                    flips.push(`${history[i - 1].bias}&rarr;${history[i].bias} @ ${history[i].timestamp}`);
                }
            }
            const flipTxt = flips.length > 0
                ? ` Regime shifts: ${flips.join(", ")}.`
                : ` Regime remained <strong>${escapeHtml(latest.bias)}</strong> across all ${n} samples.`;

            let volTxt = "";
            if (first.call_vol_share !== null && latest.call_vol_share !== null) {
                const dv = Number(latest.call_vol_share) - Number(first.call_vol_share);
                volTxt = ` Call volume flow moved ${Number(first.call_vol_share).toFixed(1)}% &rarr; ${Number(latest.call_vol_share).toFixed(1)}% (${dv >= 0 ? "+" : ""}${dv.toFixed(1)}%).`;
            }

            let spotTxt = "";
            if (first.spot && latest.spot && Number(first.spot) > 0) {
                const dpPct = ((Number(latest.spot) - Number(first.spot)) / Number(first.spot)) * 100.0;
                spotTxt = ` Underlying moved $${Number(first.spot).toFixed(2)} &rarr; $${Number(latest.spot).toFixed(2)} (${dpPct >= 0 ? "+" : ""}${dpPct.toFixed(2)}%).`;
            }

            return `Across <strong>${n}</strong> recent snapshots (${first.timestamp} &ndash; ${latest.timestamp}), composite order-flow bias ${trend}.${flipTxt}${volTxt}${spotTxt}`;
        }

        function computeDeltas(values) {
            const base = values.find(v => v !== null && v !== undefined && Number.isFinite(Number(v)));
            if (base === undefined) return values.map(() => null);
            return values.map(v => {
                if (v === null || v === undefined || !Number.isFinite(Number(v))) return null;
                return +(Number(v) - Number(base)).toFixed(2);
            });
        }

        function symmetricRange(seriesList, minSpan = 5.0, padRatio = 0.30) {
            let peak = 0;
            for (const s of seriesList) {
                for (const v of s) {
                    if (v !== null && v !== undefined && Number.isFinite(Number(v))) {
                        peak = Math.max(peak, Math.abs(Number(v)));
                    }
                }
            }
            const bound = Math.max(minSpan, +(peak * (1.0 + padRatio)).toFixed(2));
            return [-bound, bound];
        }

        function renderTickerCharts(ticker, snap, history) {
            const storyEl = document.getElementById("chart-story");
            const liqEl = document.getElementById("chart-liquidity");
            const ivEl = document.getElementById("chart-iv");

            if (!history || history.length === 0) {
                Plotly.purge(storyEl);
                storyEl.innerHTML = "<p style='padding:12px;'>No market interpretation history recorded yet.</p>";
            } else {
                const times = history.map(h => h.timestamp);
                const rounds = history.map(h => h.round);
                const scores = history.map(h => h.score);
                const biases = history.map(h => h.bias);
                const summaries = history.map(h => h.summary || "");
                const spots = history.map(h => h.spot);
                const callVols = history.map(h => h.call_vol_share);
                const callOis = history.map(h => h.call_oi_share);
                const dCallVols = computeDeltas(callVols);
                const dCallOis = computeDeltas(callOis);
                const skews = history.map(h => h.skew_diff_pct);
                const slopes = history.map(h => h.iv_slope_pct);
                const atmIvs = history.map(h => h.atm_iv);

                const markerColors = biases.map(b => b === "Bullish" ? "#16a34a" : (b === "Bearish" ? "#dc2626" : "#ca8a04"));
                const hoverStory = rounds.map((r, i) =>
                    `<b>Sample #${r} (${times[i]})</b><br>Regime: <b>${biases[i]}</b> (Score: ${scores[i] >= 0 ? "+" : ""}${Number(scores[i]).toFixed(2)})<br>Drivers: ${escapeHtml(summaries[i])}`
                );

                const traces = [
                    {
                        x: times, y: scores, name: "Bias Score", mode: "lines+markers",
                        line: { color: "#1d4ed8", width: 2.5 },
                        marker: { size: 9, color: markerColors, line: { width: 1.5, color: "white" } },
                        text: hoverStory, hovertemplate: "%{text}<extra></extra>", xaxis: "x", yaxis: "y"
                    },
                    {
                        x: times, y: spots, name: "Spot Price ($)", mode: "lines+markers",
                        line: { color: "#f59e0b", width: 2, dash: "dash" },
                        marker: { size: 6, color: "#f59e0b" },
                        hovertemplate: "Spot: $%{y:.2f}<extra></extra>", xaxis: "x", yaxis: "y2"
                    },
                    {
                        x: times, y: dCallVols, name: "Δ Call Vol Share (pts)", mode: "lines+markers",
                        line: { color: "#0d9488", width: 2.4 },
                        marker: { size: 8, color: "#0d9488", line: { width: 1.2, color: "white" } },
                        text: dCallVols.map((dv, i) => dv !== null && callVols[i] !== null ? `Δ Call Vol: ${dv >= 0 ? "+" : ""}${dv.toFixed(2)} pts (Share: ${Number(callVols[i]).toFixed(1)}%)` : "Δ Call Vol: N/A"),
                        hovertemplate: "%{text}<extra></extra>", xaxis: "x", yaxis: "y3"
                    },
                    {
                        x: times, y: dCallOis, name: "Δ Call OI Share (pts)", mode: "lines+markers",
                        line: { color: "#6366f1", width: 2.2, dash: "dash" },
                        marker: { size: 7, color: "#6366f1", line: { width: 1.2, color: "white" } },
                        text: dCallOis.map((doVal, i) => doVal !== null && callOis[i] !== null ? `Δ Call OI: ${doVal >= 0 ? "+" : ""}${doVal.toFixed(2)} pts (Share: ${Number(callOis[i]).toFixed(1)}%)` : "Δ Call OI: N/A"),
                        hovertemplate: "%{text}<extra></extra>", xaxis: "x", yaxis: "y3"
                    },
                    {
                        x: times, y: skews, name: "Put-Call IV Skew %", mode: "lines+markers",
                        line: { color: "#e11d48", width: 2 }, marker: { size: 6 },
                        hovertemplate: "Put-Call Skew: %{y:+.2f}%<extra></extra>", xaxis: "x", yaxis: "y4"
                    },
                    {
                        x: times, y: slopes, name: "IV Term Slope %", mode: "lines+markers",
                        line: { color: "#8b5cf6", width: 1.8, dash: "dash" }, marker: { size: 6 },
                        hovertemplate: "IV Term Slope: %{y:+.2f}%<extra></extra>", xaxis: "x", yaxis: "y4"
                    },
                    {
                        x: times, y: atmIvs, name: "ATM IV %", mode: "lines+markers",
                        line: { color: "#0284c7", width: 2 }, marker: { size: 6 },
                        hovertemplate: "ATM IV: %{y:.2f}%<extra></extra>", xaxis: "x", yaxis: "y5"
                    }
                ];

                const flowRange = symmetricRange([dCallVols, dCallOis], 5.0, 0.30);
                const layout = {
                    template: "plotly_white",
                    height: 760,
                    hovermode: "x unified",
                    legend: { orientation: "h", yanchor: "bottom", y: 1.03, xanchor: "left", x: 0 },
                    margin: { l: 60, r: 60, t: 75, b: 50 },
                    xaxis: { title: "Sample Timestamp", anchor: "y4", showgrid: true },
                    yaxis:  { title: "Bias Score", range: [-2.4, 2.4], domain: [0.64, 1.0] },
                    yaxis2: { title: "Spot ($)", overlaying: "y", side: "right", showgrid: false },
                    yaxis3: { title: "Δ Flow (pts)", range: flowRange, zeroline: false, domain: [0.33, 0.57] },
                    yaxis4: { title: "Skew / Slope %", domain: [0.0, 0.25] },
                    yaxis5: { title: "ATM IV %", overlaying: "y4", side: "right", showgrid: false },
                    shapes: [
                        { type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0.35, y1: 2.4, fillcolor: "rgba(22, 163, 74, 0.07)", line: { width: 0 }, layer: "below" },
                        { type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: -0.35, y1: 0.35, fillcolor: "rgba(250, 204, 21, 0.28)", line: { width: 0 }, layer: "below" },
                        { type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: -2.4, y1: -0.35, fillcolor: "rgba(220, 38, 38, 0.07)", line: { width: 0 }, layer: "below" },
                        { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0.35, y1: 0.35, line: { color: "#eab308", width: 1.2, dash: "dash" } },
                        { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: -0.35, y1: -0.35, line: { color: "#eab308", width: 1.2, dash: "dash" } },
                        { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0.0, y1: 0.0, line: { color: "#94a3b8", width: 1, dash: "dot" } },
                        { type: "line", xref: "paper", yref: "y3", x0: 0, x1: 1, y0: 0.0, y1: 0.0, line: { color: "#64748b", width: 1.2, dash: "dot" } },
                        { type: "line", xref: "paper", yref: "y4", x0: 0, x1: 1, y0: 0.0, y1: 0.0, line: { color: "#94a3b8", width: 1, dash: "dot" } }
                    ],
                    annotations: [
                        { text: `${ticker} Composite Order-Flow Bias Score vs. Spot Price (Yellow Strip = -0.35 to +0.35)`, xref: "paper", yref: "paper", x: 0.5, y: 1.025, showarrow: false, font: { size: 13, color: "#1a1a1a" } },
                        { text: "Liquidity-Weighted Order Flow Shift (Δ vs. First Sample — High Sensitivity)", xref: "paper", yref: "paper", x: 0.5, y: 0.595, showarrow: false, font: { size: 13, color: "#1a1a1a" } },
                        { text: "Volatility Structure (Put-Call IV Skew %, Term Slope % & ATM IV %)", xref: "paper", yref: "paper", x: 0.5, y: 0.275, showarrow: false, font: { size: 13, color: "#1a1a1a" } }
                    ]
                };
                Plotly.newPlot(storyEl, traces, layout, { responsive: true });
            }

            const contracts = (snap && Array.isArray(snap.contracts)) ? snap.contracts : [];
            if (contracts.length === 0) {
                Plotly.purge(liqEl);
                liqEl.innerHTML = "<p style='padding:12px;'>No contract data available for liquidity map.</p>";
                Plotly.purge(ivEl);
                ivEl.innerHTML = "<p style='padding:12px;'>No reliable IV data available for smile chart.</p>";
                return;
            }

            const cleanLiqContracts = contracts.filter(
                c => !c.noBid && !c.arb && c.avgSpread !== null && c.avgSpread !== undefined && Number.isFinite(Number(c.avgSpread))
            );
            const liqTraces = ["Call", "Put"].map(optType => {
                const sub = cleanLiqContracts.filter(c => c.type === optType);
                return {
                    x: sub.map(c => Number(c.vol || 0)),
                    y: sub.map(c => Number(c.avgSpread)),
                    name: optType,
                    mode: "markers",
                    marker: {
                        size: sub.map(c => Math.max(8, Math.sqrt(Math.max(1, Number(c.oi || 1))) * 0.14)),
                        color: optType === "Call" ? "#2563eb" : "#ef4444",
                        opacity: 0.72,
                        line: { width: 1, color: "white" }
                    },
                    text: sub.map(c => `${optType} $${fmtNum(c.strike)} (${c.exp}, ${c.dte}d)<br>Vol: ${Number(c.vol || 0).toLocaleString()} | OI: ${Number(c.oi || 0).toLocaleString()}<br>Avg Spread: ${fmtNum(c.avgSpread, true)} | Last: $${fmtNum(c.last)}`),
                    hovertemplate: "%{text}<extra></extra>"
                };
            });

            Plotly.newPlot(liqEl, liqTraces, {
                template: "plotly_white",
                height: 420,
                title: { text: `${ticker} Liquidity Map (Volume vs Avg Spread %, Clean Two-Sided Quotes)`, font: { size: 15 } },
                xaxis: { title: "Volume" },
                yaxis: { title: "Avg Spread %" },
                legend: { title: { text: "Option Type" } },
                margin: { l: 60, r: 30, t: 55, b: 50 }
            }, { responsive: true });

            const validIv = contracts.filter(c => !c.noBid && !c.arb && c.iv !== null && c.iv !== undefined && Number(c.iv) > 1 && Number(c.iv) < 400);
            if (validIv.length === 0) {
                Plotly.purge(ivEl);
                ivEl.innerHTML = "<p style='padding:12px;'>No reliable IV data available for smile chart.</p>";
                return;
            }

            const exps = Array.from(new Set(validIv.map(c => c.exp))).sort();
            const palette = ["#2563eb", "#16a34a", "#f59e0b", "#8b5cf6", "#0d9488", "#e11d48", "#6366f1", "#ca8a04"];
            const ivTraces = [];
            exps.forEach((exp, idx) => {
                ["Call", "Put"].forEach(optType => {
                    const sub = validIv
                        .filter(c => c.exp === exp && c.type === optType)
                        .sort((a, b) => Number(a.strike) - Number(b.strike));
                    if (sub.length > 0) {
                        ivTraces.push({
                            x: sub.map(c => Number(c.strike)),
                            y: sub.map(c => +(Number(c.iv) / 100.0).toFixed(4)),
                            name: `${exp} (${optType})`,
                            mode: "lines+markers",
                            line: { color: palette[idx % palette.length], width: 2, dash: optType === "Put" ? "dash" : "solid" },
                            marker: { size: 6 },
                            hovertemplate: `${exp} ${optType}<br>Strike: $%{x}<br>IV: %{y:.2%}<extra></extra>`
                        });
                    }
                });
            });

            Plotly.newPlot(ivEl, ivTraces, {
                template: "plotly_white",
                height: 420,
                title: { text: `${ticker} Implied Volatility Smile by Expiration`, font: { size: 15 } },
                xaxis: { title: "Strike Price" },
                yaxis: { title: "Implied Volatility", tickformat: ".1%" },
                margin: { l: 65, r: 30, t: 55, b: 50 }
            }, { responsive: true });
        }

        function renderTickerView(ticker) {
            ACTIVE_TICKER = ticker;
            populateWatchlistControls(ticker);

            const snap = (SNAPSHOT_PAYLOAD.tickers || {})[ticker];
            const history = (STATE_PAYLOAD.market_history || {})[ticker] || [];
            const noticeEl = document.getElementById("ticker-notice");
            const warnBanner = document.getElementById("warning-banner");
            const warnList = document.getElementById("warning-list");

            if (!snap) {
                if (noticeEl) {
                    noticeEl.textContent = `No snapshot found yet for ${ticker}. It may not be in the background watchlist or hasn't finished its first clean scan.`;
                    noticeEl.classList.add("visible");
                }
                document.getElementById("dashboard-title").textContent = `${ticker} Options Scanner Dashboard`;
                document.getElementById("dashboard-meta").textContent = "Awaiting snapshot for this symbol.";
                document.getElementById("contracts-tbody").innerHTML = "";
                document.getElementById("filter-count").textContent = "0 matching contracts";
                document.getElementById("stat-matching-count").textContent = "0";
                renderTickerCharts(ticker, null, []);
                return;
            }

            if (noticeEl) {
                noticeEl.classList.remove("visible");
                noticeEl.textContent = "";
            }

            const spotStr = (snap.spot !== null && snap.spot !== undefined) ? `$${Number(snap.spot).toFixed(2)}` : "N/A";
            const refreshNum = snap.refresh_count ?? snap.sample_rounds ?? (history.length || 1);
            const intervalMin = snap.refresh_interval_minutes ?? SNAPSHOT_PAYLOAD.refresh_interval_minutes ?? 15;
            document.title = `${ticker} Options Scan Dashboard`;
            document.getElementById("dashboard-title").textContent = `${ticker} Options Scanner Dashboard`;
            document.getElementById("dashboard-meta").innerHTML =
                `Underlying Price: <strong>${spotStr}</strong> | Refresh: <strong>#${refreshNum}</strong> | Background Interval: <strong>Every ${intervalMin} min (weekdays)</strong> | Latest Snapshot: <strong>${escapeHtml(snap.fetched_at || "-")}</strong> (${history.length} in trend)`;

            const warnings = Array.isArray(snap.warnings) ? snap.warnings : [];
            if (warnings.length > 0 && warnBanner && warnList) {
                warnList.innerHTML = warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");
                warnBanner.classList.add("visible");
            } else if (warnBanner) {
                warnBanner.classList.remove("visible");
            }

            const ins = snap.insights || {};
            const bias = ins.map_bias || "Neutral";
            const score = Number(ins.map_bias_score || 0);
            const biasBox = document.getElementById("bias-box");
            if (biasBox) {
                biasBox.className = "stat-box " + (bias === "Bullish" ? "bias-bullish" : (bias === "Bearish" ? "bias-bearish" : "bias-neutral"));
            }
            document.getElementById("stat-bias").textContent = bias;
            document.getElementById("stat-bias-score").textContent = `Score: ${score >= 0 ? "+" : ""}${score.toFixed(2)}`;
            document.getElementById("stat-hv").textContent = ins.current_30d_hv !== null && ins.current_30d_hv !== undefined ? `${Number(ins.current_30d_hv).toFixed(1)}%` : "-";
            document.getElementById("stat-iv").textContent = ins.atm_iv !== null && ins.atm_iv !== undefined ? `${Number(ins.atm_iv).toFixed(1)}%` : "-";
            document.getElementById("stat-ratio").textContent = ins.iv_hv_ratio !== null && ins.iv_hv_ratio !== undefined ? Number(ins.iv_hv_ratio).toFixed(2) : "-";
            document.getElementById("stat-pricing-bias").textContent = ins.pricing_bias || "N/A";
            document.getElementById("stat-hv-rank").textContent = ins.hv_rank !== null && ins.hv_rank !== undefined ? `${Number(ins.hv_rank).toFixed(1)}%` : "-";
            document.getElementById("bias-note").innerHTML = `<strong>Latest Map Interpretation (${escapeHtml(bias)}):</strong> ${escapeHtml(ins.map_bias_summary || "")}`;

            const counts = snap.counts || {};
            document.getElementById("stat-total-contracts").textContent = counts.total ?? 0;
            document.getElementById("stat-calls").textContent = counts.calls ?? 0;
            document.getElementById("stat-puts").textContent = counts.puts ?? 0;

            document.getElementById("story-banner").innerHTML = `<strong>Order Flow Story:</strong> ${summarizeNarrative(history)}`;

            const contracts = Array.isArray(snap.contracts) ? snap.contracts : [];
            const tbody = document.getElementById("contracts-tbody");
            tbody.innerHTML = contracts.map(r => {
                const isFlagged = Boolean(r.noBid || r.stale || r.arb);
                const rowCls = isFlagged ? ' class="flagged-row"' : "";
                const spreadTitle = `Averaged across ${r.samples || 0} clean sample(s)`;
                const calcCount = Math.max(0, Math.min(80, Number(r.greekCalcCount || 0)));
                const calcRatio = calcCount / 80.0;
                const greekAlpha = calcCount > 0 ? (0.08 + calcRatio * 0.42).toFixed(3) : "0";
                const greekStyle = calcCount > 0
                    ? ` style="background-color: rgba(251, 146, 60, ${greekAlpha}); color: #431407; font-weight: ${calcCount >= 40 ? "600" : "400"};"`
                    : "";
                const greekSrcLabel = r.greekComputed ? "Calculated fallback this run" : "From Yahoo Finance";
                const greekTitle = `${greekSrcLabel} · Calculated count: ${calcCount}/80`;
                return `<tr${rowCls}>
                    <td>${escapeHtml(r.type || "-")}</td>
                    <td>${escapeHtml(r.exp || "-")}</td>
                    <td>${r.dte ?? 0}</td>
                    <td>${fmtNum(r.strike)}</td>
                    <td>${fmtNum(r.moneyness, true)}</td>
                    <td>${fmtNum(r.last)}</td>
                    <td>${fmtNum(r.bid)}</td>
                    <td>${fmtNum(r.ask)}</td>
                    <td>${fmtNum(r.mid)}</td>
                    <td title="${escapeHtml(spreadTitle)}">${fmtNum(r.avgSpread, true)}</td>
                    <td>${r.vol ?? 0}</td>
                    <td>${r.oi ?? 0}</td>
                    <td>${fmtNum(r.volOi)}</td>
                    <td>${fmtNum(r.iv, true)}</td>
                    <td${greekStyle} title="${escapeHtml(greekTitle)}">${fmtNum(r.delta, false, 3)}</td>
                    <td${greekStyle} title="${escapeHtml(greekTitle)}">${fmtNum(r.theta, false, 3)}</td>
                    <td>${r.noBid ? "YES" : "No"}</td>
                    <td>${r.stale ? "YES" : "No"}</td>
                </tr>`;
            }).join("");

            loadFilterState();
            filterTable();
            renderTickerCharts(ticker, snap, history);
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
            const t = (ACTIVE_TICKER || document.getElementById("watchlist-input")?.value || "DEFAULT").toUpperCase();
            return "options_filter_state_" + t;
        }

        function saveFilterState() {
            const state = {
                type: document.getElementById("filter-type").value,
                dte: document.getElementById("filter-dte-num").value,
                maxDte: document.getElementById("filter-max-dte-num").value,
                price: document.getElementById("filter-price-num").value,
                vol: document.getElementById("filter-vol-num").value,
                spread: document.getElementById("filter-spread-num").value,
                oi: document.getElementById("filter-oi-num").value
            };
            try {
                localStorage.setItem(getTickerStorageKey(), JSON.stringify(state));
            } catch (e) {}
        }

        function loadFilterState() {
            let saved = null;
            try {
                saved = localStorage.getItem(getTickerStorageKey());
            } catch (e) {}
            if (!saved) return;
            try {
                const state = JSON.parse(saved);
                if (state.type !== undefined) document.getElementById("filter-type").value = state.type;
                const pairs = [
                    ["filter-dte-slider", "filter-dte-num", state.dte],
                    ["filter-max-dte-slider", "filter-max-dte-num", state.maxDte],
                    ["filter-price-slider", "filter-price-num", state.price],
                    ["filter-vol-slider", "filter-vol-num", state.vol],
                    ["filter-spread-slider", "filter-spread-num", state.spread],
                    ["filter-oi-slider", "filter-oi-num", state.oi]
                ];
                pairs.forEach(([sId, nId, val]) => {
                    if (val !== undefined && val !== "") {
                        const slider = document.getElementById(sId);
                        const num = document.getElementById(nId);
                        if (slider) slider.value = val;
                        if (num) num.value = val;
                    }
                });
            } catch (e) {}
        }

        function syncInputs(sliderId, numId) {
            const slider = document.getElementById(sliderId);
            const num = document.getElementById(numId);
            if (!slider || !num) return;
            slider.addEventListener("input", () => {
                num.value = slider.value;
                saveFilterState();
                filterTable();
            });
            num.addEventListener("input", () => {
                slider.value = num.value;
                saveFilterState();
                filterTable();
            });
        }

        function filterTable() {
            const typeVal = document.getElementById("filter-type").value;
            const minDteRaw = parseFloat(document.getElementById("filter-dte-num").value);
            const minDte = isNaN(minDteRaw) ? 0 : minDteRaw;
            const maxDteRaw = parseFloat(document.getElementById("filter-max-dte-num").value);
            const maxDte = isNaN(maxDteRaw) ? Infinity : maxDteRaw;
            const maxPriceRaw = parseFloat(document.getElementById("filter-price-num").value);
            const maxPrice = isNaN(maxPriceRaw) ? Infinity : maxPriceRaw;
            const minVolRaw = parseFloat(document.getElementById("filter-vol-num").value);
            const minVol = isNaN(minVolRaw) ? 0 : minVolRaw;
            const maxSpreadRaw = parseFloat(document.getElementById("filter-spread-num").value);
            const maxSpread = isNaN(maxSpreadRaw) ? Infinity : maxSpreadRaw;
            const minOiRaw = parseFloat(document.getElementById("filter-oi-num").value);
            const minOi = isNaN(minOiRaw) ? 0 : minOiRaw;

            const tbody = document.querySelector("#scan-results-table tbody");
            if (!tbody) return;
            const rows = tbody.rows;
            let visibleCount = 0;

            for (let r of rows) {
                const type = r.cells[0].innerText.trim();
                const dte = parseFloat(r.cells[2].innerText) || 0;
                const last = parseFloat(r.cells[5].innerText) || 0;
                const avgSpreadText = r.cells[9].innerText.trim();
                const avgSpread = avgSpreadText === "-" ? Infinity : parseFloat(avgSpreadText);
                const vol = parseFloat(r.cells[10].innerText) || 0;
                const oi = parseFloat(r.cells[11].innerText) || 0;

                let match = true;
                if (typeVal !== "ALL" && type !== typeVal) match = false;
                if (dte < minDte) match = false;
                if (dte > maxDte) match = false;
                if (last > maxPrice) match = false;
                if (vol < minVol) match = false;
                if (avgSpread > maxSpread) match = false;
                if (oi < minOi) match = false;

                r.style.display = match ? "" : "none";
                if (match) visibleCount++;
            }
            const countEl = document.getElementById("filter-count");
            if (countEl) countEl.innerText = visibleCount + " matching contracts";
            const statMatchEl = document.getElementById("stat-matching-count");
            if (statMatchEl) statMatchEl.innerText = visibleCount;
        }

        function resetFilters() {
            try {
                localStorage.removeItem(getTickerStorageKey());
            } catch (e) {}
            document.getElementById("filter-type").value = "ALL";
            const resets = [
                ["filter-dte-slider", "filter-dte-num", 0],
                ["filter-max-dte-slider", "filter-max-dte-num", 365],
                ["filter-price-slider", "filter-price-num", 999],
                ["filter-vol-slider", "filter-vol-num", 0],
                ["filter-spread-slider", "filter-spread-num", 100],
                ["filter-oi-slider", "filter-oi-num", 0]
            ];
            resets.forEach(([s, n, v]) => {
                const slider = document.getElementById(s);
                const num = document.getElementById(n);
                if (slider) slider.value = v;
                if (num) num.value = v;
            });
            filterTable();
        }

        const AUTO_POLL_INTERVAL_MS = 120000; // Check every 2 minutes for new GitHub Actions commits
        let LAST_SEEN_UPDATED_AT = null;
        let POLL_IN_FLIGHT = false;

        async function fetchLatestDashboardData({ isInitial = false } = {}) {
            if (POLL_IN_FLIGHT) return;
            POLL_IN_FLIGHT = true;
            const statusEl = document.getElementById("batch-status");
            try {
                const [snapRes, stateRes] = await Promise.all([
                    fetch("./data/snapshots.json?v=" + Date.now(), { cache: "no-store" }),
                    fetch("./data/state.json?v=" + Date.now(), { cache: "no-store" })
                ]);
                let nextSnap = SNAPSHOT_PAYLOAD;
                let nextState = STATE_PAYLOAD;
                if (snapRes.ok) {
                    nextSnap = await snapRes.json();
                }
                if (stateRes.ok) {
                    nextState = await stateRes.json();
                }

                const nextUpdatedAt = nextSnap.updated_at || null;
                const hasNewBatch = Boolean(nextUpdatedAt && nextUpdatedAt !== LAST_SEEN_UPDATED_AT);

                SNAPSHOT_PAYLOAD = nextSnap;
                STATE_PAYLOAD = nextState;

                const wl = getWatchlist();
                if (statusEl) {
                    const intervalMin = nextSnap.refresh_interval_minutes ?? 15;
                    const checkedSuffix = !isInitial ? " (auto-checked)" : "";
                    statusEl.textContent = nextUpdatedAt
                        ? `Every ${intervalMin} min · Last batch: ${nextUpdatedAt}${checkedSuffix}`
                        : `Every ${intervalMin} min · ${wl.length} tickers configured`;
                }

                if (isInitial) {
                    LAST_SEEN_UPDATED_AT = nextUpdatedAt;
                    const inputVal = (document.getElementById("watchlist-input")?.value || "").trim().toUpperCase();
                    const initialTicker = inputVal || wl[0] || "XLE";
                    renderTickerView(initialTicker);
                } else if (hasNewBatch) {
                    LAST_SEEN_UPDATED_AT = nextUpdatedAt;
                    const active = ACTIVE_TICKER || wl[0] || "XLE";
                    renderTickerView(active);
                }
            } catch (err) {
                console.error("Failed to fetch snapshot JSON files:", err);
            } finally {
                POLL_IN_FLIGHT = false;
            }
        }

        async function loadSnapshotsAndInit() {
            syncInputs("filter-dte-slider", "filter-dte-num");
            syncInputs("filter-max-dte-slider", "filter-max-dte-num");
            syncInputs("filter-price-slider", "filter-price-num");
            syncInputs("filter-vol-slider", "filter-vol-num");
            syncInputs("filter-spread-slider", "filter-spread-num");
            syncInputs("filter-oi-slider", "filter-oi-num");

            await fetchLatestDashboardData({ isInitial: true });

            setInterval(() => {
                if (document.visibilityState === "visible") {
                    fetchLatestDashboardData({ isInitial: false });
                }
            }, AUTO_POLL_INTERVAL_MS);

            document.addEventListener("visibilitychange", () => {
                if (document.visibilityState === "visible") {
                    fetchLatestDashboardData({ isInitial: false });
                }
            });
        }

        document.addEventListener("DOMContentLoaded", loadSnapshotsAndInit);
    </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(f"[Index] Wrote dashboard landing page to {out_dir / 'index.html'}.")


def main():
    config = get_scan_config()
    global GLOBAL_STATE
    GLOBAL_STATE = DashboardState(config)

    for ticker in config.tickers:
        print(f"\nProcessing {ticker}...")
        refresh_ticker_with_retry(ticker, config, max_attempts_before_log=3)

    GLOBAL_STATE.save()
    build_index_page(Path(config.out_dir), config.tickers)
    print("\n50-ticker background snapshot & index generation complete.")


if __name__ == "__main__":
    main()
