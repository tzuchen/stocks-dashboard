from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from cachetools import TTLCache
import yfinance as yf
import pandas as pd
import math

app = FastAPI(title="Stocks & Options Board")
templates = Jinja2Templates(directory="templates")

# ------- 基本設定 -------
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "AMD", "AMZN", "PLTR"]
QUOTES_TTL_SEC = 20
OPTIONS_TTL_SEC = 45

quote_cache: TTLCache = TTLCache(maxsize=512, ttl=QUOTES_TTL_SEC)
options_cache: TTLCache = TTLCache(maxsize=256, ttl=OPTIONS_TTL_SEC)


# ------- 小工具 -------
def _now_iso() -> str:
    # 顯示為 UTC ISO，前端直接呈現字串
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _to_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


# ------- 股票報價 -------
def get_quote_one(symbol: str) -> Dict[str, Any]:
    key = f"q:{symbol}"
    if key in quote_cache:
        return quote_cache[key]

    info = {
        "symbol": symbol,
        "price": None,
        "change": None,
        "pct": None,
        "currency": "",
        "time": "",
    }
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "last_price", None) is not None:
            last = _to_float(fi.last_price)
            prev = _to_float(getattr(fi, "previous_close", None))
            info["price"] = last
            if last is not None and prev is not None:
                info["change"] = round(last - prev, 2)
                info["pct"] = round((last - prev) / prev * 100, 2) if prev else None
            info["currency"] = getattr(fi, "currency", "") or ""
        else:
            # 備援：抓 1d/1m
            df = t.history(period="1d", interval="1m")
            if not df.empty:
                last = _to_float(df["Close"].iloc[-1])
                info["price"] = last
                hist5 = t.history(period="5d")
                prev_close = _to_float(hist5["Close"].iloc[-2]) if len(hist5) >= 2 else None
                if last is not None and prev_close is not None:
                    info["change"] = round(last - prev_close, 2)
                    info["pct"] = round((last - prev_close) / prev_close * 100, 2) if prev_close else None

        info["time"] = _now_iso()
    except Exception:
        # 保持空值，不讓整頁炸掉
        pass

    quote_cache[key] = info
    return info


# ------- 選擇權（取鄰近 ATM 的小片段） -------
def _pick_expiry(t: yf.Ticker, expiry: Optional[str]) -> Optional[str]:
    try:
        exps = t.options or []
    except Exception:
        exps = []
    if not exps:
        return None
    if not expiry:
        return exps[0]  # 最近到期
    return expiry if expiry in exps else None


def get_options_slice(symbol: str, expiry: Optional[str]) -> Optional[Dict[str, Any]]:
    key = f"opt:{symbol}:{expiry or ''}"
    if key in options_cache:
        return options_cache[key]

    try:
        t = yf.Ticker(symbol)
        picked = _pick_expiry(t, expiry)
        if not picked:
            return None

        # Spot
        spot = None
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "last_price", None) is not None:
            spot = _to_float(fi.last_price)
        if spot is None:
            df = t.history(period="1d", interval="1m")
            if not df.empty:
                spot = _to_float(df["Close"].iloc[-1])

        chain = t.option_chain(picked)  # 回傳 calls / puts DataFrame
        calls_df: pd.DataFrame = getattr(chain, "calls", pd.DataFrame())
        puts_df: pd.DataFrame = getattr(chain, "puts", pd.DataFrame())

        def _rows_near_atm(df: pd.DataFrame, n_each_side: int = 6):
            if df is None or df.empty or spot is None:
                return []
            # 取最接近的行，左右各 n_each_side
            df2 = df.copy()
            df2["dist"] = (df2["strike"] - spot).abs()
            df2 = df2.sort_values("dist")
            keep = df2.head(max(1, n_each_side * 2 + 1)).sort_values("strike")

            out = []
            for _, r in keep.iterrows():
                bid = _to_float(r.get("bid"))
                ask = _to_float(r.get("ask"))
                last = _to_float(r.get("lastPrice"))  # Yahoo 提供
                mid = None
                mid_from = None
                if bid is not None and ask is not None:
                    mid = round((bid + ask) / 2, 2)
                    mid_from = "ba"
                elif last is not None:
                    mid = round(last, 2)
                    mid_from = "last"

                out.append({
                    "strike": _to_float(r.get("strike")),
                    "bid": None if bid is None else round(bid, 2),
                    "ask": None if ask is None else round(ask, 2),
                    "last": None if last is None else round(last, 2),
                    "mid": mid,
                    "mid_from": mid_from,  # 'ba' = bid/ask, 'last' = 由 last 近似
                    "openInterest": int(r.get("openInterest") or 0),
                    "volume": int(r.get("volume") or 0),
                })
            return out

        calls = _rows_near_atm(calls_df)
        puts = _rows_near_atm(puts_df)

        block = {
            "symbol": symbol.upper(),
            "expiry": picked,
            "time": _now_iso(),
            "spot": None if spot is None else float(round(spot, 2)),
            "calls": calls,
            "puts": puts,
        }
        options_cache[key] = block
        return block
    except Exception:
        return None


# ------- 健康檢查 -------
@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "now": _now_iso()})


# ------- 首頁（固定渲染 index.html） -------
@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    symbols: Optional[str] = Query(None, description="逗號分隔的標的列表，如 TSLA,AAPL,SPY"),
    opt: Optional[str] = Query(None, description="要查看選擇權的標的，如 TSLA"),
    expiry: Optional[str] = Query(None, description="到期日 YYYY-MM-DD，留空自動挑最近"),
):
    # symbols 處理
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        syms = DEFAULT_TICKERS

    # 報價
    quotes = [get_quote_one(s) for s in syms]

    # 選擇權
    opt_block = get_options_slice(opt, expiry) if opt else None

    ctx = {
        "request": request,
        "symbols": syms,
        "quotes": quotes,
        "opt": opt_block,
        "expiry": expiry or "",
        "now": _now_iso(),
    }
    return templates.TemplateResponse("index.html", ctx)
