from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from cachetools import TTLCache
import math

app = FastAPI(title="Stocks & Options Board")
templates = Jinja2Templates(directory="templates")

# 回應健康
@app.get("/health")
def health():
    return {"status": "ok"}

# 預設追蹤清單（可改）
DEFAULT_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]

# 簡單快取（避免 yfinance 過度請求）
quote_cache = TTLCache(maxsize=256, ttl=15)      # 15 秒
opt_cache   = TTLCache(maxsize=256, ttl=60)      # 60 秒

def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def get_quote_one(symbol: str):
    key = symbol.upper()
    if key in quote_cache:
        return quote_cache[key]

    tk = yf.Ticker(key)
    # yfinance 最穩定是用 .fast_info 及 .history() 混搭
    price = None
    currency = None
    try:
        fi = tk.fast_info
        price = float(fi["last_price"]) if fi.get("last_price") is not None else None
        currency = fi.get("currency")
    except Exception:
        pass

    if price is None:
        try:
            h = tk.history(period="1d", interval="1m")
            if not h.empty:
                price = float(h["Close"].iloc[-1])
        except Exception:
            pass

    prev_close = None
    try:
        info = tk.info or {}
        prev_close = info.get("previousClose")
    except Exception:
        pass

    change = pct = None
    if price is not None and prev_close:
        change = price - prev_close
        if prev_close != 0:
            pct = 100.0 * change / prev_close

    data = {
        "symbol": key,
        "price": price,
        "change": change,
        "pct": pct,
        "currency": currency or "",
        "time": _now_iso(),
    }
    quote_cache[key] = data
    return data

def pick_expiry(symbol: str, want: Optional[str]) -> Optional[str]:
    try:
        exps = yf.Ticker(symbol).options or []
    except Exception:
        exps = []
    if not exps:
        return None
    if want and want in exps:
        return want
    # 選最近一個
    return exps[0]

def get_options_slice(symbol: str, expiry: Optional[str], strikes_around: int = 2):
    """
    回傳鄰近 ATM 的選擇權（call/put 各 ±N 檔）。只做展示用途。
    """
    cache_key = (symbol.upper(), expiry or "")
    if cache_key in opt_cache:
        return opt_cache[cache_key]

    q = get_quote_one(symbol)
    spot = q["price"]
    if spot is None:
        return {"symbol": symbol.upper(), "expiry": None, "chains": []}

    exp = pick_expiry(symbol, expiry)
    if not exp:
        return {"symbol": symbol.upper(), "expiry": None, "chains": []}

    try:
        chain = yf.Ticker(symbol).option_chain(exp)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
    except Exception:
        return {"symbol": symbol.upper(), "expiry": exp, "chains": []}

    # 計算中價 mid
    def add_mid(df):
        if "bid" in df and "ask" in df:
            df["mid"] = (pd.to_numeric(df["bid"], errors="coerce") +
                         pd.to_numeric(df["ask"], errors="coerce")) / 2
        else:
            df["mid"] = pd.NA
        return df

    calls = add_mid(calls)
    puts  = add_mid(puts)

    # 找最接近 spot 的行權價
    try:
        calls["diff"] = (calls["strike"] - spot).abs()
        puts["diff"]  = (puts["strike"]  - spot).abs()
        # 用 calls 的排序來當作 strike 序列（兩邊會相近）
        calls_sorted = calls.sort_values("diff")
        center_strike = float(calls_sorted.iloc[0]["strike"])
    except Exception:
        return {"symbol": symbol.upper(), "expiry": exp, "chains": []}

    # 取相近幾檔（用 strike 排序）
    def slice_side(df):
        df2 = df.sort_values("strike")
        # 找到中心位置
        idx = int((df2["strike"] - center_strike).abs().idxmin())
        pos = df2.index.get_loc(idx)
        lo = max(0, pos - strikes_around)
        hi = min(len(df2), pos + strikes_around + 1)
        return df2.iloc[lo:hi][["contractSymbol", "strike", "bid", "ask", "mid", "lastTradeDate", "openInterest", "volume"]]

    calls_sel = slice_side(calls)
    puts_sel  = slice_side(puts)

    out = {
        "symbol": symbol.upper(),
        "expiry": exp,
        "spot": spot,
        "calls": calls_sel.to_dict(orient="records"),
        "puts":  puts_sel.to_dict(orient="records"),
        "time":  _now_iso()
    }
    opt_cache[cache_key] = out
    return out

@app.get("/api/quotes", response_class=JSONResponse)
def api_quotes(symbols: Optional[str] = Query(None, description="逗號分隔，例如 TSLA,AAPL,SPY")):
    syms = [s.strip().upper() for s in (symbols.split(",") if symbols else DEFAULT_TICKERS)]
    data = [get_quote_one(s) for s in syms]
    return {"symbols": syms, "data": data, "time": _now_iso()}

@app.get("/api/options", response_class=JSONResponse)
def api_options(symbol: str, expiry: Optional[str] = None, around: int = 2):
    return get_options_slice(symbol, expiry, strikes_around=max(0, int(around)))

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request,
              symbols: Optional[str] = None,
              opt: Optional[str] = None,
              expiry: Optional[str] = None):
    """
    /?symbols=TSLA,AAPL,SPY&opt=TSLA&expiry=YYYY-MM-DD
    opt: 要顯示選擇權的標的
    """
    syms = [s.strip().upper() for s in (symbols.split(",") if symbols else DEFAULT_TICKERS)]
    quotes = [get_quote_one(s) for s in syms]

    opt_block = None
    if opt:
        opt_block = get_options_slice(opt, expiry)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "symbols": syms,
            "quotes": quotes,
            "opt": opt_block,
            "expiry": expiry or "",
            "now": _now_iso(),
        },
    )

