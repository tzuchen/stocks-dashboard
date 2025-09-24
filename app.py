from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from cachetools import TTLCache
import yfinance as yf
import pandas as pd
import math
import requests

app = FastAPI(title="Stocks & Options Board")
templates = Jinja2Templates(directory="templates")

# ------- 基本設定 -------
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "AMD", "AMZN", "PLTR"]
QUOTES_TTL_SEC = 20
OPTIONS_TTL_SEC = 45

quote_cache: TTLCache = TTLCache(maxsize=512, ttl=QUOTES_TTL_SEC)
options_cache: TTLCache = TTLCache(maxsize=256, ttl=OPTIONS_TTL_SEC)
yahoo_quote_cache: TTLCache = TTLCache(maxsize=2048, ttl=60)  # OCC 單筆補價快取

# ------- 小工具 -------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def _to_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _none_if_zero(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        return None if float(v) == 0.0 else float(v)
    except Exception:
        return None

# ------- Yahoo quote API：用 OCC 符號補價 -------
def occ_symbol(underlying: str, expiry: str, right: str, strike: float) -> str:
    # expiry: "YYYY-MM-DD"; right: "C" or "P"
    y, m, d = expiry.split("-")
    yy = y[2:]
    strike_int = int(round(strike * 1000))
    return f"{underlying.upper()}{yy}{m}{d}{right.upper()}{strike_int:08d}"

def yahoo_quote_batch(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    # 使用快取避免重複打
    need = [s for s in symbols if s not in yahoo_quote_cache]
    if need:
        url = "https://query2.finance.yahoo.com/v7/finance/quote"
        # Yahoo 對長 UA 比較友善
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"}
        # 分批（一次 ~50 筆較安全）
        for i in range(0, len(need), 50):
            batch = need[i:i+50]
            try:
                r = requests.get(url, params={"symbols": ",".join(batch)}, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json().get("quoteResponse", {}).get("result", [])
                for q in data:
                    sym = q.get("symbol")
                    last = _none_if_zero(_to_float(q.get("regularMarketPrice")))
                    bid = _none_if_zero(_to_float(q.get("bid")))
                    ask = _none_if_zero(_to_float(q.get("ask")))
                    tsec = q.get("regularMarketTime", 0) or 0
                    tstr = datetime.utcfromtimestamp(int(tsec)).strftime("%Y-%m-%d %H:%M:%SZ") if tsec else _now_iso()
                    yahoo_quote_cache[sym] = {"last": last, "bid": bid, "ask": ask, "time": tstr}
            except Exception:
                # 若整批失敗，不中斷主流程；缺的 symbol 等於查不到
                for s in batch:
                    yahoo_quote_cache[s] = {"last": None, "bid": None, "ask": None, "time": _now_iso()}
    return {s: yahoo_quote_cache.get(s, {"last": None, "bid": None, "ask": None, "time": _now_iso()}) for s in symbols}

# ------- 股票報價 -------
def get_quote_one(symbol: str) -> Dict[str, Any]:
    key = f"q:{symbol}"
    if key in quote_cache:
        return quote_cache[key]

    info = {"symbol": symbol, "price": None, "change": None, "pct": None, "currency": "", "time": ""}
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
        pass

    quote_cache[key] = info
    return info

# ------- 選擇權（鄰近 ATM 小片段 + Yahoo quote API 補價） -------
def _pick_expiry(t: yf.Ticker, expiry: Optional[str]) -> Optional[str]:
    try:
        exps = t.options or []
    except Exception:
        exps = []
    if not exps:
        return None
    if not expiry:
        return exps[0]
    return expiry if expiry in exps else None

def _rows_near_atm(df: pd.DataFrame, spot: Optional[float], n_each_side: int = 6) -> pd.DataFrame:
    if df is None or df.empty or spot is None:
        return pd.DataFrame()
    df2 = df.copy()
    df2["dist"] = (df2["strike"] - spot).abs()
    df2 = df2.sort_values("dist")
    keep = df2.head(max(1, n_each_side * 2 + 1)).sort_values("strike")
    return keep

def get_options_slice(symbol: str, expiry: Optional[str]) -> Optional[Dict[str, Any]]:
    # 版本化 cache key，避免舊格式殘留
    key = f"opt3:{symbol}:{expiry or ''}"
    if key in options_cache:
        return options_cache[key]

    try:
        t = yf.Ticker(symbol)
        picked = _pick_expiry(t, expiry)
        if not picked:
            return None

        # Spot（先 fast_info，無則 1m close）
        spot = None
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "last_price", None) is not None:
            spot = _to_float(fi.last_price)
        if spot is None:
            df = t.history(period="1d", interval="1m")
            if not df.empty:
                spot = _to_float(df["Close"].iloc[-1])

        chain = t.option_chain(picked)
        calls_df: pd.DataFrame = getattr(chain, "calls", pd.DataFrame())
        puts_df: pd.DataFrame = getattr(chain, "puts", pd.DataFrame())

        calls_keep = _rows_near_atm(calls_df, spot)
        puts_keep = _rows_near_atm(puts_df, spot)

        # 先從 yfinance 取到的欄位做初值
        def seed_rows(keep: pd.DataFrame, right: str) -> Tuple[List[Dict[str, Any]], List[str]]:
            rows: List[Dict[str, Any]] = []
            occs: List[str] = []
            for _, r in keep.iterrows():
                strike = _to_float(r.get("strike"))
                bid = _none_if_zero(_to_float(r.get("bid")))
                ask = _none_if_zero(_to_float(r.get("ask")))
                last = _none_if_zero(_to_float(r.get("lastPrice")))
                mid = None
                mid_from = None
                if bid is not None and ask is not None:
                    mid = round((bid + ask) / 2, 2)
                    mid_from = "ba"
                elif last is not None:
                    mid = round(last, 2)
                    mid_from = "last"
                row = {
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "mid": mid,
                    "mid_from": mid_from,
                    "openInterest": int(r.get("openInterest") or 0),
                    "volume": int(r.get("volume") or 0),
                }
                # 生成 OCC 符號（缺價時用來補價）
                if strike is not None:
                    occs.append(occ_symbol(symbol, picked, right, strike))
                rows.append(row)
            return rows, occs

        call_rows, call_occs = seed_rows(calls_keep, "C")
        put_rows,  put_occs  = seed_rows(puts_keep, "P")

        # 需要補價的 OCC（bid/ask/last 有缺或為 None／0）
        need_call = [s for s, row in zip(call_occs, call_rows) if row["bid"] is None or row["ask"] is None or row["last"] is None]
        need_put  = [s for s, row in zip(put_occs,  put_rows)  if row["bid"] is None or row["ask"] is None or row["last"] is None]
        batch_syms = list(dict.fromkeys(need_call + need_put))  # 去重保持順序

        if batch_syms:
            quotes_map = yahoo_quote_batch(batch_syms)
            # 寫回 rows
            def apply_fill(rows: List[Dict[str, Any]], occs: List[str]):
                for idx, occ in enumerate(occs):
                    q = quotes_map.get(occ)
                    if not q:
                        continue
                    # 只有在缺的情況下才補
                    if rows[idx]["bid"] is None: rows[idx]["bid"] = _none_if_zero(_to_float(q.get("bid")))
                    if rows[idx]["ask"] is None: rows[idx]["ask"] = _none_if_zero(_to_float(q.get("ask")))
                    if rows[idx]["last"] is None: rows[idx]["last"] = _none_if_zero(_to_float(q.get("last")))
                    # 重新計算 mid
                    bid, ask, last = rows[idx]["bid"], rows[idx]["ask"], rows[idx]["last"]
                    if bid is not None and ask is not None:
                        rows[idx]["mid"] = round((bid + ask) / 2, 2)
                        rows[idx]["mid_from"] = "ba"
                    elif last is not None:
                        rows[idx]["mid"] = round(last, 2)
                        rows[idx]["mid_from"] = "last"

            apply_fill(call_rows, call_occs)
            apply_fill(put_rows,  put_occs)

        block = {
            "symbol": symbol.upper(),
            "expiry": picked,
            "time": _now_iso(),
            "spot": None if spot is None else float(round(spot, 2)),
            "calls": call_rows,
            "puts": put_rows,
        }
        options_cache[key] = block
        return block
    except Exception:
        return None

# ------- 健康檢查 -------
@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "now": _now_iso()})

# ------- 首頁 -------
@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    symbols: Optional[str] = Query(None, description="逗號分隔，如 TSLA,AAPL,SPY"),
    opt: Optional[str] = Query(None, description="選擇權標的，如 TSLA"),
    expiry: Optional[str] = Query(None, description="到期日 YYYY-MM-DD；留空自動挑最近"),
):
    syms = [s.strip().upper() for s in symbols.split(",")] if symbols else DEFAULT_TICKERS
    quotes = [get_quote_one(s) for s in syms]
    opt_block = get_options_slice(opt, expiry) if opt else None
    ctx = {"request": request, "symbols": syms, "quotes": quotes, "opt": opt_block, "expiry": expiry or "", "now": _now_iso()}
    return templates.TemplateResponse("index.html", ctx)
