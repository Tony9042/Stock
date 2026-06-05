"""
data_fetcher.py
即時報價：TWSE MIS API（先建 session 取 cookie）→ yfinance 備援
歷史K線：yfinance（慢，只在「完整重整」時才抓）
"""
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import time

WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")

DEFAULT_WATCHLIST = {
    "stocks": [
        {"code": "2330",  "name": "台積電",           "market": "tse", "is_etf": False, "theme": ["半導體", "AI"]},
        {"code": "0050",  "name": "元大台灣50",        "market": "tse", "is_etf": True,  "theme": ["ETF"]},
        {"code": "00713", "name": "元大高息低波",      "market": "tse", "is_etf": True,  "theme": ["ETF", "高息"]},
        {"code": "00919", "name": "群益台灣精選高息",  "market": "tse", "is_etf": True,  "theme": ["ETF", "高息"]},
        {"code": "00981A","name": "主動統一台股增長",  "market": "tse", "is_etf": True,  "theme": ["ETF"]},
        {"code": "2308",  "name": "台達電",            "market": "tse", "is_etf": False, "theme": ["電力", "AI"]},
        {"code": "2382",  "name": "廣達",              "market": "tse", "is_etf": False, "theme": ["AI", "伺服器"]},
        {"code": "3017",  "name": "奇鋐",              "market": "otc", "is_etf": False, "theme": ["散熱", "AI"]},
        {"code": "3037",  "name": "欣興",              "market": "otc", "is_etf": False, "theme": ["PCB", "半導體"]},
        {"code": "6669",  "name": "緯穎",              "market": "tse", "is_etf": False, "theme": ["AI", "伺服器", "資料中心"]},
        {"code": "6770",  "name": "力積電",            "market": "tse", "is_etf": False, "theme": ["半導體", "晶圓代工"]},
        {"code": "6116",  "name": "彩晶",              "market": "otc", "is_etf": False, "theme": ["面板"]},
    ]
}

# ── watchlist I/O ──────────────────────────────────────────────
def load_watchlist() -> dict:
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(wl: dict) -> None:
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)


# ── TWSE MIS Session（帶 cookie，提升 API 成功率）──────────────
_twse_session: requests.Session | None = None
_session_init_time: datetime | None = None
_SESSION_TTL = 1800   # session 30 分鐘後重建

def _get_twse_session() -> requests.Session:
    global _twse_session, _session_init_time
    now = datetime.now()
    need_init = (
        _twse_session is None or
        _session_init_time is None or
        (now - _session_init_time).seconds > _SESSION_TTL
    )
    if need_init:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        })
        # 先訪問首頁，讓伺服器設定 session cookie
        try:
            sess.get(
                "https://mis.twse.com.tw/stock/index.jsp",
                timeout=8,
                headers={"Referer": "https://www.twse.com.tw/"},
            )
        except Exception:
            pass
        _twse_session = sess
        _session_init_time = now
    return _twse_session


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


# ── 即時報價（TWSE MIS API）────────────────────────────────────
def fetch_twse_realtime(stocks: list[dict]) -> dict:
    """呼叫 TWSE MIS API，回傳 {code: quote_dict}"""
    ex_parts = [f"{s['market']}_{s['code'].lower()}.tw" for s in stocks]
    ex_ch    = "|".join(ex_parts)
    url = (
        "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
        f"?ex_ch={ex_ch}&json=1&delay=0"
    )

    sess = _get_twse_session()
    try:
        resp = sess.get(
            url,
            timeout=12,
            headers={
                "Referer":  "https://mis.twse.com.tw/stock/fibest.jsp",
                "Accept":   "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[TWSE MIS] 連線失敗：{e}")
        return {}

    results = {}
    for item in data.get("msgArray", []):
        code = item.get("c", "").upper()
        if not code:
            continue

        prev_close  = _safe_float(item.get("y"))
        raw_price   = _safe_float(item.get("z"))
        price       = raw_price or prev_close    # 休市時 z='-'，用昨收代替
        open_p      = _safe_float(item.get("o"))
        high        = _safe_float(item.get("h"))
        low         = _safe_float(item.get("l"))
        volume      = _safe_float(item.get("v"))  # 單位：張（千股）
        lmt_up      = _safe_float(item.get("u"))
        lmt_dn      = _safe_float(item.get("w"))

        change     = (price - prev_close) if (price and prev_close) else None
        change_pct = (change / prev_close * 100) if (change is not None and prev_close) else None

        near_up = bool(price and lmt_up and (lmt_up - price) / lmt_up < 0.02)
        near_dn = bool(price and lmt_dn and (price - lmt_dn) / price < 0.02)

        results[code] = {
            "code":           code,
            "name":           item.get("n", ""),
            "price":          price,
            "open":           open_p,
            "high":           high,
            "low":            low,
            "volume":         volume,
            "prev_close":     prev_close,
            "change":         change,
            "change_pct":     change_pct,
            "limit_up":       lmt_up,
            "limit_down":     lmt_dn,
            "near_limit_up":  near_up,
            "near_limit_down":near_dn,
            "realtime":       raw_price is not None,   # 有成交價才算即時
            "update_time":    f"{item.get('d','')} {item.get('t','')}".strip(),
        }
    return results


# ── yfinance 歷史 K 線（慢，只在完整重整時呼叫）────────────────
def fetch_yfinance_history(code: str, market: str, period: str = "1y") -> pd.DataFrame | None:
    suffixes = [".TW", ".TWO"] if market == "otc" else [".TW", ".TWO"]
    for sfx in suffixes:
        try:
            ticker = yf.Ticker(f"{code}{sfx}")
            df = ticker.history(period=period, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 5:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception:
            pass
    return None


# ── 完整資料載入（含歷史 K 線，慢）────────────────────────────
def get_stock_data(stocks: list[dict]) -> dict:
    rt_map   = fetch_twse_realtime(stocks)
    all_data: dict = {}

    for stock in stocks:
        code   = stock["code"].upper()
        market = stock.get("market", "tse")
        hist   = fetch_yfinance_history(code, market, period="1y")
        rt     = _build_rt(rt_map, stock, hist)

        all_data[code] = {
            "info":    stock,
            "realtime": rt,
            "history":  hist,
        }
    return all_data


# ── 快速更新：只打 TWSE API，不重抓 yfinance（秒級）──────────
def refresh_prices_only(stocks: list[dict], existing: dict) -> dict:
    """
    只更新即時報價欄位，保留歷史 K 線與技術指標。
    回傳更新後的 existing dict（in-place 修改並回傳）。
    """
    rt_map = fetch_twse_realtime(stocks)
    for stock in stocks:
        code = stock["code"].upper()
        if code not in existing:
            continue
        old_rt   = existing[code].get("realtime") or {}
        hist     = existing[code].get("history")
        new_rt   = _build_rt(rt_map, stock, hist, fallback_rt=old_rt)
        existing[code]["realtime"] = new_rt
    return existing


# ── 共用：從 rt_map 組裝單一股票的 realtime dict ─────────────
def _build_rt(rt_map, stock, hist, fallback_rt=None) -> dict:
    code = stock["code"].upper()
    rt   = rt_map.get(code, {})

    # ── 情況 A：TWSE 有資料但市場休市（z='-'，realtime=False）
    #    → 用 yfinance 最後兩筆收盤，計算「上個交易日的漲跌」
    if rt and not rt.get("realtime") and hist is not None and not hist.empty:
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        ch   = float(last["Close"] - prev["Close"])
        chp  = ch / float(prev["Close"]) * 100
        date_str = str(last.name.date()) if hasattr(last.name, "date") else ""
        rt.update({
            "price":      float(last["Close"]),
            "open":       float(last["Open"]),
            "high":       float(last["High"]),
            "low":        float(last["Low"]),
            "volume":     float(last["Volume"]),
            "prev_close": float(prev["Close"]),
            "change":     ch,
            "change_pct": chp,
            "realtime":   False,
            "update_time": date_str,
        })

    # ── 情況 B：TWSE 完全連不到 → 全用 yfinance
    elif not rt:
        if hist is not None and not hist.empty:
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else last
            ch   = float(last["Close"] - prev["Close"])
            chp  = ch / float(prev["Close"]) * 100
            date_str = str(last.name.date()) if hasattr(last.name, "date") else ""
            rt = {
                "code":           code,
                "name":           stock.get("name", code),
                "price":          float(last["Close"]),
                "open":           float(last["Open"]),
                "high":           float(last["High"]),
                "low":            float(last["Low"]),
                "volume":         float(last["Volume"]),
                "prev_close":     float(prev["Close"]),
                "change":         ch,
                "change_pct":     chp,
                "limit_up":       None,
                "limit_down":     None,
                "near_limit_up":  False,
                "near_limit_down":False,
                "realtime":       False,
                "update_time":    date_str,
            }
        elif fallback_rt:
            rt = dict(fallback_rt)
            rt["realtime"] = False
        else:
            rt = {}

    if rt and not rt.get("name"):
        rt["name"] = stock.get("name", code)

    return rt
