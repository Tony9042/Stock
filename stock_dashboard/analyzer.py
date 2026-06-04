"""
analyzer.py
短中長期評分、操作建議、文字摘要、今日異常提醒。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── 輔助 ─────────────────────────────────────────────────────
def _v(row, col):
    """安全取值，None / NaN 均回傳 None。"""
    val = row.get(col) if isinstance(row, dict) else getattr(row, col, None)
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _score_label(score: float) -> str:
    if score >= 75:
        return "強勢 ⬆️"
    if score >= 55:
        return "偏多 📈"
    if score >= 40:
        return "中性 ➡️"
    if score >= 25:
        return "偏弱 📉"
    return "弱勢 ⬇️"


# ── 主分析函式 ────────────────────────────────────────────────
def analyze_stock(
    code: str,
    realtime: dict,
    df: pd.DataFrame | None,
    stock_info: dict,
) -> dict | None:
    if df is None or df.empty or len(df) < 5:
        return None

    last  = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) >= 2 else last
    is_etf = stock_info.get("is_etf", False)
    themes = stock_info.get("theme", [])

    price = realtime.get("price") or float(last["Close"])

    def gl(col):   return _v(last, col)
    def gp(col):   return _v(prev, col)

    sma5   = gl("sma5");   sma10  = gl("sma10")
    sma20  = gl("sma20");  sma60  = gl("sma60");  sma120 = gl("sma120")
    K      = gl("K");      D      = gl("D");       J      = gl("J")
    pK     = gp("K");      pD     = gp("D")
    rsi    = gl("rsi")
    vol    = realtime.get("volume") or float(last.get("Volume", 0) or 0)
    vm5    = gl("vol_ma5"); vm20   = gl("vol_ma20")
    high_  = realtime.get("high") or float(last["High"])
    low_   = realtime.get("low")  or float(last["Low"])
    open_  = realtime.get("open") or float(last["Open"])
    h60    = gl("high_60"); l60 = gl("low_60")

    # ═══════════════════════════════════════════════════════════
    # 短期分數（基準 50，正負調整）
    # ═══════════════════════════════════════════════════════════
    short = 50
    short_signals: list[str] = []

    if sma5:
        if price > sma5:
            short += 10; short_signals.append("站上5日線 ✅")
        else:
            short -= 12; short_signals.append("跌破5日線 ⚠️")

    if sma10:
        if price > sma10:
            short += 8
        else:
            short -= 5

    # KD 交叉
    if K is not None and D is not None and pK is not None and pD is not None:
        if pK < pD and K > D:
            short += 18; short_signals.append("KD黃金交叉 ✨")
        elif pK > pD and K < D:
            short -= 15; short_signals.append("KD死亡交叉 ❌")
        if K > 80:
            short -= 8;  short_signals.append(f"K過熱({K:.0f}) ⚠️")
        elif K < 20:
            short += 8;  short_signals.append(f"K超賣({K:.0f}) 📌")

    if J is not None:
        if J > 100:
            short -= 10; short_signals.append(f"J值過高({J:.0f}) ⚠️")
        elif J < 0:
            short += 8;  short_signals.append(f"J值極低({J:.0f}) 📌")

    # RSI
    if rsi is not None:
        if rsi > 80:
            short -= 15; short_signals.append(f"RSI過熱({rsi:.1f}) ⚠️")
        elif rsi > 70:
            short -= 5;  short_signals.append(f"RSI偏高({rsi:.1f})")
        elif rsi < 30:
            short += 12; short_signals.append(f"RSI超賣({rsi:.1f}) 📌")
        elif 40 <= rsi <= 65:
            short += 8

    # 成交量
    if vol and vm5 and vm5 > 0:
        ratio = vol / vm5
        if ratio > 2.0:
            short += 10; short_signals.append(f"爆量({ratio:.1f}倍) 🔥")
        elif ratio > 1.3:
            short += 5;  short_signals.append(f"量增({ratio:.1f}倍)")
        elif ratio < 0.5:
            short -= 5;  short_signals.append("量縮 📉")

    # 乖離5日線
    if sma5 and sma5 > 0:
        div = (price - sma5) / sma5 * 100
        if div > 10:
            short -= 15; short_signals.append(f"乖離5日線{div:.1f}%過大 ⚠️")
        elif div > 7:
            short -= 7;  short_signals.append(f"乖離{div:.1f}%偏大")

    # 長上影線
    if open_ and high_ and price:
        body = abs(price - open_)
        upper_shadow = high_ - max(price, open_)
        if body > 0 and upper_shadow > body * 2.5 and upper_shadow > price * 0.015:
            short -= 10; short_signals.append("長上影線 ⚠️")

    # 接近漲停
    if realtime.get("near_limit_up"):
        short -= 8; short_signals.append("接近漲停，追價風險 ⚠️")

    short_score = max(0, min(100, short))

    # ═══════════════════════════════════════════════════════════
    # 中期分數
    # ═══════════════════════════════════════════════════════════
    mid = 0
    mid_signals: list[str] = []

    if sma20:
        if price > sma20:
            mid += 25; mid_signals.append("站上20日線 ✅")
        else:
            mid -= 5;  mid_signals.append("跌破20日線 ⚠️")
        # 20日線方向
        if len(df) >= 6:
            sma20_5ago = _v(df.iloc[-6], "sma20")
            if sma20_5ago and sma20 > sma20_5ago:
                mid += 15; mid_signals.append("20日線向上 ✅")
            elif sma20_5ago and sma20 < sma20_5ago:
                mid -= 8;  mid_signals.append("20日線向下 ⚠️")

    if sma60:
        if price > sma60:
            mid += 25; mid_signals.append("站上60日線 ✅")
        else:
            mid -= 8;  mid_signals.append("跌破60日線 ⚠️")
        if len(df) >= 11:
            sma60_10ago = _v(df.iloc[-11], "sma60")
            if sma60_10ago and sma60 > sma60_10ago:
                mid += 10; mid_signals.append("60日線向上 ✅")

    # 多頭排列
    if sma5 and sma20 and sma60:
        if price > sma5 > sma20 > sma60:
            mid += 15; mid_signals.append("完整多頭排列 ✨")
        elif price < sma5 < sma20:
            mid -= 12; mid_signals.append("空頭排列 ❌")

    # 突破前高
    if h60 and price >= h60 * 0.98:
        mid += 10; mid_signals.append("接近60日高點")
    if l60 and price <= l60 * 1.03:
        mid -= 10; mid_signals.append("接近60日低點 ⚠️")

    mid_score = max(0, min(100, mid))

    # ═══════════════════════════════════════════════════════════
    # 長期分數
    # ═══════════════════════════════════════════════════════════
    long = 0
    long_signals: list[str] = []

    if sma120:
        if price > sma120:
            long += 33; long_signals.append("站上120日線 ✅")
        else:
            long -= 5;  long_signals.append("跌破120日線 ⚠️")

    if sma60 and sma120:
        if sma60 > sma120:
            long += 20; long_signals.append("60日線>120日線 ✅")
        if len(df) >= 21:
            sma60_20ago = _v(df.iloc[-21], "sma60")
            if sma60_20ago and sma60 > sma60_20ago:
                long += 15; long_signals.append("60日線長期向上")

    # ETF 加分
    if is_etf:
        long += 20; long_signals.append("ETF長期持有佳 💰")
    elif any(t in themes for t in ["AI", "半導體", "資料中心", "電力", "伺服器"]):
        long += 15; long_signals.append("長期趨勢題材 🚀")

    long_score = max(0, min(100, long))

    # ═══════════════════════════════════════════════════════════
    # 綜合評分
    # ═══════════════════════════════════════════════════════════
    if is_etf:
        total = short_score * 0.15 + mid_score * 0.35 + long_score * 0.50
    else:
        total = short_score * 0.30 + mid_score * 0.40 + long_score * 0.30

    # ═══════════════════════════════════════════════════════════
    # 操作建議
    # ═══════════════════════════════════════════════════════════
    suggestion = _get_suggestion(
        price, sma5, sma20, sma60, sma120,
        rsi, K, D, J, vol, vm5,
        short_score, mid_score, long_score,
        is_etf, realtime,
    )

    # ═══════════════════════════════════════════════════════════
    # 文字分析
    # ═══════════════════════════════════════════════════════════
    text = _gen_text(
        code, stock_info.get("name", code),
        price, sma5, sma20, sma60, sma120,
        K, D, J, rsi,
        short_score, mid_score, long_score,
        short_signals, mid_signals, long_signals,
        is_etf, themes, suggestion,
    )

    return {
        "short_score": round(short_score, 1),
        "mid_score":   round(mid_score, 1),
        "long_score":  round(long_score, 1),
        "total_score": round(total, 1),
        "short_label": _score_label(short_score),
        "mid_label":   _score_label(mid_score),
        "long_label":  _score_label(long_score),
        "short_signals": short_signals,
        "mid_signals":   mid_signals,
        "long_signals":  long_signals,
        "suggestion":    suggestion,
        "text":          text,
    }


# ── 操作建議 ────────────────────────────────────────────────
def _get_suggestion(
    price, sma5, sma20, sma60, sma120,
    rsi, K, D, J, vol, vm5,
    short_score, mid_score, long_score,
    is_etf, realtime,
) -> str:
    if is_etf:
        return "長期定期定額即可"

    change_pct = realtime.get("change_pct") or 0
    if realtime.get("near_limit_up") or change_pct > 8.5:
        return "不建議追價"

    if sma5 and price and sma5 > 0:
        div = (price - sma5) / sma5 * 100
        if div > 10:
            return "等拉回"

    if (rsi and rsi > 80) or (J is not None and J > 100):
        return "等拉回"

    if sma20 and price < sma20:
        if sma60 and price < sma60 * 0.97:
            return "跌破支撐停損"
        return "等拉回"

    if short_score >= 62 and mid_score >= 55:
        vol_ok = (vol and vm5 and vol > vm5 * 0.9)
        if vol_ok:
            return "可小量試單"
        return "可小量試單"

    if mid_score >= 65 and long_score >= 55:
        return "續抱"

    if short_score >= 70 and mid_score < 45:
        return "分批停利"

    if realtime.get("near_limit_down"):
        return "跌破支撐停損"

    return "等拉回"


# ── 文字摘要 ─────────────────────────────────────────────────
def _gen_text(
    code, name,
    price, sma5, sma20, sma60, sma120,
    K, D, J, rsi,
    short_score, mid_score, long_score,
    short_signals, mid_signals, long_signals,
    is_etf, themes, suggestion,
) -> str:
    lines = [f"【{name} {code}】"]

    # 短期
    slab = "偏多" if short_score >= 60 else ("中性" if short_score >= 40 else "偏弱")
    extras = []
    if rsi and rsi > 80:
        extras.append("RSI過熱，追價風險高")
    if J is not None and J > 100:
        extras.append("J值過高，短線注意")
    if sma5 and price < sma5:
        extras.append("跌破5日線，短線轉弱")
    if sma5 and price > sma5 * 1.1:
        extras.append("乖離過大，宜等拉回")
    s_txt = f"短期：{slab}"
    if extras:
        s_txt += "，" + "；".join(extras) + "。"
    else:
        s_txt += "，技術面無特別警示。"
    lines.append(s_txt)

    # 中期
    if mid_score >= 65:
        mlab = "仍維持多頭排列"
    elif mid_score >= 45:
        mlab = "中性，均線糾結"
    else:
        mlab = "偏弱，注意支撐"
    m_detail = []
    if sma20:
        if price > sma20:
            m_detail.append("20日線未跌破，趨勢不壞")
        else:
            m_detail.append("跌破20日線，中線趨弱")
    if sma60:
        if price > sma60:
            m_detail.append("站上60日線")
        else:
            m_detail.append("跌破60日線需留意")
    m_txt = f"中期：{mlab}"
    if m_detail:
        m_txt += "，" + "；".join(m_detail) + "。"
    else:
        m_txt += "。"
    lines.append(m_txt)

    # 長期
    if is_etf:
        lines.append("長期：ETF適合長期持有，定期定額分散成本，不需頻繁操作。")
    else:
        if long_score >= 65:
            llab = "長期多頭格局"
        elif long_score >= 45:
            llab = "長期中性"
        else:
            llab = "長期弱勢"
        th = "、".join(themes) if themes else ""
        th_str = f"，屬{th}題材" if th else ""
        l120 = ""
        if sma120:
            l120 = "，站上120日線趨勢向好" if price > sma120 else "，跌破120日線需觀察"
        lines.append(f"長期：{llab}{th_str}{l120}。")

    # 建議
    suggest_detail = {
        "可小量試單":       "短線站穩可小量試單，嚴設停損（跌破SMA5或-5%）。",
        "等拉回":           "建議等待拉回至均線附近（SMA5～SMA20區間）再分批介入。",
        "續抱":             "已持有可續抱，注意5日線支撐。",
        "分批停利":         "短線強勢但中期訊號待確認，可分批減碼停利。",
        "不建議追價":       "漲幅已大或接近漲停，高追風險高，等回測再評估。",
        "跌破支撐停損":     "已跌破重要均線支撐，持有者應設停損，新資金暫緩進場。",
        "長期定期定額即可": "ETF適合長期定期定額，不需短線操作，持續布局即可。",
    }.get(suggestion, suggestion)
    lines.append(f"建議：{suggest_detail}")

    return "\n".join(lines)


# ── 今日異常提醒 ──────────────────────────────────────────────
SEVERITY_ORDER = {"danger": 0, "warning": 1, "success": 2, "info": 3}


def get_alerts(
    code: str,
    realtime: dict,
    df: pd.DataFrame | None,
    stock_info: dict,
) -> list[dict]:
    if df is None or df.empty:
        return []

    alerts: list[dict] = []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    price  = realtime.get("price") or float(last["Close"])
    open_  = realtime.get("open")  or float(last["Open"])
    high_  = realtime.get("high")  or float(last["High"])
    low_   = realtime.get("low")   or float(last["Low"])
    vol    = realtime.get("volume") or float(last.get("Volume", 0) or 0)

    def add(t, s): alerts.append({"type": t, "severity": s})

    def g(col): return _v(last, col)
    def gp(col): return _v(prev, col)

    if realtime.get("near_limit_up"):  add("接近漲停 🔴", "warning")
    if realtime.get("near_limit_down"):add("接近跌停 🔵", "danger")

    vm20 = g("vol_ma20")
    if vol and vm20 and vm20 > 0 and vol > vm20 * 2:
        add("爆量 📊", "info")

    sma5  = g("sma5")
    sma20 = g("sma20")
    if sma5  and price < sma5:  add("跌破5日線 ⬇️",  "warning")
    if sma20 and price < sma20: add("跌破20日線 ⬇️", "danger")

    K = g("K"); D = g("D"); pK = gp("K"); pD = gp("D")
    if all(v is not None for v in [K, D, pK, pD]):
        if pK < pD and K > D: add("KD黃金交叉 ✨", "success")
        if pK > pD and K < D: add("KD死亡交叉 ❌", "danger")

    rsi = g("rsi")
    if rsi and rsi > 80: add(f"RSI過熱({rsi:.0f}) 🌡️", "warning")
    if rsi and rsi < 25: add(f"RSI超賣({rsi:.0f}) 📌", "info")

    # 跳空
    prev_high = float(prev["High"])
    prev_low  = float(prev["Low"])
    if open_ and open_ > prev_high * 1.005: add("跳空上漲 ⬆️", "info")
    if open_ and open_ < prev_low  * 0.995: add("跳空下跌 ⬇️", "warning")

    # 長上影線
    if open_ and high_ and price:
        body = abs(price - open_)
        upper_shadow = high_ - max(price, open_)
        if body > 0 and upper_shadow > body * 2.5 and upper_shadow > price * 0.015:
            add("長上影線 ⚠️", "warning")

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 9))
    return alerts
