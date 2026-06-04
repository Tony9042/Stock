"""
台股自選股看盤與短中長期評估系統
streamlit run app.py
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime
import time

from data_fetcher import (
    get_stock_data, load_watchlist, save_watchlist, DEFAULT_WATCHLIST
)
from indicators import calculate_indicators
from analyzer import analyze_stock, get_alerts

# ── 頁面設定 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="台股自選股看盤系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
/* ══════════════════════════════════════════
   基礎 & 全域
══════════════════════════════════════════ */
html, body, [class*="css"] {
    -webkit-tap-highlight-color: transparent;
}

/* ── 快速卡片 ── */
.metric-box {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 4px 0;
}
.metric-box .code-name {
    font-size: 0.82em;
    color: #c0c8e8;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 4px;
}
.metric-box .price {
    font-size: 1.55em;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.01em;
}
.metric-box .scores {
    font-size: 0.82em;
    color: #8899bb;
    margin-top: 4px;
}
.metric-box .suggest {
    font-size: 0.80em;
    color: #d0d8f0;
    margin-top: 3px;
}

/* ── 漲跌色 ── */
.up      { color: #ff5252; font-weight: 700; }
.down    { color: #00e676; font-weight: 700; }
.neutral { color: #90a4ae; }

/* ── 今日提醒框 ── */
.alert-danger {
    background: #4a1010;
    border-left: 5px solid #ff1744;
    padding: 8px 14px;
    border-radius: 5px;
    margin: 4px 0;
    color: #ffcdd2;
    font-size: 0.92em;
    font-weight: 500;
}
.alert-warning {
    background: #4a3000;
    border-left: 5px solid #ff9100;
    padding: 8px 14px;
    border-radius: 5px;
    margin: 4px 0;
    color: #ffe0b2;
    font-size: 0.92em;
    font-weight: 500;
}
.alert-success {
    background: #0a3d1f;
    border-left: 5px solid #00e676;
    padding: 8px 14px;
    border-radius: 5px;
    margin: 4px 0;
    color: #b9f6ca;
    font-size: 0.92em;
    font-weight: 500;
}
.alert-info {
    background: #0d2540;
    border-left: 5px solid #40c4ff;
    padding: 8px 14px;
    border-radius: 5px;
    margin: 4px 0;
    color: #b3e5fc;
    font-size: 0.92em;
    font-weight: 500;
}

/* ══════════════════════════════════════════
   平板 (≤ 1024px)
══════════════════════════════════════════ */
@media screen and (max-width: 1024px) {
    /* 讓 5 欄變 2 欄排列 */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="column"] {
        min-width: 45% !important;
        flex: 1 1 45% !important;
    }
    .metric-box .price { font-size: 1.35em; }
}

/* ══════════════════════════════════════════
   手機 (≤ 768px)
══════════════════════════════════════════ */
@media screen and (max-width: 768px) {
    /* 所有欄位強制滿版 */
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* 標題縮小 */
    h1 { font-size: 1.4em !important; }
    h2 { font-size: 1.15em !important; }
    h3 { font-size: 1.05em !important; }

    /* 快速卡片放大字體方便手指點 */
    .metric-box {
        padding: 14px 16px;
        margin: 6px 0;
    }
    .metric-box .code-name { font-size: 0.9em; }
    .metric-box .price     { font-size: 1.8em; }
    .metric-box .scores    { font-size: 0.88em; }
    .metric-box .suggest   { font-size: 0.86em; }

    /* 提醒框字體加大 */
    .alert-danger, .alert-warning,
    .alert-success, .alert-info {
        font-size: 1em;
        padding: 10px 14px;
    }

    /* Dataframe 水平捲動 */
    [data-testid="stDataFrame"] > div {
        overflow-x: auto !important;
    }

    /* 按鈕加大點擊區 */
    button[kind="primary"],
    button[kind="secondary"] {
        min-height: 48px !important;
        font-size: 1em !important;
    }

    /* Tab 標籤縮短 */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.78em !important;
        padding: 8px 6px !important;
    }

    /* Metric 元件放大 */
    [data-testid="stMetric"] label {
        font-size: 0.85em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3em !important;
    }

    /* Sidebar 預設收合，漢堡選單仍可點 */
    [data-testid="collapsedControl"] {
        display: flex !important;
    }

    /* Plotly 圖表全寬 */
    .js-plotly-plot {
        width: 100% !important;
    }

    /* Radio 橫排改直排 */
    [data-testid="stRadio"] > div {
        flex-direction: column !important;
        gap: 4px !important;
    }

    /* Slider 加大觸控點 */
    [data-testid="stSlider"] > div > div > div {
        height: 24px !important;
    }
}

/* ══════════════════════════════════════════
   小手機 (≤ 480px)
══════════════════════════════════════════ */
@media screen and (max-width: 480px) {
    h1 { font-size: 1.2em !important; }
    .metric-box .price { font-size: 2em; }

    /* Tab 文字再縮 */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.70em !important;
        padding: 6px 4px !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Session state 初始化 ──────────────────────────────────────
if "watchlist"    not in st.session_state:
    st.session_state.watchlist    = load_watchlist()
if "stock_data"   not in st.session_state:
    st.session_state.stock_data   = {}
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "loading"      not in st.session_state:
    st.session_state.loading      = False
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# ── 市場交易時間判斷 ──────────────────────────────────────────
def is_trading_hours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:          # 六日休市
        return False
    t = now.time()
    return dtime(8, 55) <= t <= dtime(13, 35)


# ── 評分顏色 helper ───────────────────────────────────────────
def score_color(v: float) -> str:
    if v >= 70: return "#00e676"    # 綠：強
    if v >= 55: return "#69f0ae"    # 淺綠：偏多
    if v >= 40: return "#ffd740"    # 黃：中性
    if v >= 25: return "#ff9100"    # 橘：偏弱
    return "#ff5252"                # 紅：弱


# ── 資料載入函式 ─────────────────────────────────────────────
def do_refresh():
    stocks = st.session_state.watchlist["stocks"]
    raw    = get_stock_data(stocks)
    processed: dict = {}
    for code, data in raw.items():
        hist = calculate_indicators(data["history"])
        rt   = data["realtime"] or {}
        processed[code] = {
            "info":     data["info"],
            "realtime": rt,
            "history":  hist,
            "analysis": analyze_stock(code, rt, hist, data["info"]),
            "alerts":   get_alerts(code, rt, hist, data["info"]),
        }
    st.session_state.stock_data   = processed
    st.session_state.last_refresh = datetime.now()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📋 自選股管理")

    # ── 新增股票 ──────────────────────────────────────────────
    with st.expander("➕ 新增股票"):
        nc = st.text_input("股票代號（如 2454）", key="nc")
        nn = st.text_input("股票名稱（如 聯發科）", key="nn")
        nm = st.selectbox("市場", ["tse", "otc"], key="nm",
                          help="上市選 tse；上櫃選 otc")
        ne = st.checkbox("ETF", key="ne")
        nth = st.text_input("題材（用逗號分隔，如 AI,半導體）", key="nth")
        if st.button("確認新增", use_container_width=True):
            if nc.strip():
                new_s = {
                    "code":   nc.strip().upper(),
                    "name":   nn.strip() or nc.strip(),
                    "market": nm,
                    "is_etf": ne,
                    "theme":  [t.strip() for t in nth.split(",") if t.strip()],
                }
                existing = [s["code"] for s in st.session_state.watchlist["stocks"]]
                if new_s["code"] in existing:
                    st.warning("此代號已在清單中")
                else:
                    st.session_state.watchlist["stocks"].append(new_s)
                    save_watchlist(st.session_state.watchlist)
                    st.success(f"✅ 已新增 {new_s['code']}")
                    st.rerun()
            else:
                st.warning("請輸入股票代號")

    # ── 刪除股票 ──────────────────────────────────────────────
    with st.expander("➖ 刪除股票"):
        options = [
            f"{s['code']} {s.get('name','')}"
            for s in st.session_state.watchlist["stocks"]
        ]
        if options:
            rm_sel = st.selectbox("選擇刪除", options, key="rm_sel")
            if st.button("確認刪除", use_container_width=True):
                rm_code = rm_sel.split()[0]
                st.session_state.watchlist["stocks"] = [
                    s for s in st.session_state.watchlist["stocks"]
                    if s["code"] != rm_code
                ]
                save_watchlist(st.session_state.watchlist)
                st.success(f"✅ 已刪除 {rm_code}")
                st.rerun()
        else:
            st.info("清單為空")

    # ── 重設為預設清單 ────────────────────────────────────────
    if st.button("🔁 重設預設自選股", use_container_width=True):
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
        save_watchlist(st.session_state.watchlist)
        st.success("已重設")
        st.rerun()

    st.divider()

    # ── 重新整理 ──────────────────────────────────────────────
    if st.button("🔄 重新整理資料", use_container_width=True, type="primary"):
        with st.spinner("資料載入中，請稍候…"):
            do_refresh()
        st.success("✅ 資料更新完成！")
        st.rerun()

    if st.session_state.last_refresh:
        st.caption(f"最後更新：{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

    st.divider()

    # ── 自動刷新 ──────────────────────────────────────────────
    st.subheader("⏱ 自動刷新")
    auto_on = st.toggle(
        "交易時間自動刷新",
        value=st.session_state.auto_refresh,
        help="僅在 09:00–13:30 平日生效",
    )
    st.session_state.auto_refresh = auto_on
    refresh_interval = st.select_slider(
        "刷新間隔（秒）", [30, 60, 120, 180, 300],
        value=60, disabled=not auto_on,
    )
    if auto_on:
        if is_trading_hours():
            st.success(f"🟢 交易中，每 {refresh_interval}s 刷新")
        else:
            st.warning("⏸ 非交易時間，自動刷新暫停")

    st.divider()

    # ── 篩選 ──────────────────────────────────────────────────
    st.subheader("🔍 篩選")
    show_etf      = st.checkbox("顯示 ETF",    value=True)
    show_stock    = st.checkbox("顯示個股",    value=True)
    hide_highrisk = st.checkbox("隱藏高風險股票\n（短期評分 < 30）", value=False)
    min_total     = st.slider("最低綜合評分", 0, 100, 0, step=5)

    st.divider()
    st.caption(
        "⚠️ **免責聲明**\n\n"
        "本系統僅提供資料整理與技術分析輔助，"
        "不構成任何投資建議。投資人應自行判斷風險，"
        "盈虧自負。"
    )


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════
st.title("📈 台股自選股看盤與評估系統")

# ── 自動刷新執行 ──────────────────────────────────────────────
if st.session_state.auto_refresh and is_trading_hours() and st.session_state.stock_data:
    _ph = st.empty()
    _last = st.session_state.last_refresh
    if _last is None or (datetime.now() - _last).seconds >= refresh_interval:
        with _ph.container():
            st.info("🔄 自動刷新中…")
        do_refresh()
        _ph.empty()
        st.rerun()

# 第一次進入：自動載入
if not st.session_state.stock_data:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("👈 點擊左側「重新整理資料」載入自選股，或點擊下方快速載入。")
    with col2:
        if st.button("⚡ 立即載入", type="primary"):
            with st.spinner("資料載入中…"):
                do_refresh()
            st.rerun()
    st.stop()

# 篩選
all_data = st.session_state.stock_data
filtered: dict = {}
for code, data in all_data.items():
    info   = data.get("info", {})
    is_etf = info.get("is_etf", False)
    if is_etf and not show_etf:
        continue
    if not is_etf and not show_stock:
        continue
    an = data.get("analysis") or {}
    if an.get("total_score", 0) < min_total:
        continue
    if hide_highrisk and an.get("short_score", 100) < 30:
        continue
    filtered[code] = data

if not filtered:
    st.warning("無符合條件的股票，請調整篩選條件。")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────
tab_ov, tab_alert, tab_rank, tab_detail = st.tabs(
    ["📊 總覽", "🚨 今日異常提醒", "🏆 排行榜", "📈 個股詳細分析"]
)


# ════════════════════════════════════════════════════════════
# TAB 1：總覽
# ════════════════════════════════════════════════════════════
with tab_ov:
    st.subheader("自選股總覽")

    rows = []
    for code, data in filtered.items():
        rt  = data.get("realtime") or {}
        an  = data.get("analysis") or {}
        inf = data.get("info", {})

        price     = rt.get("price")
        change    = rt.get("change")
        chg_pct   = rt.get("change_pct")
        vol       = rt.get("volume")
        hi        = rt.get("high")
        lo        = rt.get("low")
        op        = rt.get("open")
        is_rt     = rt.get("realtime", False)
        near_up   = rt.get("near_limit_up", False)
        near_dn   = rt.get("near_limit_down", False)

        def fmt(v, dec=2):
            return f"{v:.{dec}f}" if v is not None else "-"

        chg_str = f"{change:+.2f}" if change is not None else "-"
        pct_str = f"{chg_pct:+.2f}%" if chg_pct is not None else "-"

        # 近日高低（從歷史）
        hist = data.get("history")
        h20 = lo20 = h60 = lo60 = None
        if hist is not None and not hist.empty:
            lr = hist.iloc[-1]
            def _hv(c):
                try: return float(lr[c])
                except: return None
            h20  = _hv("high_20"); lo20 = _hv("low_20")
            h60  = _hv("high_60"); lo60 = _hv("low_60")

        short_s = an.get("short_score", 0) if an else 0
        mid_s   = an.get("mid_score",   0) if an else 0
        long_s  = an.get("long_score",  0) if an else 0
        tot_s   = an.get("total_score", 0) if an else 0

        rows.append({
            "代號":    code,
            "名稱":    rt.get("name") or inf.get("name", code),
            "即時":    "✅" if is_rt else "📋",
            "成交價":  fmt(price),
            "漲跌":    chg_str,
            "漲跌幅":  pct_str,
            "量(張)":  f"{vol/1000:.0f}K" if vol and vol >= 1000 else fmt(vol, 0),
            "今高":    fmt(hi),
            "今低":    fmt(lo),
            "開盤":    fmt(op),
            "20日高":  fmt(h20),
            "20日低":  fmt(lo20),
            "60日高":  fmt(h60),
            "60日低":  fmt(lo60),
            "漲停警示": "🔴" if near_up else ("🔵" if near_dn else ""),
            "短期":    short_s,
            "中期":    mid_s,
            "長期":    long_s,
            "總評":    tot_s,
            "建議":    an.get("suggestion", "-") if an else "-",
        })

    df_table = pd.DataFrame(rows)

    def color_chg(val):
        try:
            v = float(str(val).replace("%", "").replace("+", ""))
            if v > 0: return "color:#ff5252;font-weight:700"
            if v < 0: return "color:#00e676;font-weight:700"
        except Exception:
            pass
        return ""

    def color_score(val):
        try:
            v = float(val)
            if v >= 70: return "background:#0a3d1f;color:#00e676;font-weight:700"
            if v >= 55: return "background:#0d2e14;color:#69f0ae;font-weight:600"
            if v >= 40: return "background:#3d3000;color:#ffd740"
            if v >= 25: return "background:#3d1d00;color:#ff9100"
            return "background:#3d0a0a;color:#ff5252;font-weight:700"
        except Exception:
            return ""

    styled = (
        df_table.style
        .map(color_chg,   subset=["漲跌", "漲跌幅"])
        .map(color_score, subset=["短期", "中期", "長期", "總評"])
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{
            "selector": "th",
            "props": [("text-align","center"),("background","#1e2130"),("color","#c0c8e8")]
        }])
        .format({
            "短期": "{:.0f}", "中期": "{:.0f}",
            "長期": "{:.0f}", "總評": "{:.0f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=500)

    # 快速卡片
    st.divider()
    st.subheader("快速卡片")
    cols = st.columns(2)          # 2欄：手機CSS會疊成1欄，桌機看起來整齊
    for i, (code, data) in enumerate(filtered.items()):
        rt = data.get("realtime") or {}
        an = data.get("analysis") or {}
        with cols[i % 2]:
            price   = rt.get("price")
            chg_pct = rt.get("change_pct")
            cls = "up" if (chg_pct or 0) > 0 else ("down" if (chg_pct or 0) < 0 else "neutral")
            p_str  = f"{price:.2f}" if price else "-"
            cp_str = f"{chg_pct:+.2f}%" if chg_pct is not None else "-"
            name = rt.get("name") or data["info"].get("name", code)
            ss = an.get('short_score', 0)
            ms = an.get('mid_score',   0)
            ls = an.get('long_score',  0)
            st.markdown(f"""
<div class="metric-box">
  <div class="code-name">{code} &nbsp;{name}</div>
  <div>
    <span class="price">{p_str}</span>
    &nbsp;<span class="{cls}" style="font-size:1.05em">{cp_str}</span>
  </div>
  <div class="scores">
    短&nbsp;<b style="color:#ff8a65">{ss:.0f}</b>
    &nbsp;|&nbsp;中&nbsp;<b style="color:#42a5f5">{ms:.0f}</b>
    &nbsp;|&nbsp;長&nbsp;<b style="color:#66bb6a">{ls:.0f}</b>
  </div>
  <div class="suggest">💡 {an.get('suggestion','-')}</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2：今日異常提醒
# ════════════════════════════════════════════════════════════
with tab_alert:
    st.subheader("🚨 今日異常提醒")

    all_alerts = []
    for code, data in filtered.items():
        name = (data.get("realtime") or {}).get("name") or data["info"].get("name", code)
        for a in (data.get("alerts") or []):
            all_alerts.append({"code": code, "name": name, **a})

    if not all_alerts:
        st.success("目前無異常訊號。")
    else:
        severity_label = {"danger": "🔴 高", "warning": "🟠 中", "success": "🟢", "info": "🔵 資訊"}
        # 依 severity 分組顯示
        for sev in ["danger", "warning", "success", "info"]:
            group = [a for a in all_alerts if a["severity"] == sev]
            if not group:
                continue
            st.markdown(f"**{severity_label.get(sev, sev)}**")
            for a in group:
                st.markdown(
                    f'<div class="alert-{sev}">'
                    f'<b>{a["code"]} {a["name"]}</b> — {a["type"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════
# TAB 3：排行榜
# ════════════════════════════════════════════════════════════
with tab_rank:

    def build_rank_rows():
        rows = []
        for code, d in filtered.items():
            rt = d.get("realtime") or {}
            an = d.get("analysis") or {}
            rows.append({
                "代號":   code,
                "名稱":   rt.get("name") or d["info"].get("name", code),
                "成交價": rt.get("price"),
                "漲跌幅": rt.get("change_pct"),
                "短期":   an.get("short_score", 0),
                "中期":   an.get("mid_score",   0),
                "長期":   an.get("long_score",  0),
                "總評":   an.get("total_score", 0),
                "建議":   an.get("suggestion",  "-"),
                "ETF":    d["info"].get("is_etf", False),
            })
        return rows

    def score_bar_chart(df_: pd.DataFrame, score_col: str, title: str, color: str):
        df_ = df_.head(12)
        labels = df_["代號"] + " " + df_["名稱"]
        fig = go.Figure(go.Bar(
            x=df_[score_col], y=labels,
            orientation="h",
            marker=dict(
                color=df_[score_col],
                colorscale=[[0,"#ff5252"],[0.4,"#ff9100"],[0.6,"#ffd740"],[0.8,"#69f0ae"],[1,"#00e676"]],
                cmin=0, cmax=100,
                showscale=False,
            ),
            text=[f"{v:.0f}" for v in df_[score_col]],
            textposition="outside",
            customdata=df_[["成交價","漲跌幅","建議"]].values,
            hovertemplate="<b>%{y}</b><br>評分：%{x:.0f}<br>價：%{customdata[0]:.2f} / %{customdata[1]:+.2f}%<br>建議：%{customdata[2]}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font_size=13, font_color="#c0c8e8"),
            height=max(280, len(df_) * 32 + 60),
            margin=dict(l=130,r=50,t=45,b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc", size=11),
            xaxis=dict(range=[0,115], gridcolor="#1e2130"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    rank_rows = build_rank_rows()
    df_rank = pd.DataFrame(rank_rows)

    col_l, col_r = st.columns(2)
    with col_l:
        score_bar_chart(
            df_rank.sort_values("短期", ascending=False),
            "短期", "🏆 短線機會排行（短期評分）", "#ff8a65"
        )
        score_bar_chart(
            df_rank.sort_values("總評", ascending=False),
            "總評", "🏆 綜合評分排行", "#66bb6a"
        )

    with col_r:
        score_bar_chart(
            df_rank.sort_values("短期", ascending=True),
            "短期", "⚠️ 風險排行（短期評分低→高）", "#ef5350"
        )
        score_bar_chart(
            df_rank.sort_values("中期", ascending=True),
            "中期", "📉 中期弱勢排行", "#ffa726"
        )

    # 完整評分明細表
    st.divider()
    st.markdown("**📋 完整評分明細**")
    df_show = df_rank.sort_values("總評", ascending=False).copy()
    df_show["漲跌幅"] = df_show["漲跌幅"].apply(lambda v: f"{v:+.2f}%" if v is not None else "-")
    df_show["成交價"] = df_show["成交價"].apply(lambda v: f"{v:.2f}" if v else "-")
    df_show["ETF"]    = df_show["ETF"].apply(lambda v: "✅" if v else "")
    df_show = df_show[["代號","名稱","ETF","成交價","漲跌幅","短期","中期","長期","總評","建議"]]
    styled_r = (
        df_show.style
        .map(lambda v: f"color:#ff5252;font-weight:700" if isinstance(v,str) and v.startswith("+") else
                       (f"color:#00e676;font-weight:700" if isinstance(v,str) and v.startswith("-") else ""),
             subset=["漲跌幅"])
        .map(lambda v: f"background:#0a3d1f;color:#00e676;font-weight:700" if isinstance(v,(int,float)) and v>=70 else
                       (f"background:#3d3000;color:#ffd740" if isinstance(v,(int,float)) and 40<=v<70 else
                       (f"background:#3d0a0a;color:#ff5252;font-weight:700" if isinstance(v,(int,float)) and v<40 else "")),
             subset=["短期","中期","長期","總評"])
        .format({"短期":"{:.0f}","中期":"{:.0f}","長期":"{:.0f}","總評":"{:.0f}"})
        .set_properties(**{"text-align":"center"})
    )
    st.dataframe(styled_r, use_container_width=True, height=380)


# ════════════════════════════════════════════════════════════
# TAB 4：個股詳細分析 + 圖表
# ════════════════════════════════════════════════════════════
with tab_detail:
    st.subheader("📈 個股詳細分析")

    code_options = [
        f"{code} {(data.get('realtime') or {}).get('name') or data['info'].get('name', code)}"
        for code, data in filtered.items()
    ]
    sel = st.selectbox("選擇股票", code_options, key="detail_sel")
    sel_code = sel.split()[0]
    data = filtered.get(sel_code, {})

    if not data:
        st.warning("找不到資料")
        st.stop()

    rt  = data.get("realtime") or {}
    an  = data.get("analysis") or {}
    df  = data.get("history")
    inf = data.get("info", {})
    name = rt.get("name") or inf.get("name", sel_code)

    # ── 基本資訊列 ────────────────────────────────────────────
    is_rt = rt.get("realtime", False)
    rt_tag = "即時" if is_rt else "非即時（歷史收盤）"

    price   = rt.get("price");   change  = rt.get("change")
    chg_pct = rt.get("change_pct")
    vol     = rt.get("volume");  hi      = rt.get("high")
    lo      = rt.get("low");     op      = rt.get("open")
    lmt_up  = rt.get("limit_up"); lmt_dn = rt.get("limit_down")
    near_up = rt.get("near_limit_up", False)
    near_dn = rt.get("near_limit_down", False)

    st.markdown(f"### {name}（{sel_code}） <small style='color:#aaa;font-size:0.6em'>[{rt_tag}]</small>",
                unsafe_allow_html=True)
    if not is_rt:
        st.info("⚠️ 無法取得即時報價（可能休市或代號有誤），顯示最近收盤資料。")

    # 第一列：3欄（手機CSS疊成1欄）
    c1, c2, c3 = st.columns(3)
    def _m(label, val, delta=None):
        st.metric(label, f"{val:.2f}" if val else "-",
                  delta=f"{delta:+.2f}" if delta is not None else None)
    with c1: _m("成交價", price, change)
    with c2:
        pct_str = f"{chg_pct:+.2f}%" if chg_pct is not None else "-"
        st.metric("漲跌幅", pct_str)
    with c3:
        v_str = f"{vol/1000:.1f}K張" if vol and vol >= 1000 else (f"{vol:.0f}張" if vol else "-")
        st.metric("成交量", v_str)

    # 第二列：4欄
    c4, c5, c6, c7 = st.columns(4)
    with c4: _m("今高", hi)
    with c5: _m("今低", lo)
    with c6: _m("開盤", op)
    with c7:
        flags = []
        if near_up: flags.append("🔴漲停")
        if near_dn: flags.append("🔵跌停")
        st.metric("警示", " ".join(flags) if flags else "無")

    # ── 技術評分 ──────────────────────────────────────────────
    st.divider()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.metric("短期評分", f"{an.get('short_score',0):.0f}/100",
                  an.get("short_label", "-"))
        st.metric("長期評分", f"{an.get('long_score',0):.0f}/100",
                  an.get("long_label", "-"))
    with sc2:
        st.metric("中期評分", f"{an.get('mid_score',0):.0f}/100",
                  an.get("mid_label", "-"))
        st.metric("綜合評分", f"{an.get('total_score',0):.0f}/100",
                  f"💡 {an.get('suggestion', '-')}")

    # 評分視覺化
    scores = {
        "短期": an.get("short_score", 0),
        "中期": an.get("mid_score",   0),
        "長期": an.get("long_score",  0),
        "綜合": an.get("total_score", 0),
    }
    fig_bar = go.Figure(go.Bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        marker_color=["#ff7043","#ffa726","#42a5f5","#66bb6a"],
        text=[f"{v:.0f}" for v in scores.values()],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=220, margin=dict(l=0,r=0,t=20,b=0),
        yaxis_range=[0,110], paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 今日訊號 ──────────────────────────────────────────────
    alerts = data.get("alerts", [])
    if alerts:
        st.markdown("**今日訊號**")
        cols_a = st.columns(min(4, len(alerts)))
        for i, a in enumerate(alerts):
            with cols_a[i % 4]:
                st.markdown(
                    f'<div class="alert-{a["severity"]}">{a["type"]}</div>',
                    unsafe_allow_html=True,
                )

    # ── 文字分析 ──────────────────────────────────────────────
    st.divider()
    with st.expander("📝 文字分析報告", expanded=True):
        st.markdown(an.get("text", "（無資料）").replace("\n", "\n\n"))

    # ── 指標信號明細 ──────────────────────────────────────────
    col_sig1, col_sig2, col_sig3 = st.columns(3)
    with col_sig1:
        st.markdown("**短期信號**")
        for s in an.get("short_signals", []):
            st.markdown(f"- {s}")
    with col_sig2:
        st.markdown("**中期信號**")
        for s in an.get("mid_signals", []):
            st.markdown(f"- {s}")
    with col_sig3:
        st.markdown("**長期信號**")
        for s in an.get("long_signals", []):
            st.markdown(f"- {s}")

    # ── 指標數值＋狀態卡 ─────────────────────────────────────
    if df is not None and not df.empty:
        st.divider()
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) >= 2 else last_row

        def gv(col):
            v = last_row.get(col) if hasattr(last_row,"get") else getattr(last_row, col, None)
            try: return float(v) if v is not None and not np.isnan(float(v)) else None
            except: return None
        def gvp(col):
            v = prev_row.get(col) if hasattr(prev_row,"get") else getattr(prev_row, col, None)
            try: return float(v) if v is not None and not np.isnan(float(v)) else None
            except: return None

        # 均線距離一覽
        st.markdown("**📐 價格與均線距離**")
        sma_def = [("SMA5","sma5"),("SMA10","sma10"),("SMA20","sma20"),
                   ("SMA60","sma60"),("SMA120","sma120")]
        sma_cols = st.columns(3)   # 桌機3欄，手機CSS疊成1欄
        for ci, (label, col_n) in enumerate(sma_def):
            sma_v = gv(col_n)
            with sma_cols[ci]:
                if sma_v and price:
                    diff = (price - sma_v) / sma_v * 100
                    above = price >= sma_v
                    arrow = "▲" if above else "▼"
                    color = "#00e676" if above else "#ff5252"
                    st.markdown(
                        f"""<div style="background:#1e2130;border-radius:8px;padding:10px;text-align:center;border:1px solid #2d3148">
                        <div style="font-size:0.8em;color:#8899bb">{label}</div>
                        <div style="font-size:1.1em;color:#fff;font-weight:600">{sma_v:.2f}</div>
                        <div style="font-size:0.95em;color:{color};font-weight:700">{arrow} {diff:+.1f}%</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div style="background:#1e2130;border-radius:8px;padding:10px;text-align:center">
                        <div style="font-size:0.8em;color:#8899bb">{label}</div>
                        <div style="color:#555">資料不足</div></div>""",
                        unsafe_allow_html=True,
                    )

        st.markdown("")

        # 振盪指標狀態卡
        st.markdown("**🎛 振盪指標狀態**")
        ind2_cols = st.columns(2)   # 桌機2欄，手機疊成1欄

        # KD 卡
        K_v = gv("K"); D_v = gv("D"); J_v = gv("J")
        Kp  = gvp("K"); Dp  = gvp("D")
        with ind2_cols[0]:
            if K_v is not None:
                cross = ""
                if Kp and Dp:
                    if Kp < Dp and K_v > D_v: cross = "🌟 黃金交叉"
                    elif Kp > Dp and K_v < D_v: cross = "❌ 死亡交叉"
                zone = "過熱" if K_v>80 else ("超賣" if K_v<20 else "正常")
                zone_c = "#ff5252" if K_v>80 else ("#00e676" if K_v<20 else "#ffd740")
                st.markdown(f"""<div style="background:#1e2130;border-radius:8px;padding:10px;border:1px solid #2d3148">
                <div style="font-size:0.8em;color:#8899bb">KD 隨機指標</div>
                <div style="color:#fff">K: <b>{K_v:.1f}</b> &nbsp; D: <b>{D_v:.1f}</b></div>
                <div style="color:#ab47bc">J: <b>{J_v:.1f}</b></div>
                <div style="color:{zone_c};font-size:0.85em">{zone} {cross}</div>
                </div>""", unsafe_allow_html=True)

        # RSI 卡
        rsi_v = gv("rsi")
        with ind2_cols[1]:
            if rsi_v is not None:
                if rsi_v>80:   rzone,rc="過熱 ⚠️","#ff5252"
                elif rsi_v>70: rzone,rc="偏高","#ff9100"
                elif rsi_v<30: rzone,rc="超賣 📌","#00e676"
                elif rsi_v<40: rzone,rc="偏低","#69f0ae"
                else:          rzone,rc="健康","#ffd740"
                pct = int(rsi_v)
                st.markdown(f"""<div style="background:#1e2130;border-radius:8px;padding:10px;border:1px solid #2d3148">
                <div style="font-size:0.8em;color:#8899bb">RSI 14</div>
                <div style="font-size:1.4em;color:#ab47bc;font-weight:700">{rsi_v:.1f}</div>
                <div style="background:#2d3148;border-radius:4px;height:6px;margin:4px 0">
                  <div style="background:{rc};width:{pct}%;height:6px;border-radius:4px"></div></div>
                <div style="color:{rc};font-size:0.85em">{rzone}</div>
                </div>""", unsafe_allow_html=True)

        # MACD 卡
        macd_v = gv("macd"); sig_v = gv("macd_signal"); hist_v = gv("macd_hist")
        macd_p = gvp("macd_hist")
        with ind2_cols[0]:
            if macd_v is not None:
                trend = ""
                if hist_v is not None and macd_p is not None:
                    trend = "柱體放大 ↑" if hist_v > macd_p else "柱體縮小 ↓"
                above = macd_v > sig_v if sig_v else None
                color = "#00e676" if (above or False) else "#ff5252"
                st.markdown(f"""<div style="background:#1e2130;border-radius:8px;padding:10px;border:1px solid #2d3148">
                <div style="font-size:0.8em;color:#8899bb">MACD (12/26/9)</div>
                <div style="color:#ef5350">MACD: <b>{macd_v:.3f}</b></div>
                <div style="color:#42a5f5">Signal: <b>{sig_v:.3f}</b></div>
                <div style="color:{color};font-size:0.85em">{'多頭' if above else '空頭'} ｜ {trend}</div>
                </div>""", unsafe_allow_html=True)

        # 布林通道卡
        bb_u = gv("bb_upper"); bb_m = gv("bb_middle"); bb_l = gv("bb_lower")
        with ind2_cols[1]:
            if bb_u and bb_l and price:
                bw = bb_u - bb_l
                pos = (price - bb_l) / bw * 100 if bw > 0 else 50
                if pos>90:   bpos,bc="接近上軌 ⚠️","#ff5252"
                elif pos>70: bpos,bc="偏上","#ff9100"
                elif pos<10: bpos,bc="接近下軌 📌","#00e676"
                elif pos<30: bpos,bc="偏下","#69f0ae"
                else:        bpos,bc="中性","#ffd740"
                st.markdown(f"""<div style="background:#1e2130;border-radius:8px;padding:10px;border:1px solid #2d3148">
                <div style="font-size:0.8em;color:#8899bb">布林通道 (20,2)</div>
                <div style="color:#ff7043">上: <b>{bb_u:.2f}</b></div>
                <div style="color:#90a4ae">中: <b>{bb_m:.2f}</b></div>
                <div style="color:#42a5f5">下: <b>{bb_l:.2f}</b></div>
                <div style="color:{bc};font-size:0.85em">位置 {pos:.0f}% ｜ {bpos}</div>
                </div>""", unsafe_allow_html=True)

        # 近期高低
        st.markdown("")
        h20v = gv("high_20"); lo20v = gv("low_20")
        h60v = gv("high_60"); lo60v = gv("low_60")
        hl_cols = st.columns(2)
        def hl_card(label, val, is_high):
            if val and price:
                diff = (price - val) / val * 100
                color = "#ff5252" if (is_high and diff > -5) else ("#00e676" if (not is_high and diff < 5) else "#ffd740")
                return f"""<div style="background:#1e2130;border-radius:8px;padding:10px;text-align:center;border:1px solid #2d3148">
                <div style="font-size:0.8em;color:#8899bb">{label}</div>
                <div style="font-size:1.1em;color:#fff;font-weight:600">{val:.2f}</div>
                <div style="font-size:0.85em;color:{color}">{diff:+.1f}%</div></div>"""
            return f"""<div style="background:#1e2130;border-radius:8px;padding:10px;text-align:center"><div style="color:#555">{label}: -</div></div>"""
        with hl_cols[0]:
            st.markdown(hl_card("20日最高",h20v,True),  unsafe_allow_html=True)
            st.markdown(hl_card("60日最高",h60v,True),  unsafe_allow_html=True)
        with hl_cols[1]:
            st.markdown(hl_card("20日最低",lo20v,False), unsafe_allow_html=True)
            st.markdown(hl_card("60日最低",lo60v,False), unsafe_allow_html=True)

    # ── 互動圖表 ──────────────────────────────────────────────
    st.divider()
    st.markdown("**互動圖表**")

    if df is None or df.empty or len(df) < 5:
        st.warning("歷史資料不足，無法顯示圖表。")
    else:
        # 選擇顯示區間
        period_map = {"1個月": 22, "3個月": 66, "6個月": 132, "1年": 252, "全部": len(df)}
        period_sel = st.radio("顯示區間", list(period_map.keys()), index=3,
                              horizontal=True, key="chart_period")
        n = min(period_map[period_sel], len(df))
        plot_df = df.iloc[-n:].copy()

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.40, 0.13, 0.15, 0.15, 0.17],
            subplot_titles=["K線 + 均線 + 布林", "成交量", "KD", "RSI", "MACD"],
        )

        x = plot_df.index

        # ── Row 1：K線 + SMA + Bollinger ──────────────────────
        fig.add_trace(go.Candlestick(
            x=x, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"],
            name="K線",
            increasing_line_color="#ff4b4b",
            decreasing_line_color="#00c853",
        ), row=1, col=1)

        sma_colors = {"sma5": "#ffeb3b", "sma20": "#29b6f6",
                      "sma60": "#ab47bc", "sma120": "#ff7043"}
        sma_labels = {"sma5": "SMA5", "sma20": "SMA20",
                      "sma60": "SMA60", "sma120": "SMA120"}
        for col_n, color in sma_colors.items():
            if col_n in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=plot_df[col_n], name=sma_labels[col_n],
                    line=dict(color=color, width=1.2), opacity=0.85,
                ), row=1, col=1)

        if "bb_upper" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["bb_upper"], name="BB上",
                line=dict(color="#90a4ae", width=1, dash="dot"), opacity=0.7,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["bb_lower"], name="BB下",
                line=dict(color="#90a4ae", width=1, dash="dot"), opacity=0.7,
                fill="tonexty", fillcolor="rgba(144,164,174,0.07)",
            ), row=1, col=1)

        # ── Row 2：成交量 ────────────────────────────────────
        vol_colors = ["#ff4b4b" if c >= o else "#00c853"
                      for c, o in zip(plot_df["Close"], plot_df["Open"])]
        fig.add_trace(go.Bar(
            x=x, y=plot_df["Volume"], name="量",
            marker_color=vol_colors, opacity=0.75,
        ), row=2, col=1)
        if "vol_ma5" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["vol_ma5"], name="量MA5",
                line=dict(color="#ffeb3b", width=1),
            ), row=2, col=1)
        if "vol_ma20" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["vol_ma20"], name="量MA20",
                line=dict(color="#ef5350", width=1, dash="dot"),
            ), row=2, col=1)

        # ── Row 3：KD ────────────────────────────────────────
        if "K" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["K"], name="K",
                line=dict(color="#ef5350", width=1.3),
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["D"], name="D",
                line=dict(color="#42a5f5", width=1.3),
            ), row=3, col=1)
        if "J" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["J"], name="J",
                line=dict(color="#ab47bc", width=1, dash="dot"),
                opacity=0.7,
            ), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="#ff7043",
                      line_width=0.8, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#607d8b",
                      line_width=0.6, row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="#66bb6a",
                      line_width=0.8, row=3, col=1)

        # ── Row 4：RSI ───────────────────────────────────────
        if "rsi" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["rsi"], name="RSI",
                line=dict(color="#ab47bc", width=1.3),
            ), row=4, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="#ff7043",
                      line_width=0.8, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#607d8b",
                      line_width=0.6, row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#66bb6a",
                      line_width=0.8, row=4, col=1)

        # ── Row 5：MACD ──────────────────────────────────────
        if "macd" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["macd"], name="MACD",
                line=dict(color="#ef5350", width=1.2),
            ), row=5, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=plot_df["macd_signal"], name="Signal",
                line=dict(color="#42a5f5", width=1.2),
            ), row=5, col=1)
            hist_colors = [
                "#ff4b4b" if v >= 0 else "#00c853"
                for v in plot_df["macd_hist"].fillna(0)
            ]
            fig.add_trace(go.Bar(
                x=x, y=plot_df["macd_hist"], name="Hist",
                marker_color=hist_colors, opacity=0.7,
            ), row=5, col=1)

        fig.update_layout(
            height=900,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#ccc", size=11),
            legend=dict(orientation="h", y=1.02, x=0, font_size=10),
            margin=dict(l=50, r=30, t=60, b=30),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        for i in range(1, 6):
            fig.update_yaxes(gridcolor="#1e2130", gridwidth=0.5, row=i, col=1)
            fig.update_xaxes(gridcolor="#1e2130", gridwidth=0.5, row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

