# 台股自選股看盤與短中長期評估系統

## 快速啟動

```bash
cd stock_dashboard
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器自動開啟 http://localhost:8501

## 功能說明

| 功能 | 說明 |
|------|------|
| 即時報價 | TWSE MIS API，非交易時間自動改用 yfinance 最近收盤 |
| 歷史K線 | yfinance 抓近 1 年日K |
| 技術指標 | SMA5/10/20/60/120、KD、RSI、MACD、布林通道、量MA |
| 評估評分 | 短期（1-10天）、中期（1-3月）、長期（6-12月）各 0-100 分 |
| 操作建議 | 可小量試單 / 等拉回 / 續抱 / 分批停利 / 不建議追價 / 停損 / 定期定額 |
| 今日提醒 | 接近漲停/跌停、爆量、KD交叉、RSI過熱、跳空、長上影線 |
| 自選股管理 | 新增/刪除，存 watchlist.json |

## 資料來源

- **即時報價**：[TWSE MIS API](https://mis.twse.com.tw/stock/api/getStockInfo.jsp)
- **歷史資料**：[yfinance](https://github.com/ranaroussi/yfinance)（台股代號加 `.TW` 或 `.TWO`）
- **預留**：FinMind API（法人買賣超、月營收、財報）

## 免責聲明

本系統僅提供資料整理與技術分析輔助，**不構成任何投資建議**。
投資人應自行判斷風險，盈虧自負。
