# Stocks & Options Board 開發者說明

這個小小的網站利用 FastAPI + Jinja2 組成簡潔的股票/選擇權看板。預設支援 AAPL、MSFT、NVDA… 等熱門股票，會自動調用 yfinance 與 OCC 的公開資訊取得最新報價、成交量、未平倉量與買賣價差，並緩存結果以降低外部呼叫次數。

## 如何運行

1. 確認你的系統已安裝 Python 3.9 以上版本。
2. 在專案根目錄安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
3. 啟動開發伺服器：
   ```bash
   uvicorn app:app --reload --port 8000
   ```
   `--reload` 會在你修改程式時自動重新載入，適合開發時使用。開啟瀏覽器訪問 http://localhost:8000 即可看到看板，http://localhost:8000/health 會回傳簡單的健康檢查訊息。

## 專案結構

- `app.py`：FastAPI 應用程式主入口，定義路由與頁面渲染邏輯。
- `templates/`：Jinja2 模板資料夾，儲存 HTML 頁面佈局。
- `static/`：靜態檔案（CSS、JS）存放處。
- `tests/`：簡易 smoke test，確保首頁能成功渲染，可以使用 `pytest -q` 執行。

## 開發小技巧

- 預設股票清單在 `DEFAULT_TICKERS` 變數，可依需求增減。
- `QUOTES_TTL_SEC` 與 `OPTIONS_TTL_SEC` 控制快取時間，避免頻繁呼叫外部 API。
- 若需要額外的報價來源，可在 `yahoo_quote_batch` 函式中擴充。

## 未來展望

這只是個起點：你可以加入技術指標、長短期移動平均、甚至即時警示。彈性設計讓你輕鬆迭代，讓看板成為專屬於你的投資儀表板。
