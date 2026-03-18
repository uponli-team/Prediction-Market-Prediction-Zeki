#!/usr/bin/env python3
"""
Polymarket Alpha Prediction Bot v7.8
- Multi‑timeframe price analysis (1,2,3d,1w,1m,6h)
- Enhanced rebound detection
- Correct end times for easy tracking (next hour, midnight UTC, next trading day 4pm ET)
- Backtesting support: added symbol, start price to accuracy sheet
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from transformers import pipeline
from newsapi import NewsApiClient
import calendar
import warnings
import pytz
from zoneinfo import ZoneInfo  # Python 3.9+
warnings.filterwarnings('ignore')

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import *

# ============================================
# CONFIGURATION
# ============================================

GMAIL_USER = "zekisebsib@gmail.com"
RECIPIENT_EMAILS = ["uponli.team@gmail.com", "zekariassebsib11@gmail.com"]
BCC_RECIPIENTS = False
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
SPREADSHEET_NAME = os.environ.get("SPREADSHEET_NAME", "PolymarketPredictions")
SHEET_CREDENTIALS_FILE = "google-sheets-credentials.json"

# Crypto expiry detection
def is_crypto_expiry_today() -> bool:
    """Check if today is a major monthly/quarterly crypto options expiry."""
    today = datetime.now(timezone.utc).date()          # ← use UTC date
    c = calendar.monthcalendar(today.year, today.month)
    fridays = [week[4] for week in c if week[4] != 0]
    if len(fridays) >= 3 and today.day == fridays[2]:
        return True
    if today.month in [3,6,9,12] and today.day == calendar.monthrange(today.year, today.month)[1]:
        return True
    return False

# ============================================
# POLYMARKET API
# ============================================

class PolymarketAPI:
    def __init__(self):
        self.gamma_api = "https://gamma-api.polymarket.com"

    async def fetch_crypto_markets(self, asset: str, min_volume: float = 500_000, min_duration_hours: float = 1) -> List[Dict]:
        markets = []
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.gamma_api}/markets"
                params = {
                    "limit": 50,
                    "closed": "false",
                    "order": "volume",
                    "volume_min": min_volume,
                    "tag": "crypto",
                    "search": asset
                }
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        now = datetime.now(timezone.utc)
                        for m in data:
                            end_str = m.get('endDate')
                            if not end_str:
                                continue
                            end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                            if (end - now).total_seconds() < min_duration_hours * 3600:
                                continue
                            # Ensure outcomes and prices exist and are lists
                            outcomes = m.get('outcomes', [])
                            prices = m.get('prices', [])
                            if not outcomes or not prices:
                                continue
                            markets.append({
                                "id": m.get('id'),
                                "title": m.get('question'),
                                "volume": float(m.get('volume', 0)),
                                "end_date": end_str,
                                "outcomes": outcomes,
                                "prices": [float(p) for p in prices],
                                "url": f"https://polymarket.com/market/{m.get('slug', '')}"
                            })
        except Exception as e:
            print(f"⚠️ Polymarket API error: {e}")
        return markets

    async def get_best_odds(self, asset: str, direction: str = "Up") -> float:
        markets = await self.fetch_crypto_markets(asset, min_volume=200_000, min_duration_hours=1)
        if not markets:
            return 0.5
        best = max(markets, key=lambda m: m['volume'])
        try:
            idx = best['outcomes'].index(direction)
            # Safety check: ensure idx is within prices list bounds
            if 0 <= idx < len(best['prices']):
                return best['prices'][idx]
            else:
                print(f"⚠️ Mismatch: outcomes and prices length for {best.get('title')}")
                return 0.5
        except ValueError:
            return 0.5

# ============================================
# NEWS SENTIMENT ANALYZER
# ============================================

class NewsSentimentAnalyzer:
    def __init__(self):
        self.newsapi_key = NEWSAPI_KEY
        if self.newsapi_key:
            try:
                self.newsapi = NewsApiClient(api_key=self.newsapi_key)
                print("✅ NewsAPI initialized")
            except:
                self.newsapi = None
        else:
            self.newsapi = None
            print("⚠️ NewsAPI key missing – using neutral sentiment")

        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            self.use_transformers = True
            print("✅ FinBERT loaded")
        except:
            self.use_transformers = False
            print("⚠️ FinBERT not available – using TextBlob")

    async def get_market_sentiment(self) -> float:
        """Return a sentiment score between -1 (bearish) and 1 (bullish)."""
        if not self.newsapi:
            return 0.0

        keywords = ["war", "tariff", "federal reserve", "inflation", "interest rates", "geopolitical"]
        total_score = 0.0
        count = 0

        for kw in keywords:
            try:
                articles = self.newsapi.get_everything(q=kw, language='en', page_size=10)
                if articles['status'] == 'ok':
                    for article in articles['articles']:
                        title = article['title']
                        if self.use_transformers:
                            result = self.sentiment_pipeline(title[:512])[0]
                            score = result['score'] if result['label'] == 'positive' else -result['score']
                        else:
                            blob = TextBlob(title)
                            score = blob.sentiment.polarity
                        total_score += score
                        count += 1
                await asyncio.sleep(0.2)
            except Exception as e:
                print(f"⚠️ News fetch error for {kw}: {e}")
                continue

        if count == 0:
            return 0.0
        return total_score / count

# ============================================
# UNIFIED PREDICTION ENGINE with Rebound Detection & Multi‑Timeframe
# ============================================

class UnifiedPredictor:
    def __init__(self, polymarket_api: PolymarketAPI, news_analyzer: NewsSentimentAnalyzer):
        self.poly_api = polymarket_api
        self.news_analyzer = news_analyzer
        self._daily_cache = {}      # symbol -> daily DataFrame (1mo)
        self._hourly_cache = {}     # symbol -> hourly DataFrame (5d)

    async def predict_hourly(self, symbol: str, name: str) -> Optional[Dict]:
        """Crypto 1‑hour prediction."""
        ticker = yf.Ticker(symbol)
        # Fetch 5 days of hourly data for RSI and 6h return
        hist = ticker.history(period="5d", interval="1h")
        if hist.empty:
            return None
        return await self._compute_prediction(hist, symbol, name, "crypto", "1h")

    async def predict_daily(self, symbol: str, name: str) -> Optional[Dict]:
        """Crypto 1‑day prediction."""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo", interval="1d")
        if hist.empty:
            return None
        return await self._compute_prediction(hist, symbol, name, "crypto", "1d")

    async def predict_index(self, symbol: str, name: str) -> Optional[Dict]:
        """Stock index prediction (next trading day)."""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo", interval="1d")
        if hist.empty:
            return None
        return await self._compute_prediction(hist, symbol, name, "index", "1d")

    async def predict_stock(self, symbol: str, name: str) -> Optional[Dict]:
        """Major stock prediction (next trading day)."""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo", interval="1d")
        if hist.empty:
            return None
        return await self._compute_prediction(hist, symbol, name, "stock", "1d")

    async def _compute_prediction(self, hist: pd.DataFrame, symbol: str, name: str,
                                  asset_type: str, timeframe: str) -> Dict:
        """
        Core prediction logic.
        - hist: DataFrame with the primary interval (hourly for 1h, daily for 1d)
        - timeframe: "1h" or "1d"
        """
        current_price = hist['Close'].iloc[-1]

        # ---- Fetch additional historical data (cached) ----
        ticker = yf.Ticker(symbol)

        # Daily data (1 month) for longer‑term metrics
        if symbol not in self._daily_cache:
            self._daily_cache[symbol] = ticker.history(period="1mo", interval="1d")
        hist_daily = self._daily_cache[symbol]
        if hist_daily.empty:
            hist_daily = hist  # fallback to the provided histogram if daily is empty

        # Hourly data (5 days) for 6h metrics (only for crypto 1h)
        hist_hourly = None
        if asset_type == "crypto" and timeframe == "1h":
            if symbol not in self._hourly_cache:
                self._hourly_cache[symbol] = ticker.history(period="5d", interval="1h")
            hist_hourly = self._hourly_cache[symbol]
        # -------------------------------------------------

        # ---- Compute multi‑timeframe lows, highs, returns ----
        # Daily based
        low_1m = hist_daily['Low'].min()
        high_1m = hist_daily['High'].max()
        low_1w = hist_daily['Low'].tail(7).min()
        high_1w = hist_daily['High'].tail(7).max()
        low_3d = hist_daily['Low'].tail(3).min()
        high_3d = hist_daily['High'].tail(3).max()
        low_1d = hist_daily['Low'].iloc[-1] if len(hist_daily) > 0 else current_price
        high_1d = hist_daily['High'].iloc[-1] if len(hist_daily) > 0 else current_price

        # Returns using daily closes
        closes_daily = hist_daily['Close']
        ret_1d = ((closes_daily.iloc[-1] / closes_daily.iloc[-2]) - 1) * 100 if len(closes_daily) >= 2 else 0
        ret_2d = ((closes_daily.iloc[-1] / closes_daily.iloc[-3]) - 1) * 100 if len(closes_daily) >= 3 else 0
        ret_3d = ((closes_daily.iloc[-1] / closes_daily.iloc[-4]) - 1) * 100 if len(closes_daily) >= 4 else 0
        ret_1w = ((closes_daily.iloc[-1] / closes_daily.iloc[-7]) - 1) * 100 if len(closes_daily) >= 8 else 0
        ret_1m = ((closes_daily.iloc[-1] / closes_daily.iloc[-30]) - 1) * 100 if len(closes_daily) >= 31 else 0

        # 6‑hour return (if hourly data available)
        ret_6h = 0
        if hist_hourly is not None and len(hist_hourly) >= 6:
            ret_6h = ((hist_hourly['Close'].iloc[-1] / hist_hourly['Close'].iloc[-6]) - 1) * 100

        # Existing indicators from primary hist
        rsi = self._calculate_rsi(hist['Close'])
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
        recent_low = hist['Low'].tail(20).min()
        recent_high = hist['High'].tail(20).max()
        ret_last = (current_price / hist['Close'].iloc[-2] - 1) * 100 if len(hist) > 1 else 0

        # ---- Rebound detection (uses multi‑timeframe data) ----
        rebound = self._calculate_rebound(current_price, low_1m, low_1w, low_3d, rsi,
                                          ret_1d, ret_1w, ret_1m)

        # ---- Polymarket odds & news ----
        poly_prob = 0.5
        if asset_type == "crypto":
            poly_prob = await self.poly_api.get_best_odds(name, "Up")
        news_score = await self.news_analyzer.get_market_sentiment()
        expiry_penalty = -0.2 if (asset_type == "crypto" and is_crypto_expiry_today()) else 0.0

        # ---- Enhanced technical score ----
        tech_score = 0.0

        # RSI
        if rsi < 30:
            tech_score += 0.3
        elif rsi > 70:
            tech_score -= 0.3
        else:
            tech_score += (50 - rsi) / 100

        # SMA position
        if current_price > sma_20:
            tech_score += 0.2
        else:
            tech_score -= 0.2
        if sma_50 and current_price > sma_50:
            tech_score += 0.1
        elif sma_50:
            tech_score -= 0.1

        # 20‑day range position
        range_pct = (current_price - recent_low) / (recent_high - recent_low) * 100
        if range_pct < 30:
            tech_score += 0.2
        elif range_pct > 70:
            tech_score -= 0.2

        # Distance to 1‑month low (oversold)
        dist_1m_low = ((current_price - low_1m) / low_1m) * 100 if low_1m > 0 else 999
        if dist_1m_low < 5:
            tech_score += 0.3
        elif dist_1m_low < 10:
            tech_score += 0.2
        elif dist_1m_low < 15:
            tech_score += 0.1

        # Distance to 1‑month high (overbought)
        dist_1m_high = ((high_1m - current_price) / current_price) * 100 if current_price > 0 else 0
        if dist_1m_high < 5:
            tech_score -= 0.3
        elif dist_1m_high < 10:
            tech_score -= 0.2
        elif dist_1m_high < 15:
            tech_score -= 0.1

        # Multi‑timeframe returns
        if ret_2d < -5:      tech_score += 0.2   # strong 2‑day drop
        elif ret_2d > 5:     tech_score -= 0.2   # strong 2‑day rise
        if ret_1w < -10:     tech_score += 0.3   # weekly drop
        elif ret_1w > 10:    tech_score -= 0.3
        if ret_1m < -15:     tech_score += 0.4   # monthly drop
        elif ret_1m > 15:    tech_score -= 0.4

        # 6‑hour momentum (only for crypto 1h)
        if ret_6h < -3:      tech_score += 0.1
        elif ret_6h > 3:     tech_score -= 0.1

        # Recent daily change (keep original weight)
        if ret_last > 2:
            tech_score += 0.2
        elif ret_last < -2:
            tech_score -= 0.2

        tech_score = max(min(tech_score, 1.0), -1.0)

        # ---- Combine scores ----
        poly_score = (poly_prob - 0.5) * 2
        final_score = (0.6 * tech_score) + (0.2 * poly_score) + (0.2 * news_score) + expiry_penalty
        final_score = max(min(final_score, 1.0), -1.0)

        prob_bullish = 1.0 / (1.0 + np.exp(-5.0 * final_score))
        confidence_pct = prob_bullish * 100.0

        prediction = "BULLISH 🚀" if confidence_pct >= 60.0 else "BEARISH 🔻"

        # ---- Time boundaries based on asset & timeframe ----
        now = datetime.now(timezone.utc)            # ← use UTC now
        start_time = now.strftime('%Y-%m-%d %H:%M UTC')

        if timeframe == "1h":
            # Next full hour (top of the hour)
            end_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            end_str = end_time.strftime('%Y-%m-%d %H:%M UTC')
        else:  # 1d predictions
            if asset_type == "crypto":
                # Crypto 1‑day: end at next midnight UTC
                end_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_str = end_time.strftime('%Y-%m-%d %H:%M UTC')
            else:  # indices or stocks
                # Next trading day at 4:00 PM Eastern Time (converted to UTC)
                end_time = self._next_trading_day_close_et()
                end_str = end_time.strftime('%Y-%m-%d %H:%M UTC')

        return {
            "name": name,
            "symbol": symbol,
            "current_price": f"${current_price:,.2f}",
            "daily_change": f"{ret_last:+.2f}%",
            "rsi": f"{rsi:.1f}",
            "poly_odds": f"{poly_prob*100:.1f}%",
            "news_score": f"{news_score:.2f}",
            "prediction": prediction,
            "confidence": f"{confidence_pct:.1f}%",
            "rebound": rebound,
            "start_time": start_time,
            "end_time": end_str
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
        return rsi

    def _calculate_rebound(self, price: float, low_1m: float, low_1w: float,
                           low_3d: float, rsi: float, ret_1d: float,
                           ret_1w: float, ret_1m: float) -> str:
        """
        Determine rebound potential based on multiple timeframes.
        Returns 'High', 'Medium', 'Low', or 'No'.
        """
        score = 0

        # Distance to 1‑month low
        dist_1m = ((price - low_1m) / low_1m) * 100 if low_1m > 0 else 999
        if dist_1m < 3:
            score += 3
        elif dist_1m < 7:
            score += 2
        elif dist_1m < 12:
            score += 1

        # Distance to 1‑week low
        dist_1w = ((price - low_1w) / low_1w) * 100 if low_1w > 0 else 999
        if dist_1w < 3:
            score += 2
        elif dist_1w < 7:
            score += 1

        # Distance to 3‑day low
        dist_3d = ((price - low_3d) / low_3d) * 100 if low_3d > 0 else 999
        if dist_3d < 2:
            score += 1

        # RSI oversold
        if rsi < 30:
            score += 3
        elif rsi < 40:
            score += 2
        elif rsi < 50:
            score += 1

        # Negative returns over different horizons
        if ret_1d < -3:
            score += 1
        if ret_1w < -8:
            score += 2
        if ret_1m < -15:
            score += 3

        # Classify
        if score >= 8:
            return "High"
        elif score >= 5:
            return "Medium"
        elif score >= 3:
            return "Low"
        else:
            return "No"

    def _next_trading_day_close_et(self) -> datetime:
        """
        Return a datetime (UTC) representing the next trading day's close at 4:00 PM Eastern Time.
        Uses zoneinfo for accurate DST.
        """
        now_et = datetime.now(ZoneInfo("America/New_York"))
        next_day = now_et + timedelta(days=1)

        # Skip weekends
        while next_day.weekday() >= 5:  # 5=Sat, 6=Sun
            next_day += timedelta(days=1)

        # Set to 4:00 PM ET
        close_et = next_day.replace(hour=16, minute=0, second=0, microsecond=0)

        # Convert to UTC
        close_utc = close_et.astimezone(timezone.utc)
        return close_utc.replace(tzinfo=None)  # return naive datetime for consistency

# ============================================
# GOOGLE SHEETS MANAGER (5 worksheets + Rebound column)
# ============================================

class GoogleSheetsManager:
    def __init__(self, cred_file: str, sheet_name: str):
        self.cred_file = cred_file
        self.sheet_name = sheet_name
        self.client = None
        self.sheet = None

    def initialize(self):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.cred_file, scope)
            self.client = gspread.authorize(creds)
            try:
                self.sheet = self.client.open(self.sheet_name)
            except gspread.SpreadsheetNotFound:
                self.sheet = self.client.create(self.sheet_name)
            self._create_worksheets()
            print(f"✅ Google Sheets ready: {self.sheet_name}")
            return True
        except Exception as e:
            print(f"⚠️ Sheets init error: {e}")
            return False

    def _create_worksheets(self):
        worksheets = {
            "Crypto 1-Hour": ["Timestamp", "Coin", "Price", "24h%", "RSI", "Poly Odds", "News Score", "Prediction", "Confidence", "Rebound", "Start Time", "End Time"],
            "Crypto 1-Day": ["Timestamp", "Coin", "Price", "24h%", "RSI", "Poly Odds", "News Score", "Prediction", "Confidence", "Rebound", "Start Time", "End Time"],
            "Stock Indices": ["Timestamp", "Index", "Price", "RSI", "News Score", "Prediction", "Confidence", "Rebound", "Start Time", "End Time"],
            "Major Stocks": ["Timestamp", "Symbol", "Price", "RSI", "News Score", "Prediction", "Confidence", "Rebound", "Start Time", "End Time"],
            "Prediction Accuracy": ["ID", "Timestamp", "Symbol", "Asset", "Prediction", "Confidence",
                                    "PolyOdds", "News Score", "Rebound", "Start Time", "Start Price",
                                    "End Time", "Actual Outcome", "Correct", "Verified Date"]
        }
        for name, headers in worksheets.items():
            try:
                self.sheet.worksheet(name)
            except:
                ws = self.sheet.add_worksheet(title=name, rows=1000, cols=len(headers))
                ws.append_row(headers)

    def save_prediction(self, worksheet_name: str, row: list):
        try:
            ws = self.sheet.worksheet(worksheet_name)
            # Ensure first column (Timestamp) is UTC
            row[0] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            ws.append_row(row)
        except Exception as e:
            print(f"⚠️ Error saving to {worksheet_name}: {e}")

    def save_accuracy(self, asset: str, symbol: str, prediction: str, confidence: str,
                      poly_odds: str, news_score: str, rebound: str,
                      start_time: str, start_price: str, end_time: str):
        try:
            ws = self.sheet.worksheet("Prediction Accuracy")
            now_utc = datetime.now(timezone.utc)
            pred_id = now_utc.strftime("%Y%m%d%H%M%S") + asset[:3]
            row = [pred_id,                                          # ID
                   now_utc.strftime("%Y-%m-%d %H:%M:%S"),           # Timestamp (recording time)
                   symbol,                                           # Symbol
                   asset,                                            # Asset (full name with type)
                   prediction,                                       # Prediction
                   confidence,                                       # Confidence
                   poly_odds,                                        # PolyOdds
                   news_score,                                       # News Score
                   rebound,                                          # Rebound
                   start_time,                                       # Start Time
                   start_price,                                      # Start Price
                   end_time,                                         # End Time
                   "", "", ""]                                       # Actual Outcome, Correct, Verified Date (empty)
            ws.append_row(row)
        except Exception as e:
            print(f"⚠️ Accuracy save error: {e}")

# ============================================
# MAIN BOT
# ============================================

class PolymarketBot:
    def __init__(self):
        self.poly_api = PolymarketAPI()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.predictor = UnifiedPredictor(self.poly_api, self.news_analyzer)
        self.sheets = GoogleSheetsManager(SHEET_CREDENTIALS_FILE, SPREADSHEET_NAME)

    async def run(self):
        print("\n" + "="*70)
        print("🚀 POLYMARKET BOT v7.8 – BACKTESTING READY")
        print("="*70)

        self.sheets.initialize()

        cryptos = [
            ("BTC-USD", "Bitcoin"),
            ("ETH-USD", "Ethereum"),
            ("SOL-USD", "Solana"),
            ("BNB-USD", "Binance Coin"),
            ("XRP-USD", "XRP"),
        ]
        indices = [
            ("^GSPC", "S&P 500"),
            ("^IXIC", "NASDAQ"),
            ("^DJI", "Dow Jones"),
        ]
        stocks = [
            ("AAPL", "Apple"),
            ("MSFT", "Microsoft"),
            ("GOOGL", "Google"),
            ("AMZN", "Amazon"),
            ("NVDA", "NVIDIA"),
        ]

        all_hourly = []
        all_daily = []
        index_preds = []
        stock_preds = []

        print("\n🔍 Analyzing cryptocurrencies (1-hour)...")
        for sym, name in cryptos:
            pred = await self.predictor.predict_hourly(sym, name)
            if pred:
                all_hourly.append(pred)
                row = ["", name, pred['current_price'],
                       pred['daily_change'], pred['rsi'], pred['poly_odds'], pred['news_score'],
                       pred['prediction'], pred['confidence'], pred['rebound'],
                       pred['start_time'], pred['end_time']]
                self.sheets.save_prediction("Crypto 1-Hour", row)
                self.sheets.save_accuracy(name + " (1h)", sym, pred['prediction'], pred['confidence'],
                                          pred['poly_odds'], pred['news_score'], pred['rebound'],
                                          pred['start_time'], pred['current_price'], pred['end_time'])

        print("\n🔍 Analyzing cryptocurrencies (1-day)...")
        for sym, name in cryptos:
            pred = await self.predictor.predict_daily(sym, name)
            if pred:
                all_daily.append(pred)
                row = ["", name, pred['current_price'],
                       pred['daily_change'], pred['rsi'], pred['poly_odds'], pred['news_score'],
                       pred['prediction'], pred['confidence'], pred['rebound'],
                       pred['start_time'], pred['end_time']]
                self.sheets.save_prediction("Crypto 1-Day", row)
                self.sheets.save_accuracy(name + " (1d)", sym, pred['prediction'], pred['confidence'],
                                          pred['poly_odds'], pred['news_score'], pred['rebound'],
                                          pred['start_time'], pred['current_price'], pred['end_time'])

        print("\n📈 Analyzing stock indices...")
        for sym, name in indices:
            pred = await self.predictor.predict_index(sym, name)
            if pred:
                index_preds.append(pred)
                row = ["", name, pred['current_price'],
                       pred['rsi'], pred['news_score'], pred['prediction'], pred['confidence'],
                       pred['rebound'], pred['start_time'], pred['end_time']]
                self.sheets.save_prediction("Stock Indices", row)
                self.sheets.save_accuracy(name, sym, pred['prediction'], pred['confidence'],
                                          "N/A", pred['news_score'], pred['rebound'],
                                          pred['start_time'], pred['current_price'], pred['end_time'])

        print("\n🏢 Analyzing major stocks...")
        for sym, name in stocks:
            pred = await self.predictor.predict_stock(sym, name)
            if pred:
                stock_preds.append(pred)
                row = ["", sym, pred['current_price'],
                       pred['rsi'], pred['news_score'], pred['prediction'], pred['confidence'],
                       pred['rebound'], pred['start_time'], pred['end_time']]
                self.sheets.save_prediction("Major Stocks", row)
                self.sheets.save_accuracy(name, sym, pred['prediction'], pred['confidence'],
                                          "N/A", pred['news_score'], pred['rebound'],
                                          pred['start_time'], pred['current_price'], pred['end_time'])

        # Generate and send email
        print("\n📧 Generating and sending email...")
        html = self._generate_email(all_hourly, all_daily, index_preds, stock_preds)
        with open("prediction_report.html", "w", encoding="utf-8") as f:
            f.write(html)

        if GMAIL_PASSWORD:
            self._send_email(html)
        else:
            print("⚠️ GMAIL_APP_PASSWORD not set, email not sent.")

        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        if all_hourly:
            print("1‑Hour Crypto:")
            for p in all_hourly[:3]:
                print(f"  {p['name']}: {p['prediction']} ({p['confidence']}) rebound {p['rebound']} ends {p['end_time']}")
        if all_daily:
            print("\n1‑Day Crypto:")
            for p in all_daily[:3]:
                print(f"  {p['name']}: {p['prediction']} ({p['confidence']}) rebound {p['rebound']} ends {p['end_time']}")
        if index_preds:
            print("\nStock Indices:")
            for p in index_preds[:3]:
                print(f"  {p['name']}: {p['prediction']} ({p['confidence']}) rebound {p['rebound']} ends {p['end_time']}")
        if stock_preds:
            print("\nMajor Stocks:")
            for p in stock_preds[:3]:
                print(f"  {p['name']}: {p['prediction']} ({p['confidence']}) rebound {p['rebound']} ends {p['end_time']}")
        print("\n✅ Data saved to Google Sheets. Check 'Prediction Accuracy' for later verification.")
        print("="*70)

    def _generate_email(self, hourly, daily, index_preds, stock_preds):
        now_utc = datetime.now(timezone.utc)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"><style>
            body {{ font-family: Arial; background: #f3f4f6; padding:20px; }}
            .container {{ max-width:1000px; margin:auto; background: #d1d5db; padding:20px; border-radius:10px; }}
            h1 {{ color:#4f46e5; }}
            h2 {{ color:#1f2937; border-left:4px solid #4f46e5; padding-left:10px; }}
            table {{ border-collapse: collapse; width:100%; background:white; margin:10px 0; }}
            th {{ background:#4f46e5; color:white; padding:8px; }}
            td {{ border:1px solid #ddd; padding:8px; }}
            .bullish {{ color:green; font-weight:bold; }}
            .bearish {{ color:red; font-weight:bold; }}
            .high-conf {{ background:#d1fae5; }}
            .rebound-high {{ background: #fef3c7; }}
            .rebound-medium {{ background: #e0f2fe; }}
            .rebound-low {{ background: #e5e7eb; }}
        </style></head>
        <body>
        <div class="container">
            <h1>🎯 Polymarket Alpha v7.8</h1>
            <p>{now_utc.strftime('%B %d, %Y %H:%M')} UTC</p>

            <h2>🪙 Crypto 1‑Hour Predictions</h2>
            <table>
                <tr><th>Coin</th><th>Price</th><th>RSI</th><th>Poly Odds</th><th>News</th><th>Prediction</th><th>Confidence</th><th>Rebound</th><th>Ends At</th></tr>
        """
        for p in hourly:
            cls = "bullish" if "BULLISH" in p['prediction'] else "bearish"
            conf_class = "high-conf" if float(p['confidence'].rstrip('%')) >= 60 else ""
            rebound_class = f"rebound-{p['rebound'].lower()}" if p['rebound'] != "No" else ""
            html += f"<tr><td>{p['name']}</td><td>{p['current_price']}</td><td>{p['rsi']}</td><td>{p['poly_odds']}</td><td>{p['news_score']}</td><td class='{cls}'>{p['prediction']}</td><td class='{conf_class}'>{p['confidence']}</td><td class='{rebound_class}'>{p['rebound']}</td><td>{p['end_time']}</td></tr>"
        html += "</table>"

        html += "<h2>🪙 Crypto 1‑Day Predictions</h2><table><tr><th>Coin</th><th>Price</th><th>RSI</th><th>Poly Odds</th><th>News</th><th>Prediction</th><th>Confidence</th><th>Rebound</th><th>Ends At</th></tr>"
        for p in daily:
            cls = "bullish" if "BULLISH" in p['prediction'] else "bearish"
            conf_class = "high-conf" if float(p['confidence'].rstrip('%')) >= 60 else ""
            rebound_class = f"rebound-{p['rebound'].lower()}" if p['rebound'] != "No" else ""
            html += f"<tr><td>{p['name']}</td><td>{p['current_price']}</td><td>{p['rsi']}</td><td>{p['poly_odds']}</td><td>{p['news_score']}</td><td class='{cls}'>{p['prediction']}</td><td class='{conf_class}'>{p['confidence']}</td><td class='{rebound_class}'>{p['rebound']}</td><td>{p['end_time']}</td></tr>"
        html += "</table>"

        html += "<h2>📈 Stock Indices (Next Trading Day)</h2><table><tr><th>Index</th><th>Price</th><th>RSI</th><th>News</th><th>Prediction</th><th>Confidence</th><th>Rebound</th><th>Ends At</th></tr>"
        for p in index_preds:
            cls = "bullish" if "BULLISH" in p['prediction'] else "bearish"
            conf_class = "high-conf" if float(p['confidence'].rstrip('%')) >= 60 else ""
            rebound_class = f"rebound-{p['rebound'].lower()}" if p['rebound'] != "No" else ""
            html += f"<tr><td>{p['name']}</td><td>{p['current_price']}</td><td>{p['rsi']}</td><td>{p['news_score']}</td><td class='{cls}'>{p['prediction']}</td><td class='{conf_class}'>{p['confidence']}</td><td class='{rebound_class}'>{p['rebound']}</td><td>{p['end_time']}</td></tr>"
        html += "</table>"

        html += "<h2>🏢 Major Stocks (Next Trading Day)</h2><table><tr><th>Stock</th><th>Price</th><th>RSI</th><th>News</th><th>Prediction</th><th>Confidence</th><th>Rebound</th><th>Ends At</th></tr>"
        for p in stock_preds:
            cls = "bullish" if "BULLISH" in p['prediction'] else "bearish"
            conf_class = "high-conf" if float(p['confidence'].rstrip('%')) >= 60 else ""
            rebound_class = f"rebound-{p['rebound'].lower()}" if p['rebound'] != "No" else ""
            html += f"<tr><td>{p['name']}</td><td>{p['current_price']}</td><td>{p['rsi']}</td><td>{p['news_score']}</td><td class='{cls}'>{p['prediction']}</td><td class='{conf_class}'>{p['confidence']}</td><td class='{rebound_class}'>{p['rebound']}</td><td>{p['end_time']}</td></tr>"
        html += "</table>"

        html += f"<p>📊 Google Sheet: {SPREADSHEET_NAME} (see 'Prediction Accuracy' for historical tracking)</p>"
        html += "<p style='color:#666;'>Sent to: " + ", ".join(RECIPIENT_EMAILS) + "</p>"
        html += "</div></body></html>"
        return html

    def _send_email(self, html_content: str):
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🎯 Polymarket Predictions - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
            msg['From'] = GMAIL_USER
            if BCC_RECIPIENTS:
                msg['To'] = GMAIL_USER
                msg['Bcc'] = ', '.join(RECIPIENT_EMAILS)
            else:
                msg['To'] = ', '.join(RECIPIENT_EMAILS)

            text = "Please view this email in HTML format."
            msg.attach(MIMEText(text, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))

            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"✅ Email sent to {', '.join(RECIPIENT_EMAILS)}")
        except Exception as e:
            print(f"❌ Email sending failed: {e}")

# ============================================
# MAIN
# ============================================

async def main():
    if not GMAIL_PASSWORD:
        print("❌ GMAIL_APP_PASSWORD environment variable not set.")
        return
    bot = PolymarketBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())