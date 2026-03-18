#!/usr/bin/env python3
"""
Backtest Polymarket Predictions
Reads unverified rows from the Prediction Accuracy sheet, fetches end prices via yfinance,
and marks predictions as correct/incorrect. If end time is in the future (or too recent),
the row is skipped (pending).
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import pytz
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time

# ---------- Configuration ----------
SPREADSHEET_NAME = os.environ.get("SPREADSHEET_NAME", "PolymarketPredictions")
CREDENTIALS_FILE = "google-sheets-credentials.json"   # same as bot
# -----------------------------------

def parse_datetime(dt_str: str) -> datetime:
    """Parse strings like '2026-03-12 16:56 UTC' into a naive UTC datetime."""
    # Remove trailing ' UTC' if present
    dt_str = dt_str.replace(' UTC', '')
    return datetime.strptime(dt_str, '%Y-%m-%d %H:%M')

def get_end_price(symbol: str, end_time: datetime, is_hourly: bool):
    """
    Fetch the price at the exact end_time using yfinance.
    For hourly (crypto): use 1h data, end_time is at a full hour UTC.
    For daily:
      - Crypto: end_time is midnight UTC, use daily data.
      - Stocks/Indices: end_time is 4pm ET in UTC, convert to ET date and get daily close.
    Returns price as float, or None if not found.
    """
    try:
        ticker = yf.Ticker(symbol)
        if is_hourly:
            # Get a few days of hourly data to ensure we have the candle
            data = ticker.history(period="5d", interval="1h")
            if data.empty:
                return None
            # yfinance hourly index is timezone-aware (UTC) for crypto
            # Make end_time timezone-aware for comparison
            end_utc = end_time.replace(tzinfo=pytz.UTC)
            # Ensure data index is UTC-aware
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            else:
                data.index = data.index.tz_convert('UTC')
            # Find the exact timestamp (or closest)
            # Use get_indexer with nearest to handle slight mismatches
            idx = data.index.get_indexer([end_utc], method='nearest')
            if idx[0] != -1:
                # Verify it's within a reasonable tolerance (e.g., 1 hour)
                closest_time = data.index[idx[0]]
                if abs((closest_time - end_utc).total_seconds()) < 3600:  # within 1 hour
                    return data.iloc[idx[0]]['Close']
            return None
        else:
            # Daily prediction
            # Check if symbol is crypto (ends with -USD or similar)
            if symbol.endswith('-USD') or symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']:
                # Crypto daily: end_time is midnight UTC. Get daily data for that date.
                data = ticker.history(period="5d", interval="1d")
                if data.empty:
                    return None
                target_date = end_time.date()
                # Daily index is date (timezone-naive) at 00:00 of that day in UTC? Actually yfinance daily is date only.
                for idx, row in data.iterrows():
                    # idx is a Timestamp; get date part
                    if idx.date() == target_date:
                        return row['Close']
                return None
            else:
                # Stocks/Indices: end_time is 4pm ET in UTC.
                # Convert to Eastern Time date.
                et = pytz.timezone('America/New_York')
                end_et = end_time.replace(tzinfo=pytz.UTC).astimezone(et)
                target_date = end_et.date()
                data = ticker.history(period="5d", interval="1d")
                if data.empty:
                    return None
                for idx, row in data.iterrows():
                    if idx.date() == target_date:
                        return row['Close']
                return None
    except Exception as e:
        print(f"Error fetching price for {symbol} at {end_time}: {e}")
        return None

def main():
    # Google Sheets authentication
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SPREADSHEET_NAME)
        ws = sheet.worksheet("Prediction Accuracy")
    except Exception as e:
        print(f"Failed to open sheet: {e}")
        sys.exit(1)

    # Get all records
    all_rows = ws.get_all_values()
    if len(all_rows) < 2:
        print("No data rows.")
        return

    headers = all_rows[0]
    # Find column indices
    col_map = {name: idx for idx, name in enumerate(headers)}
    required_cols = ['Symbol', 'Prediction', 'Start Time', 'Start Price', 'End Time',
                     'Actual Outcome', 'Correct', 'Verified Date']
    for col in required_cols:
        if col not in col_map:
            print(f"Column '{col}' not found in sheet. Has the bot been updated?")
            return

    # Current UTC time (naive) – we'll use it for comparisons
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        # --- DEBUG: Print current time and first end time ---
    print(f"Current UTC time (naive): {now_utc}")
    if len(all_rows) > 1:
        first_end = parse_datetime(all_rows[1][col_map['End Time']])
        print(f"First data row end time: {first_end}")
        print(f"End time > now? {first_end > now_utc}")
    # ----------------------------------------------------
    # Optional debug: uncomment to see current time and first end time
    # print(f"Current UTC time (naive): {now_utc}")
    # if len(all_rows) > 1:
    #     print(f"Row 2 end time: {parse_datetime(all_rows[1][col_map['End Time']])}")

    # Process each data row (skip header)
    for i, row in enumerate(all_rows[1:], start=2):  # row numbers are 1-indexed in gspread
        # Check if already verified
        if row[col_map['Verified Date']].strip():
            continue

        symbol = row[col_map['Symbol']]
        pred_str = row[col_map['Prediction']]
        start_time_str = row[col_map['Start Time']]
        start_price_str = row[col_map['Start Price']]
        end_time_str = row[col_map['End Time']]

        if not all([symbol, pred_str, start_time_str, start_price_str, end_time_str]):
            print(f"Skipping row {i}: missing data")
            continue

        try:
            start_price = float(start_price_str.replace('$', '').replace(',', ''))
        except:
            print(f"Row {i}: invalid start price '{start_price_str}'")
            continue

        # Parse times
        try:
            start_dt = parse_datetime(start_time_str)
            end_dt = parse_datetime(end_time_str)
        except Exception as e:
            print(f"Row {i}: failed to parse datetime: {e}")
            continue

        # Check if end time is in the future or too recent (allow 5 minutes for data availability)
        if end_dt > now_utc + timedelta(minutes=5):
            print(f"Row {i}: {symbol} -> Pending (future end time: {end_dt})")
            continue

        # Determine if hourly (based on time difference)
        delta = end_dt - start_dt
        is_hourly = abs(delta.total_seconds() - 3600) < 60  # within 1 minute of 1 hour

        # Fetch end price
        end_price = get_end_price(symbol, end_dt, is_hourly)
        if end_price is None:
            print(f"Row {i}: could not fetch end price for {symbol} at {end_dt}")
            continue

        # Compute actual direction
        ret = (end_price - start_price) / start_price
        actual = "UP" if ret > 0 else "DOWN"

        # Determine correctness
        if ("BULLISH" in pred_str and actual == "UP") or ("BEARISH" in pred_str and actual == "DOWN"):
            correct = "TRUE"
        else:
            correct = "FALSE"

        verified = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Update the row
        update_cols = {
            'Actual Outcome': actual,
            'Correct': correct,
            'Verified Date': verified
        }
        # Prepare cell updates
        cells = []
        for col_name, value in update_cols.items():
            col_idx = col_map[col_name] + 1  # gspread columns are 1-indexed
            cells.append({
                'range': f'{chr(64+col_idx)}{i}',
                'values': [[value]]
            })
        # Batch update
        if cells:
            ws.batch_update(cells)

        print(f"Row {i}: {symbol} -> {actual} ({correct})")

        # Small delay to avoid rate limits
        time.sleep(0.5)

    print("Backtesting completed.")

if __name__ == "__main__":
    main()