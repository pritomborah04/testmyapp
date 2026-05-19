import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import requests
import datetime
import time
import os
import sys
import json
import math
import concurrent.futures
import yfinance as yf
from flask import Flask, jsonify, request, render_template, session
from typing import Optional, Dict, List, Any, Tuple

YF_SYMBOLS = {
    'NIFTY': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'FINNIFTY': '^CNXFIN',
    'MIDCPNIFTY': '^NSEMDCP50',
    'NIFTYNXT50': '^CRSLDX',
}

app = Flask(__name__)
app.secret_key = os.urandom(24)
FIX_IV_FILE = 'fix_iv.json'
_gift_nifty_cache = {'data': None, 'timestamp': 0}
GIFT_NIFTY_CACHE_TTL = 10
_ai_history: Dict[str, Dict] = {}
_AI_HISTORY_MAX = 20


def load_fix_iv() -> Dict[str, Any]:
    if os.path.exists(FIX_IV_FILE):
        try:
            with open(FIX_IV_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'date': '', 'ce_iv': 0.0, 'pe_iv': 0.0}


def save_fix_iv(ce_iv: float, pe_iv: float) -> None:
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    with open(FIX_IV_FILE, 'w') as f:
        json.dump({'date': today, 'ce_iv': round(ce_iv, 2), 'pe_iv': round(pe_iv, 2)}, f)

red = "#e53935"
green = "#00e676"


class NseWeb:
    def __init__(self) -> None:
        self.session: requests.Session = requests.Session()
        self.headers: Dict[str, str] = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
            'accept-language': 'en,gu;q=0.9,hi;q=0.8',
            'accept-encoding': 'gzip, deflate, br'}
        self.cookies: Dict[str, str] = {}
        self.indices: List[str] = []
        self.stocks: List[str] = []
        self.url_oc: str = "https://www.nseindia.com/option-chain"
        self.url_symbols: str = "https://www.nseindia.com/api/underlying-information"
        self.url_index_data: str = "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={}&expiry={}"
        self.url_stock_data: str = "https://www.nseindia.com/api/option-chain-v3?type=Equity&symbol={}&expiry={}"
        self.url_vix: str = "https://www.nseindia.com/api/allIndices"
        self._index_cache: Dict[str, str] = {}
        self.get_symbols()

    def _refresh_session(self) -> None:
        self.session.close()
        self.session = requests.Session()
        r = self.session.get(self.url_oc, headers=self.headers, timeout=5)
        self.cookies = dict(r.cookies)

    def _fetch(self, url: str) -> Optional[requests.Response]:
        try:
            resp = self.session.get(url, headers=self.headers, timeout=5, cookies=self.cookies)
            if resp.status_code == 401:
                self._refresh_session()
                resp = self.session.get(url, headers=self.headers, timeout=5, cookies=self.cookies)
            return resp
        except Exception:
            try:
                self._refresh_session()
                resp = self.session.get(url, headers=self.headers, timeout=5, cookies=self.cookies)
                return resp
            except Exception:
                return None

    def get_symbols(self) -> None:
        try:
            r = self.session.get(self.url_oc, headers=self.headers, timeout=5)
            self.cookies = dict(r.cookies)
            resp = self.session.get(self.url_symbols, headers=self.headers, timeout=5, cookies=self.cookies)
            data = resp.json()
            self.indices = [item['symbol'] for item in data['data']['IndexList']]
            self.stocks = [item['symbol'] for item in data['data']['UnderlyingList']]
        except Exception as e:
            print(f"get_symbols error: {e}", file=sys.stderr)

    def get_expiry_dates(self, symbol: str, mode: str = 'Index') -> List[str]:
        url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}"
        resp = self._fetch(url)
        if resp is None:
            return []
        try:
            data = resp.json()
            return data.get('expiryDates', data.get('records', {}).get('expiryDates', []))
        except Exception:
            return []

    def get_vix(self) -> Optional[dict]:
        resp = self._fetch(self.url_vix)
        if resp is None or resp.status_code != 200:
            return None
        try:
            for item in resp.json().get('data', []):
                if 'INDIA VIX' in item.get('index', ''):
                    return {
                        'last': item.get('last'),
                        'change': item.get('change'),
                        'percentChange': item.get('percentChange')
                    }
        except Exception:
            pass
        return None

    def _resolve_index_name(self, symbol: str) -> str:
        if symbol in self._index_cache:
            return self._index_cache[symbol]
        try:
            resp = self._fetch(self.url_vix)
            if resp and resp.status_code == 200:
                sym_up = symbol.upper().replace(' ', '')
                for item in resp.json().get('data', []):
                    idx_up = item.get('index', '').upper().replace(' ', '')
                    if 'VIX' not in idx_up and (sym_up in idx_up or idx_up in sym_up):
                        name = item.get('index', symbol)
                        self._index_cache[symbol] = name
                        return name
        except Exception:
            pass
        self._index_cache[symbol] = symbol
        return symbol

    def fetch_intraday_1min(self, symbol: str, mode: str) -> Optional[pd.DataFrame]:
        try:
            if mode == 'Index':
                yf_sym = YF_SYMBOLS.get(symbol)
                if not yf_sym:
                    resolved = self._resolve_index_name(symbol)
                    for nse_sym, yf_ticker in YF_SYMBOLS.items():
                        if nse_sym in resolved.upper() or resolved.upper() in nse_sym:
                            yf_sym = yf_ticker
                            break
                    if not yf_sym:
                        return None
            else:
                yf_sym = f'{symbol}.NS'
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period='1d', interval='1m')
            if hist.empty or len(hist) < 5:
                return None
            return hist
        except Exception:
            return None

    @staticmethod
    def _calc_crossover_15m(hist: pd.DataFrame) -> Optional[Dict]:
        try:
            if hist is None or hist.empty or len(hist) < 30:
                return None
            df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.index = pd.to_datetime(df.index)
            resampled = df.resample('15min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            if len(resampled) < 6:
                return None
            closes = resampled['Close'].values
            def ema(vals, p):
                m = 2 / (p + 1)
                r = [sum(vals[:p]) / p]
                for v in vals[p:]:
                    r.append((v - r[-1]) * m + r[-1])
                return r
            ema5 = ema(closes, 5)[-1]
            ema15 = ema(closes, 15)[-1]
            prev_ema5 = ema(closes[:15], 5)[-1] if len(closes) > 15 else ema(closes[:-1], 5)[-1]
            prev_ema15 = ema(closes[:15], 15)[-2] if len(closes) > 16 else ema(closes[:-1], 15)[-1]
            signal = None
            if prev_ema5 <= prev_ema15 and ema5 > ema15:
                signal = 'CROSS UP'
            elif prev_ema5 >= prev_ema15 and ema5 < ema15:
                signal = 'CROSS DN'
            return {
                'signal': signal,
                'ema5': round(ema5, 2),
                'ema15': round(ema15, 2),
                'candles': len(resampled),
            }
        except Exception:
            return None

    @staticmethod
    def _calc_vwap(candles: List[Dict]) -> Optional[float]:
        try:
            pv = sum((c['high'] + c['low'] + c['close']) / 3 * c['volume'] for c in candles)
            vol = sum(c['volume'] for c in candles)
            return round(pv / vol, 2) if vol > 0 else None
        except Exception:
            return None

    @staticmethod
    def _calc_ema_cross(candles: List[Dict]) -> Optional[Dict]:
        closes = [c['close'] for c in candles]
        if len(closes) < 20:
            return None
        def ema(vals, p):
            m = 2 / (p + 1)
            r = [sum(vals[:p]) / p]
            for v in vals[p:]:
                r.append((v - r[-1]) * m + r[-1])
            return r
        e9 = ema(closes, 9)
        e19 = ema(closes, 19)
        sig = 'BULLISH' if e9[-1] > e19[-1] else 'BEARISH' if e9[-1] < e19[-1] else ''
        cross = ''
        if len(e9) > 1 and len(e19) > 1:
            if e9[-2] < e19[-2] and e9[-1] > e19[-1]:
                cross = 'CROSS UP'
            elif e9[-2] > e19[-2] and e9[-1] < e19[-1]:
                cross = 'CROSS DN'
        return {'ema9': round(e9[-1], 2), 'ema19': round(e19[-1], 2), 'signal': sig, 'cross': cross}

    @staticmethod
    def _calc_supertrend(candles: List[Dict]) -> Optional[Dict]:
        if len(candles) < 12:
            return None
        p, m = 10, 3
        h = [c['high'] for c in candles]
        l = [c['low'] for c in candles]
        c = [c['close'] for c in candles]
        tr = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(candles))]
        atr = [sum(tr[:p])/p]
        for i in range(p, len(tr)):
            atr.append((atr[-1]*9+tr[i])/p)
        st, ubv, lbv = [1], [0.0], [0.0]
        for i in range(p, len(candles)):
            a = atr[i-p]
            hl = (h[i]+l[i])/2
            ub, lb = hl+m*a, hl-m*a
            if st[-1] == 1:
                if c[i] < lbv[-1]:
                    st.append(-1); ubv.append(ub); lbv.append(lb)
                else:
                    st.append(1); ubv.append(max(ub, ubv[-1])); lbv.append(lb)
            else:
                if c[i] > ubv[-1]:
                    st.append(1); ubv.append(ub); lbv.append(lb)
                else:
                    st.append(-1); ubv.append(ub); lbv.append(min(lb, lbv[-1]))
        sig = 'BULLISH' if st[-1] == 1 else 'BEARISH'
        return {'signal': sig}

    @staticmethod
    def _calculate_adx(ohlc: List[Dict]) -> Optional[Dict]:
        if len(ohlc) < 30:
            return None
        n = len(ohlc)
        high = [d['high'] for d in ohlc]
        low = [d['low'] for d in ohlc]
        close = [d['close'] for d in ohlc]
        tr = []
        for i in range(1, n):
            tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
        plus_dm, minus_dm = [], []
        for i in range(1, n):
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            plus_dm.append(up if up > dn and up > 0 else 0.0)
            minus_dm.append(dn if dn > up and dn > 0 else 0.0)
        p = 14
        atr = [0.0] * len(tr)
        atr[p - 1] = sum(tr[:p]) / p
        for i in range(p, len(tr)):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / p
        pds = [0.0] * len(plus_dm)
        pds[p - 1] = sum(plus_dm[:p]) / p
        for i in range(p, len(plus_dm)):
            pds[i] = (pds[i - 1] * 13 + plus_dm[i]) / p
        mds = [0.0] * len(minus_dm)
        mds[p - 1] = sum(minus_dm[:p]) / p
        for i in range(p, len(minus_dm)):
            mds[i] = (mds[i - 1] * 13 + minus_dm[i]) / p
        pdi = [(pds[i] / atr[i] * 100) if atr[i] > 0 else 0.0 for i in range(len(atr))]
        ndi = [(mds[i] / atr[i] * 100) if atr[i] > 0 else 0.0 for i in range(len(atr))]
        dx = [abs(pdi[i] - ndi[i]) / (pdi[i] + ndi[i]) * 100 if (pdi[i] + ndi[i]) > 0 else 0.0 for i in range(len(pdi))]
        adx = [0.0] * len(dx)
        adx[p - 1] = sum(dx[:p]) / p
        for i in range(p, len(dx)):
            adx[i] = (adx[i - 1] * 13 + dx[i]) / p
        v = round(adx[-1], 2)
        last_pdi = pdi[-1] if len(pdi) > 0 else 0.0
        last_ndi = ndi[-1] if len(ndi) > 0 else 0.0
        if v >= 20 and last_pdi > last_ndi:
            trend_sig = 'UPTREND'
        elif v >= 20 and last_ndi > last_pdi:
            trend_sig = 'DOWNTREND'
        else:
            trend_sig = 'NO TREND'
        return {'adx': v, 'signal': trend_sig}

    def get_oc_raw(self, symbol: str, expiry: str, mode: str = 'Index') -> Optional[dict]:
        url = self.url_index_data.format(symbol, expiry) if mode == 'Index' else self.url_stock_data.format(symbol, expiry)
        resp = self._fetch(url)
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    def process_oc_data(self, json_data: dict, mode: str = 'Index', symbol: str = '') -> Optional[dict]:
        try:
            records = json_data.get('records', {})
            data_list = records.get('data', [])
            if not data_list:
                return None

            expiry_lower = (records.get('expiryDates') or json_data.get('expiryDates', ['']))[0].lower()

            ce_values = [d['CE'] for d in data_list if 'CE' in d]
            pe_values = [d['PE'] for d in data_list if 'PE' in d]

            if not ce_values or not pe_values:
                return None

            points = pe_values[0].get('underlyingValue', 0) or 0
            if points == 0:
                for item in pe_values:
                    if item.get('underlyingValue', 0) != 0:
                        points = item['underlyingValue']
                        break

            lot_size = json_data.get('meta', {}).get('marketLot') or records.get('marketLot', 1)

            ce_df = pd.DataFrame(ce_values)
            pe_df = pd.DataFrame(pe_values)

            cols_ce = ['openInterest', 'changeinOpenInterest', 'totalTradedVolume', 'impliedVolatility',
                       'lastPrice', 'change', 'buyQuantity1', 'buyPrice1', 'sellPrice1', 'sellQuantity1', 'strikePrice']
            cols_pe = ['strikePrice', 'buyQuantity1', 'buyPrice1', 'sellPrice1', 'sellQuantity1', 'change',
                       'lastPrice', 'impliedVolatility', 'totalTradedVolume', 'changeinOpenInterest', 'openInterest']

            ce_df = ce_df[cols_ce]
            pe_df = pe_df[cols_pe]

            merged = pd.merge(left=ce_df, right=pe_df, left_on='strikePrice', right_on='strikePrice')
            merged.columns = [
                'Open Interest', 'Change in Open Interest', 'Traded Volume', 'Implied Volatility',
                'Last Traded Price', 'Net Change', 'Bid Quantity', 'Bid Price', 'Ask Price',
                'Ask Quantity', 'Strike Price',
                'Bid Quantity', 'Bid Price', 'Ask Price', 'Ask Quantity', 'Net Change',
                'Last Traded Price', 'Implied Volatility', 'Traded Volume', 'Change in Open Interest', 'Open Interest']

            timestamp = json_data.get('timestamp', records.get('timestamp', ''))
            current_time = timestamp

            hist = self.fetch_intraday_1min(symbol, mode) if symbol else None
            candles = None
            if hist is not None and not hist.empty:
                candles = []
                for i in range(0, len(hist), 3):
                    g = hist.iloc[i:i+3]
                    if len(g) < 2:
                        break
                    candles.append({
                        'high': g['High'].max(),
                        'low': g['Low'].min(),
                        'close': g['Close'].iloc[-1],
                        'volume': int(g['Volume'].sum()),
                    })
                if len(candles) < 20:
                    candles = None
            vwap = self._calc_vwap(candles) if candles else None
            adx_data = self._calculate_adx(candles) if candles else None
            ema_data = self._calc_ema_cross(candles) if candles else None
            st_data = self._calc_supertrend(candles) if candles else None
            crossover_15m = self._calc_crossover_15m(hist) if hist is not None else None
            return self._compute_results(merged, current_time, points, lot_size, mode,
                                         vwap, adx_data, ema_data, st_data,
                                         crossover_15m)
        except Exception as e:
            print(f"process_oc_data error: {e}", file=sys.stderr)
            return None

    def _compute_results(self, oc: pd.DataFrame, current_time: str, points: float, lot_size: int,
                         mode: str, vwap: Optional[float] = None,
                         adx_data: Optional[Dict] = None,
                         ema_data: Optional[Dict] = None,
                         st_data: Optional[Dict] = None,
                         crossover_15m: Optional[Dict] = None) -> dict:
        round_factor = 1000 if mode == 'Index' else 10
        units_str = 'in K' if mode == 'Index' else 'in 10s'

        call_oi_list = oc.iloc[:, 0].tolist()
        put_oi_list = oc.iloc[:, 20].tolist()

        max_call_oi = round(max(call_oi_list) / round_factor, 1) if call_oi_list else 0
        max_call_oi_sp = float(oc.iloc[call_oi_list.index(max(call_oi_list))]['Strike Price']) if call_oi_list else 0
        max_put_oi = round(max(put_oi_list) / round_factor, 1) if put_oi_list else 0
        max_put_oi_sp = float(oc.iloc[put_oi_list.index(max(put_oi_list))]['Strike Price']) if put_oi_list else 0

        total_call_oi = sum(call_oi_list)
        total_put_oi = sum(put_oi_list)
        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0

        max_pain = self._calc_max_pain(oc)

        atm_idx = int((oc['Strike Price'] - points).abs().idxmin())
        start = max(0, atm_idx - 15)
        end = min(len(oc), atm_idx + 16)

        kept = [0, 1, 2, 3, 4, 10, 16, 17, 18, 19, 20]
        sliced = oc.iloc[start:end, kept]

        ce_net_chg = oc.iloc[start:end, 5]
        pe_net_chg = oc.iloc[start:end, 15]

        ce_mf = (sliced.iloc[:, 1] * 25 /
                 sliced.iloc[:, 2].replace(0, float('nan'))).fillna(0).astype(int)
        pe_mf = (sliced.iloc[:, 9] * 25 /
                 sliced.iloc[:, 8].replace(0, float('nan'))).fillna(0).astype(int)

        ce_signal = pd.Series('', index=ce_net_chg.index)
        pe_signal = pd.Series('', index=pe_net_chg.index)
        ce_signal[(sliced.iloc[:, 1] > 0) & (ce_net_chg > 0)] = 'CALL BUY'
        ce_signal[(sliced.iloc[:, 1] > 0) & (ce_net_chg < 0)] = 'CALL SELL'
        ce_signal[(sliced.iloc[:, 1] < 0) & (ce_net_chg > 0)] = 'COVER'
        ce_signal[(sliced.iloc[:, 1] < 0) & (ce_net_chg < 0)] = 'UNWIND'
        pe_signal[(sliced.iloc[:, 9] > 0) & (pe_net_chg > 0)] = 'PUT BUY'
        pe_signal[(sliced.iloc[:, 9] > 0) & (pe_net_chg < 0)] = 'PUT SELL'
        pe_signal[(sliced.iloc[:, 9] < 0) & (pe_net_chg > 0)] = 'COVER'
        pe_signal[(sliced.iloc[:, 9] < 0) & (pe_net_chg < 0)] = 'UNWIND'

        # Detect unusual activity: only show signals where OI change or volume is extreme
        def _unusual_mask(series):
            vals = pd.to_numeric(series, errors='coerce').values
            m, s = pd.Series(vals).mean(), pd.Series(vals).std()
            if pd.isna(m) or s == 0:
                return pd.Series([False] * len(series), index=series.index)
            return pd.Series([abs(v - m) / s > 1.5 for v in vals], index=series.index)
        ce_unusual = _unusual_mask(sliced.iloc[:, 1]) | _unusual_mask(sliced.iloc[:, 2])
        pe_unusual = _unusual_mask(sliced.iloc[:, 9]) | _unusual_mask(sliced.iloc[:, 8])
        ce_signal[~ce_unusual] = ''
        pe_signal[~pe_unusual] = ''

        table_rows = []
        for i in range(len(sliced)):
            row = sliced.values.tolist()[i]
            table_rows.append({
                'ce_mf': int(ce_mf.iloc[i]) if pd.notna(ce_mf.iloc[i]) else '',
                'ce_sig': ce_signal.iloc[i],
                'ce_oi': self._int_val(row[0]),
                'ce_chg_oi': self._int_val(row[1]),
                'ce_vol': self._int_val(row[2]),
                'ce_iv': round(row[3], 2) if isinstance(row[3], (int, float)) else row[3],
                'ce_ltp': round(row[4], 2) if isinstance(row[4], (int, float)) else row[4],
                'strike': self._int_val(row[5]),
                'pe_ltp': round(row[6], 2) if isinstance(row[6], (int, float)) else row[6],
                'pe_iv': round(row[7], 2) if isinstance(row[7], (int, float)) else row[7],
                'pe_vol': self._int_val(row[8]),
                'pe_chg_oi': self._int_val(row[9]),
                'pe_oi': self._int_val(row[10]),
                'pe_sig': pe_signal.iloc[i],
                'pe_mf': int(pe_mf.iloc[i]) if pd.notna(pe_mf.iloc[i]) else '',
                '_atm': (i == atm_idx - start),
                '_near_atm': (1 <= abs(i - (atm_idx - start)) <= 5),
                '_unusual_ce': bool(ce_unusual.iloc[i]),
                '_unusual_pe': bool(pe_unusual.iloc[i]),
            })

        oi_buildup = []
        for i in range(atm_idx - 5, atm_idx + 5):
            if 0 <= i < len(oc):
                ce_chg = int(oc.iloc[i, 1])
                pe_chg = int(oc.iloc[i, 19])
                oi_buildup.append({
                    'strike': int(oc.iloc[i]['Strike Price']),
                    'ce_chg': ce_chg,
                    'pe_chg': pe_chg,
                    'diff': ce_chg - pe_chg,
                })

        strikes_arr = oc['Strike Price'].values.astype(float)
        indices_10 = [i for i in range(atm_idx - 5, atm_idx + 6) if i != atm_idx and 0 <= i < len(strikes_arr)]
        ce_top_sr = sorted(indices_10, key=lambda i: oc.iloc[i, 2], reverse=True)[:3]
        pe_top_sr = sorted(indices_10, key=lambda i: oc.iloc[i, 18], reverse=True)[:3]
        ce_avg_sr = int(sum(strikes_arr[i] for i in ce_top_sr) / 3) if ce_top_sr else 0
        pe_avg_sr = int(sum(strikes_arr[i] for i in pe_top_sr) / 3) if pe_top_sr else 0
        sr_range = abs(ce_avg_sr - pe_avg_sr)

        vix_data = self.get_vix() or {}
        vix = vix_data.get('last', 0.0) or 0.0

        call_sum = oc.iloc[atm_idx, 1] + oc.iloc[atm_idx + 1, 1] + oc.iloc[atm_idx + 2, 1] if atm_idx + 2 < len(oc) else oc.iloc[atm_idx, 1]
        put_sum = oc.iloc[atm_idx, 19] + oc.iloc[atm_idx + 1, 19] + oc.iloc[atm_idx + 2, 19] if atm_idx + 2 < len(oc) else oc.iloc[atm_idx, 19]
        oi_label = "Bearish" if call_sum >= put_sum else "Bullish"

        iv_indices = [i for i in range(atm_idx - 5, atm_idx + 6) if i != atm_idx and 0 <= i < len(strikes_arr)]
        iv_ce_vals = [oc.iloc[i, 3] for i in iv_indices]
        iv_pe_vals = [oc.iloc[i, 17] for i in iv_indices]
        avg_ce_iv = round(sum(iv_ce_vals) / len(iv_ce_vals), 2) if iv_ce_vals else 0.0
        avg_pe_iv = round(sum(iv_pe_vals) / len(iv_pe_vals), 2) if iv_pe_vals else 0.0

        fix = load_fix_iv()
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if fix['date'] != today:
            now = datetime.datetime.now()
            market_open = now.replace(hour=9, minute=45, second=0, microsecond=0)
            if now >= market_open:
                save_fix_iv(avg_ce_iv, avg_pe_iv)
                fix = load_fix_iv()

        fix_ce = fix['ce_iv']
        fix_pe = fix['pe_iv']
        mov_ce = round(avg_ce_iv - fix_ce, 2) if fix_ce else 0.0
        mov_pe = round(avg_pe_iv - fix_pe, 2) if fix_pe else 0.0

        vwap_signal = ''
        if vwap and points:
            diff = points - vwap
            if diff > 0:
                vwap_signal = 'ABOVE'
            elif diff < 0:
                vwap_signal = 'BELOW'

        # AI Market Direction Signal
        ai_score = 50
        ai_factors = []
        if vwap_signal == 'ABOVE':
            ai_score += 10; ai_factors.append({'f': 'VWAP', 'i': 'BULLISH', 'w': 10})
        elif vwap_signal == 'BELOW':
            ai_score -= 10; ai_factors.append({'f': 'VWAP', 'i': 'BEARISH', 'w': 10})
        if ema_data:
            if ema_data['signal'] == 'BULLISH':
                ai_score += 10
                if ema_data['cross'] == 'CROSS UP':
                    ai_score += 5; ai_factors.append({'f': 'EMA Cross', 'i': 'BULLISH', 'w': 15})
                else:
                    ai_factors.append({'f': 'EMA', 'i': 'BULLISH', 'w': 10})
            elif ema_data['signal'] == 'BEARISH':
                ai_score -= 10
                if ema_data['cross'] == 'CROSS DN':
                    ai_score -= 5; ai_factors.append({'f': 'EMA Cross', 'i': 'BEARISH', 'w': 15})
                else:
                    ai_factors.append({'f': 'EMA', 'i': 'BEARISH', 'w': 10})
        if st_data:
            if st_data['signal'] == 'BULLISH':
                ai_score += 12; ai_factors.append({'f': 'SuperTrend', 'i': 'BULLISH', 'w': 12})
            elif st_data['signal'] == 'BEARISH':
                ai_score -= 12; ai_factors.append({'f': 'SuperTrend', 'i': 'BEARISH', 'w': 12})
        if adx_data and adx_data.get('adx'):
            a = adx_data['adx']
            if a >= 25:
                if adx_data['signal'] == 'UPTREND':
                    ai_score += 8; ai_factors.append({'f': f'ADX({int(a)})', 'i': 'BULLISH', 'w': 8})
                elif adx_data['signal'] == 'DOWNTREND':
                    ai_score -= 8; ai_factors.append({'f': f'ADX({int(a)})', 'i': 'BEARISH', 'w': 8})
            elif a >= 20:
                if adx_data['signal'] == 'UPTREND':
                    ai_score += 3
                elif adx_data['signal'] == 'DOWNTREND':
                    ai_score -= 3
        if oi_label == 'Bullish':
            ai_score += 8; ai_factors.append({'f': 'OI Shift', 'i': 'BULLISH', 'w': 8})
        elif oi_label == 'Bearish':
            ai_score -= 8; ai_factors.append({'f': 'OI Shift', 'i': 'BEARISH', 'w': 8})
        if pcr >= 1.2:
            ai_score += 6; ai_factors.append({'f': 'PCR', 'i': 'BULLISH', 'w': 6})
        elif pcr <= 0.8:
            ai_score -= 6; ai_factors.append({'f': 'PCR', 'i': 'BEARISH', 'w': 6})
        elif pcr >= 1:
            ai_score += 3
        if points and max_pain:
            mpd = points - max_pain
            if mpd > 0:
                ai_score += 5; ai_factors.append({'f': 'Spot>MaxPain', 'i': 'BULLISH', 'w': 5})
            elif mpd < 0:
                ai_score -= 5; ai_factors.append({'f': 'Spot<MaxPain', 'i': 'BEARISH', 'w': 5})
        if avg_ce_iv and avg_pe_iv:
            sk = avg_pe_iv - avg_ce_iv
            if sk > 5:
                ai_score -= 4; ai_factors.append({'f': 'IV Skew', 'i': 'BEARISH', 'w': 4})
            elif sk < -5:
                ai_score += 4; ai_factors.append({'f': 'IV Skew', 'i': 'BULLISH', 'w': 4})
        if vix:
            if vix > 25:
                ai_score -= 5; ai_factors.append({'f': 'VIX Fear', 'i': 'BEARISH', 'w': 5})
            elif vix < 15:
                ai_score += 3; ai_factors.append({'f': 'VIX Calm', 'i': 'BULLISH', 'w': 3})
        ai_score = max(0, min(100, round(ai_score)))
        ai_direction = 'BULLISH' if ai_score >= 65 else 'NEUTRAL' if ai_score >= 45 else 'BEARISH'

        final_signal = 'HOLD'
        if ai_direction == 'BULLISH' and vwap_signal == 'ABOVE' and pcr >= 1:
            final_signal = 'BUY CALL'
        elif ai_direction == 'BEARISH' and vwap_signal == 'BELOW' and pcr < 1:
            final_signal = 'BUY PUT'

        return {
            'timestamp': current_time,
            'spot': int(round(points)),
            'vwap': round(vwap, 2) if vwap else None,
            'vwap_signal': vwap_signal,
            'adx': adx_data['adx'] if adx_data else None,
            'adx_signal': adx_data['signal'] if adx_data else '',

            'ema_signal': ema_data['signal'] if ema_data else '',
            'ema_cross': ema_data['cross'] if ema_data else '',
            'st_signal': st_data['signal'] if st_data else '',
            'vix': round(vix, 2),
            'max_pain': int(round(max_pain)),
            'pcr': pcr,
            'oi_label': oi_label,
            'oi_color': red if call_sum >= put_sum else green,
            'pcr_color': green if pcr >= 1 else red,
            'max_call_oi': max_call_oi,
            'max_call_oi_sp': max_call_oi_sp,
            'max_put_oi': max_put_oi,
            'max_put_oi_sp': max_put_oi_sp,
            'ce_avg_iv': avg_ce_iv,
            'pe_avg_iv': avg_pe_iv,
            'fix_ce_iv': fix_ce,
            'fix_pe_iv': fix_pe,
            'mov_ce': mov_ce,
            'mov_pe': mov_pe,
            'units': units_str,
            'table': table_rows,
            'oi_buildup': oi_buildup,
            'sr_resistance': ce_avg_sr,
            'sr_support': pe_avg_sr,
            'sr_range': sr_range,
            'ai_score': ai_score,
            'ai_direction': ai_direction,
            'ai_factors': ai_factors,
            'final_signal': final_signal,
            'crossover_15m': crossover_15m,
        }

    def _calc_max_pain(self, oc: pd.DataFrame) -> float:
        try:
            strikes = oc['Strike Price'].values.astype(float)
            ce_oi = oc.iloc[:, 0].values.astype(float)
            pe_oi = oc.iloc[:, 20].values.astype(float)
            min_pain = float('inf')
            max_pain_sp = 0.0
            for i in range(len(strikes)):
                call_pain = sum((strikes[j] - strikes[i]) * ce_oi[j] for j in range(i + 1, len(strikes)))
                put_pain = sum((strikes[i] - strikes[j]) * pe_oi[j] for j in range(i))
                total = call_pain + put_pain
                if total < min_pain:
                    min_pain = total
                    max_pain_sp = strikes[i]
            return max_pain_sp
        except Exception:
            return 0.0

    @staticmethod
    def _calc_rsi(candles: List[Dict], period: int = 14) -> Optional[float]:
        if not candles or len(candles) < period + 1:
            return None
        closes = [c['close'] for c in candles]
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(d if d > 0 else 0.0)
            losses.append(-d if d < 0 else 0.0)
        avg_g = sum(gains[:period]) / period
        avg_l = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            avg_g = (avg_g * (period - 1) + gains[i]) / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            return 100.0 if avg_g > 0 else 50.0
        rs = avg_g / avg_l
        return round(100.0 - (100.0 / (1.0 + rs)), 1)

    def tech_scan_stock(self, symbol: str) -> Optional[Dict]:
        try:
            yf_sym = f'{symbol}.NS'
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period='3d', interval='5m')
            if hist.empty or len(hist) < 30:
                hist2 = ticker.history(period='5d', interval='15m')
                if hist2.empty or len(hist2) < 20:
                    return None
                hist = hist2
            candles = []
            step = max(1, len(hist) // 30)
            for i in range(0, len(hist), step):
                g = hist.iloc[i:i + step]
                if len(g) < 2:
                    break
                candles.append({
                    'high': g['High'].max(), 'low': g['Low'].min(),
                    'close': g['Close'].iloc[-1], 'volume': int(g['Volume'].sum()),
                })
            if len(candles) < 15:
                return None
            lc = candles[-1]['close']
            vwap = self._calc_vwap(candles)
            vs = ''
            if vwap and lc:
                vs = 'ABOVE' if lc > vwap else 'BELOW' if lc < vwap else ''
            ema = self._calc_ema_cross(candles)
            st = self._calc_supertrend(candles)
            adx = self._calculate_adx(candles)
            rsi = self._calc_rsi(candles)
            vols = [c['volume'] for c in candles]
            avg_v = sum(vols) / len(vols) if vols else 1
            latest_vol = vols[-1] if vols else 0
            vol_ratio = round(latest_vol / avg_v, 2) if avg_v > 0 else 1.0
            return {
                'vwap': round(vwap, 2) if vwap else None,
                'vwapSignal': vs,
                'adx': round(adx['adx'], 1) if adx and adx.get('adx') else None,
                'adxSignal': adx['signal'] if adx else '',
                'emaSignal': ema['signal'] if ema else '',
                'emaCross': ema['cross'] if ema else '',
                'stSignal': st['signal'] if st else '',
                'rsi': rsi,
                'volume': int(sum(vols)),
                'avgVolume': int(avg_v),
                'volRatio': vol_ratio,
            }
        except Exception:
            return None

    def get_delivery_data(self, symbol: str) -> Optional[Dict]:
        try:
            today = datetime.datetime.now()
            from_d = (today - datetime.timedelta(days=5)).strftime('%d-%m-%Y')
            to_d = today.strftime('%d-%m-%Y')
            url = f'https://www.nseindia.com/api/corporates/securityArchives?from={from_d}&to={to_d}&symbol={symbol}'
            resp = self._fetch(url)
            if resp and resp.status_code == 200:
                rows = resp.json()
                if rows and len(rows) > 0:
                    latest = rows[-1]
                    delivered_qty = float(latest.get('deliveredQuantity', 0) or 0)
                    total_qty = float(latest.get('totalTradedQuantity', 0) or 0)
                    delivery_pct = round(delivered_qty / total_qty * 100, 1) if total_qty > 0 else 0
                    prev = rows[-2] if len(rows) > 1 else None
                    prev_pct = 0
                    if prev:
                        pq = float(prev.get('deliveredQuantity', 0) or 0)
                        tq = float(prev.get('totalTradedQuantity', 0) or 0)
                        prev_pct = round(pq / tq * 100, 1) if tq > 0 else 0
                    return {
                        'deliveryQty': int(delivered_qty),
                        'totalQty': int(total_qty),
                        'deliveryPct': delivery_pct,
                        'prevDeliveryPct': prev_pct,
                    }
            return None
        except Exception:
            return None

    @staticmethod
    def _int_val(v) -> int:
        if isinstance(v, (int, float)) and v == v:
            return int(v)
        return 0


def calc_ai_score(data: Dict) -> Tuple[int, str, List[str]]:
    s = 50
    reasons = []
    tr = data.get('vwapSignal', '')
    if tr == 'ABOVE':
        s += 8; reasons.append('VWAP_ABV')
    elif tr == 'BELOW':
        s -= 8; reasons.append('VWAP_BEL')
    em = data.get('emaSignal', '')
    ec = data.get('emaCross', '')
    if em == 'BULLISH':
        s += 7
        if ec == 'CROSS UP':
            s += 5; reasons.append('EMA_BL_CR')
        else:
            reasons.append('EMA_BL')
    elif em == 'BEARISH':
        s -= 7
        if ec == 'CROSS DN':
            s -= 5; reasons.append('EMA_BR_CR')
        else:
            reasons.append('EMA_BR')
    st = data.get('stSignal', '')
    if st == 'BULLISH':
        s += 10; reasons.append('ST_BL')
    elif st == 'BEARISH':
        s -= 10; reasons.append('ST_BR')
    adx = data.get('adx', 0) or 0
    ads = data.get('adxSignal', '')
    if adx >= 25:
        if ads == 'UPTREND':
            s += 8; reasons.append(f'ADX{int(adx)}_UP')
        elif ads == 'DOWNTREND':
            s -= 8; reasons.append(f'ADX{int(adx)}_DN')
    elif adx >= 20:
        if ads == 'UPTREND':
            s += 3
        elif ads == 'DOWNTREND':
            s -= 3
    rsi = data.get('rsi', 50) or 50
    if rsi > 70:
        s -= 5; reasons.append('RSI_OB')
    elif rsi > 60:
        s += 3; reasons.append('RSI_MOM')
    elif rsi > 50:
        s += 1
    elif rsi > 40:
        s -= 1
    elif rsi > 30:
        s -= 3; reasons.append('RSI_WK')
    else:
        s -= 5; reasons.append('RSI_OS')
    vr = data.get('volRatio', 1.0) or 1.0
    if vr > 2.0:
        s += 5; reasons.append('VOL_SURGE')
    elif vr > 1.5:
        s += 3; reasons.append('VOL_HIGH')
    elif vr < 0.5:
        s -= 3; reasons.append('VOL_LOW')
    chg = data.get('pChange', 0) or 0
    if chg > 2:
        s += 3; reasons.append('GAIN{:.0f}'.format(chg))
    elif chg < -2:
        s -= 3; reasons.append('LOSS{:.0f}'.format(abs(chg)))
    elif chg > 1:
        s += 1
    elif chg < -1:
        s -= 1
    s = max(0, min(100, round(s)))
    if s >= 80:
        sig = 'STRONG BUY'
    elif s >= 65:
        sig = 'BUY'
    elif s >= 45:
        sig = 'NEUTRAL'
    elif s >= 30:
        sig = 'SELL'
    else:
        sig = 'STRONG SELL'
    return s, sig, reasons


def update_forward_bias(symbol: str, d: Dict) -> Dict:
    global _ai_history
    now = time.time()
    if symbol not in _ai_history:
        _ai_history[symbol] = {'data': [], 'last_gift': None, 'last_gift_ts': 0}

    hist = _ai_history[symbol]
    snap = {
        'ts': now,
        'spot': d.get('spot'),
        'pcr': d.get('pcr'),
        'max_pain': d.get('max_pain'),
        'vwap': d.get('vwap'),
        'vwap_signal': d.get('vwap_signal'),
        'ce_oi_sp': d.get('max_call_oi_sp'),
        'pe_oi_sp': d.get('max_put_oi_sp'),
        'oi_label': d.get('oi_label'),
        'st_signal': d.get('st_signal'),
        'ema_signal': d.get('ema_signal'),
        'ema_cross': d.get('ema_cross'),
        'adx': d.get('adx'),
        'adx_signal': d.get('adx_signal'),
    }
    hist['data'].append(snap)
    if len(hist['data']) > _AI_HISTORY_MAX:
        hist['data'] = hist['data'][-_AI_HISTORY_MAX:]

    fb_score = 50
    fb_factors = []

    # 1. PCR Momentum (weight 25)
    pcr_vals = [p['pcr'] for p in hist['data'] if p.get('pcr') is not None]
    if len(pcr_vals) >= 3:
        recent = pcr_vals[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            fb_score += 20; fb_factors.append({'f': 'PCR Rising', 'i': 'BULLISH', 'w': 20, 'd': f'{recent[-3]:.2f}→{recent[-1]:.2f}'})
        elif recent[-1] < recent[-2] < recent[-3]:
            fb_score -= 20; fb_factors.append({'f': 'PCR Falling', 'i': 'BEARISH', 'w': 20, 'd': f'{recent[-3]:.2f}→{recent[-1]:.2f}'})
        elif recent[-1] > recent[-2]:
            fb_score += 8
        elif recent[-1] < recent[-2]:
            fb_score -= 8

    # 2. Max Pain Drift (weight 20)
    mp_diffs = []
    for p in hist['data']:
        if p.get('spot') and p.get('max_pain'):
            mp_diffs.append(p['spot'] - p['max_pain'])
    if len(mp_diffs) >= 3:
        if abs(mp_diffs[-1]) > abs(mp_diffs[-3]) and mp_diffs[-1] > 0:
            fb_score += 15; fb_factors.append({'f': 'Spot↑ away MP', 'i': 'BULLISH', 'w': 15, 'd': f'{int(mp_diffs[-3])}→{int(mp_diffs[-1])}'})
        elif abs(mp_diffs[-1]) > abs(mp_diffs[-3]) and mp_diffs[-1] < 0:
            fb_score -= 15; fb_factors.append({'f': 'Spot↓ away MP', 'i': 'BEARISH', 'w': 15, 'd': f'{int(mp_diffs[-3])}→{int(mp_diffs[-1])}'})
        elif abs(mp_diffs[-1]) < abs(mp_diffs[-3]):
            fb_score += 10 if mp_diffs[-1] > 0 else -10
            fb_factors.append({'f': 'Spot→MP revert', 'i': 'NEUTRAL', 'w': 10, 'd': f'{int(mp_diffs[-3])}→{int(mp_diffs[-1])}'})

    # 3. OI Concentration Shift (weight 20)
    call_sp = [p['ce_oi_sp'] for p in hist['data'] if p.get('ce_oi_sp')]
    put_sp = [p['pe_oi_sp'] for p in hist['data'] if p.get('pe_oi_sp')]
    if len(call_sp) >= 2:
        if call_sp[-1] > call_sp[-2]:
            fb_score += 10; fb_factors.append({'f': 'Call OI↑ strike', 'i': 'BULLISH', 'w': 10, 'd': f'{int(call_sp[-2])}→{int(call_sp[-1])}'})
        elif call_sp[-1] < call_sp[-2]:
            fb_score -= 6
    if len(put_sp) >= 2:
        if put_sp[-1] > put_sp[-2]:
            fb_score += 6
        elif put_sp[-1] < put_sp[-2]:
            fb_score -= 10; fb_factors.append({'f': 'Put OI↓ strike', 'i': 'BEARISH', 'w': 10, 'd': f'{int(put_sp[-2])}→{int(put_sp[-1])}'})

    # 4. VWAP Momentum (weight 15)
    vwap_vals = [p['vwap'] for p in hist['data'] if p.get('vwap')]
    if len(vwap_vals) >= 3 and d.get('vwap'):
        if vwap_vals[-1] > vwap_vals[-3] and d.get('vwap_signal') == 'ABOVE':
            fb_score += 12; fb_factors.append({'f': 'VWAP rising↑', 'i': 'BULLISH', 'w': 12})
        elif vwap_vals[-1] < vwap_vals[-3] and d.get('vwap_signal') == 'BELOW':
            fb_score -= 12; fb_factors.append({'f': 'VWAP falling↓', 'i': 'BEARISH', 'w': 12})

    # 5. Gift Nifty Divergence (weight 20)
    try:
        gs = requests.Session()
        gs.headers.update({'user-agent': 'Mozilla/5.0'})
        gr = gs.get('https://giftcitynifty.com', timeout=5)
        if gr.status_code == 200:
            import re
            txt = gr.text
            gm = re.search(r'Gift Nifty Live price is ([\d,]+\.?\d*)', txt)
            gm2 = re.search(r'up by ([\d,]+\.?\d*)', txt)
            gm3 = re.search(r'down by ([\d,]+\.?\d*)', txt)
            if gm:
                gift_chg = float(gm2.group(1).replace(',', '')) if gm2 else (-float(gm3.group(1).replace(',', '')) if gm3 else 0)
                spot_chg = d.get('spot_chg', 0) or 0
                if gift_chg > 30 and spot_chg < 10:
                    fb_score += 16; fb_factors.append({'f': 'Gift↑ Spot flat', 'i': 'BULLISH', 'w': 16, 'd': f'G+{gift_chg}'})
                elif gift_chg < -30 and spot_chg > -10:
                    fb_score -= 16; fb_factors.append({'f': 'Gift↓ Spot flat', 'i': 'BEARISH', 'w': 16, 'd': f'G{gift_chg}'})
    except Exception:
        pass

    fb_score = max(0, min(100, round(fb_score)))
    if fb_score >= 65:
        fb_dir = 'BULLISH'
    elif fb_score >= 45:
        fb_dir = 'NEUTRAL'
    else:
        fb_dir = 'BEARISH'
    return {'fb_score': fb_score, 'fb_direction': fb_dir, 'fb_factors': fb_factors}


nse = NseWeb()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prochain')
def prochain():
    return render_template('prochain.html')

@app.route('/indices')
def indices():
    return render_template('indices.html')

@app.route('/stocks')
def stocks():
    return render_template('stocks.html')


@app.route('/api/iv-fix', methods=['GET', 'POST'])
def api_iv_fix():
    if request.method == 'POST':
        data = request.get_json()
        ce_iv = float(data.get('ce_iv', 0))
        pe_iv = float(data.get('pe_iv', 0))
        save_fix_iv(ce_iv, pe_iv)
        return jsonify({'status': 'ok'})
    fix = load_fix_iv()
    return jsonify(fix)


@app.route('/api/init')
def api_init():
    if not nse.indices or not nse.stocks:
        nse.get_symbols()
    return jsonify({
        'indices': nse.indices,
        'stocks': nse.stocks,
    })


@app.route('/api/dates')
def api_dates():
    symbol = request.args.get('symbol', '')
    mode = request.args.get('mode', 'Index')
    dates = nse.get_expiry_dates(symbol, mode)
    return jsonify({'dates': dates})


@app.route('/api/vix')
def api_vix():
    vix = nse.get_vix()
    if vix:
        return jsonify({'vix': vix.get('last'), 'change': vix.get('change'), 'percentChange': vix.get('percentChange')})
    return jsonify({'vix': None, 'change': None, 'percentChange': None})

_api_data_cache: Dict[str, Tuple[float, dict]] = {}
API_CACHE_TTL = 4

@app.route('/api/data')
def api_data():
    symbol = request.args.get('symbol', '')
    expiry = request.args.get('expiry', '')
    mode = request.args.get('mode', 'Index')
    if not symbol or not expiry:
        return jsonify({'error': 'symbol and expiry required'}), 400
    cache_key = f'{symbol}|{expiry}|{mode}'
    now = time.time()
    cached = _api_data_cache.get(cache_key)
    if cached and now - cached[0] < API_CACHE_TTL:
        return jsonify(cached[1])
    raw = nse.get_oc_raw(symbol, expiry, mode)
    if raw is None:
        return jsonify({'error': 'Failed to fetch data'}), 500
    result = nse.process_oc_data(raw, mode, symbol)
    if result is None:
        return jsonify({'error': 'Failed to process data'}), 500
    expiries = raw.get('records', {}).get('expiryDates') or raw.get('expiryDates', [])
    result['expiries'] = expiries
    fb = update_forward_bias(symbol, result)
    result['forward_bias'] = fb
    _api_data_cache[cache_key] = (now, result)
    return jsonify(result)


@app.route('/api/all-indices')
def api_all_indices():
    resp = nse._fetch(nse.url_vix)
    if resp and resp.status_code == 200:
        return jsonify(resp.json().get('data', []))
    return jsonify({'error': 'Failed to fetch indices'}), 500


_idx_const_cache: Dict[str, Tuple[float, set]] = {}
_INDEX_TAGS = [
    'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200',
    'NIFTY MIDCAP 100', 'NIFTY MIDCAP 150',
    'NIFTY SMALLCAP 250', 'NIFTY SMALLCAP 50',
]


def _get_idx_constituents(index: str) -> set:
    now = time.time()
    cached = _idx_const_cache.get(index)
    if cached and now - cached[0] < 300:
        return cached[1]
    resp = nse._fetch(f'https://www.nseindia.com/api/equity-stockIndices?index={index}')
    if resp and resp.status_code == 200:
        items = resp.json().get('data', [])
        syms = {s['symbol'] for s in items[1:] if 'symbol' in s}
        _idx_const_cache[index] = (now, syms)
        return syms
    return set()


@app.route('/api/stocks')
def api_stocks():
    resp = nse._fetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500')
    if resp and resp.status_code == 200:
        data = resp.json()
        all_items = data.get('data', [])
        stocks = all_items[1:]

        index_map: Dict[str, set] = {}
        for tag in _INDEX_TAGS:
            index_map[tag] = _get_idx_constituents(tag)

        for s in stocks:
            sym = s.get('symbol', '')
            s['indexTags'] = [t for t, syms in index_map.items() if sym in syms]

        sorted_by_pct = sorted(stocks, key=lambda x: float(x.get('pChange', 0) or 0), reverse=True)
        return jsonify({
            'timestamp': data.get('timestamp', ''),
            'total': len(stocks),
            'stocks': stocks,
            'gainers': sorted_by_pct[:10],
            'losers': sorted_by_pct[-10:][::-1],
        })
    return jsonify({'error': 'Failed to fetch'}), 502


@app.route('/api/index-tags')
def api_index_tags():
    return jsonify({'indices': _INDEX_TAGS})


@app.route('/api/index-constituents')
def api_index_constituents():
    index = request.args.get('index', '')
    if not index:
        return jsonify({'error': 'index parameter required'}), 400
    resp = nse._fetch(f'https://www.nseindia.com/api/equity-stockIndices?index={index}')
    if resp and resp.status_code == 200:
        data = resp.json()
        all_items = data.get('data', [])
        stocks = all_items[1:]
        return jsonify({
            'name': data.get('name', index),
            'advance': data.get('advance', {}),
            'timestamp': data.get('timestamp', ''),
            'stocks': stocks
        })
    return jsonify({'error': 'Failed to fetch'}), (resp.status_code if resp else 502)


@app.route('/api/gift-nifty')
def api_gift_nifty():
    global _gift_nifty_cache
    now = time.time()
    if now - _gift_nifty_cache['timestamp'] < GIFT_NIFTY_CACHE_TTL and _gift_nifty_cache['data']:
        return jsonify(_gift_nifty_cache['data'])
    try:
        s = requests.Session()
        s.headers.update({'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        r = s.get('https://giftcitynifty.com', timeout=10)
        if r.status_code == 200:
            import re
            text = r.text
            m = re.search(r'Gift Nifty Live price is ([\d,]+\.?\d*)', text)
            if m:
                last = float(m.group(1).replace(',', ''))
                m2 = re.search(r'up by ([\d,]+\.?\d*)', text)
                m3 = re.search(r'down by ([\d,]+\.?\d*)', text)
                if m2:
                    change = float(m2.group(1).replace(',', ''))
                elif m3:
                    change = -float(m3.group(1).replace(',', ''))
                else:
                    change = 0
                m4 = re.search(r'Previous Close</td>\s*<td>([\d,]+\.?\d*)</td>', r.text)
                prev_close = float(m4.group(1).replace(',', '')) if m4 else last - change
                pct = round(change / prev_close * 100, 2) if prev_close else 0
                data = {'last': last, 'change': round(change, 2), 'percentChange': pct}
                _gift_nifty_cache = {'data': data, 'timestamp': now}
                return jsonify(data)
        return jsonify({'error': 'Could not fetch'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 502


@app.route('/api/fiidii')
def api_fiidii():
    try:
        resp = nse._fetch('https://www.nseindia.com/api/fiidii')
        if resp and resp.status_code == 200:
            return jsonify(resp.json())
        return jsonify({'error': 'FII data unavailable'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 502


@app.route('/api/adv-scan', methods=['POST'])
def api_adv_scan():
    req = request.get_json() or {}
    filters = req.get('filters', {})
    sel_idx = req.get('indices', []) or []
    min_p = float(req.get('minPrice', 0) or 0)
    max_p = float(req.get('maxPrice', 999999) or 999999)
    max_s = int(req.get('maxStocks', 500) or 500)

    vf = filters.get('vwap', '')
    ax = float(filters.get('adxMin', 0) or 0)
    ef = filters.get('emaSignal', '')
    sf = filters.get('superTrend', '')
    rmin = float(filters.get('rsiMin', 0) or 0)
    rmax = float(filters.get('rsiMax', 100) or 100)
    sb = filters.get('sortBy', 'aiScore')
    so = filters.get('sortOrder', 'desc')
    mas = float(filters.get('minAiScore', 0) or 0)
    mdel = float(filters.get('minDeliveryPct', 0) or 0)

    resp = nse._fetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500')
    if not resp or resp.status_code != 200:
        return jsonify({'error': 'NIFTY 500 data failed'}), 502

    nd = resp.json()
    all_s = nd.get('data', [])[1:]
    im: Dict[str, set] = {}
    for t in _INDEX_TAGS:
        im[t] = _get_idx_constituents(t)

    cands = []
    for s in all_s:
        sym = s.get('symbol', '')
        tags = [t for t, syms in im.items() if sym in syms]
        if sel_idx and not any(t in sel_idx for t in tags):
            continue
        ltp = float(s.get('lastPrice', 0) or 0)
        if ltp < min_p or ltp > max_p:
            continue
        s['_ltp'] = ltp
        s['indexTags'] = tags
        cands.append(s)

    cands = cands[:max_s]
    start_t = time.time()
    results = []
    scanned = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        fm = {ex.submit(nse.tech_scan_stock, s['symbol']): s for s in cands}
        for f in concurrent.futures.as_completed(fm):
            s = fm[f]; scanned += 1
            try:
                tc = f.result()
                if tc is None:
                    continue
                ltp = float(s.get('_ltp', 0) or 0)
                pch = float(s.get('pChange', 0) or 0)
                rec = {
                    'symbol': s['symbol'], 'ltp': ltp,
                    'change': float(s.get('change', 0) or 0),
                    'pChange': round(pch, 2),
                    'vwap': tc.get('vwap'), 'vwapSignal': tc.get('vwapSignal', ''),
                    'adx': tc.get('adx'), 'adxSignal': tc.get('adxSignal', ''),
                    'emaSignal': tc.get('emaSignal', ''), 'emaCross': tc.get('emaCross', ''),
                    'stSignal': tc.get('stSignal', ''),
                    'rsi': tc.get('rsi'), 'volume': tc.get('volume', 0),
                    'avgVolume': tc.get('avgVolume', 0), 'volRatio': tc.get('volRatio', 1.0),
                    'indexTags': s.get('indexTags', []),
                }
                ai_s, ai_sig, ai_r = calc_ai_score(rec)
                rec['aiScore'] = ai_s
                rec['aiSignal'] = ai_sig
                rec['aiReasons'] = ai_r
                if vf and rec['vwapSignal'] != vf:
                    continue
                if ef and rec['emaSignal'] != ef:
                    continue
                if sf and rec['stSignal'] != sf:
                    continue
                if ax > 0 and (rec['adx'] is None or rec['adx'] < ax):
                    continue
                if rmin > 0 and (rec['rsi'] is None or rec['rsi'] < rmin):
                    continue
                if rmax < 100 and (rec['rsi'] is None or rec['rsi'] > rmax):
                    continue
                if mas > 0 and rec['aiScore'] < mas:
                    continue
                if mdel > 0 and (rec.get('deliveryPct') is None or rec.get('deliveryPct', 0) < mdel):
                    continue
                results.append(rec)
            except Exception:
                continue

    # Fetch delivery data for results in parallel
    if results:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as dx:
            dm = {dx.submit(nse.get_delivery_data, r['symbol']): r for r in results}
            for f in concurrent.futures.as_completed(dm):
                r = dm[f]
                try:
                    dd = f.result()
                    if dd:
                        r['deliveryPct'] = dd.get('deliveryPct', 0)
                        r['deliveryQty'] = dd.get('deliveryQty', 0)
                        r['prevDeliveryPct'] = dd.get('prevDeliveryPct', 0)
                        # Boost AI score for high delivery %
                        dp = dd.get('deliveryPct', 0)
                        if dp > 70:
                            r['aiScore'] = min(100, r['aiScore'] + 5)
                            r['aiReasons'].append('DEL{}'.format(int(dp)))
                        elif dp > 55:
                            r['aiScore'] = min(100, r['aiScore'] + 3)
                            r['aiReasons'].append('DEL{}'.format(int(dp)))
                        elif dp < 20:
                            r['aiScore'] = max(0, r['aiScore'] - 3)
                            r['aiReasons'].append('LOWDEL{}'.format(int(dp)))
                        # Recompute signal
                        if r['aiScore'] >= 80:
                            r['aiSignal'] = 'STRONG BUY'
                        elif r['aiScore'] >= 65:
                            r['aiSignal'] = 'BUY'
                        elif r['aiScore'] >= 45:
                            r['aiSignal'] = 'NEUTRAL'
                        elif r['aiScore'] >= 30:
                            r['aiSignal'] = 'SELL'
                        else:
                            r['aiSignal'] = 'STRONG SELL'
                except Exception:
                    pass

    rev = so != 'asc'
    sk = {'pChange': 'pChange', 'volume': 'volume', 'adx': 'adx', 'rsi': 'rsi', 'deliveryPct': 'deliveryPct'}.get(sb, 'aiScore')
    results.sort(key=lambda x: x.get(sk, 0) if isinstance(x.get(sk), (int, float)) else 0, reverse=rev)

    elapsed = round(time.time() - start_t, 1)
    # Add FII aggregate data
    fii_data = None
    try:
        fii_r = nse._fetch('https://www.nseindia.com/api/fiidii')
        if fii_r and fii_r.status_code == 200:
            fii_data = fii_r.json()
    except Exception:
        pass
    return jsonify({
        'scanned': scanned, 'results': len(results), 'time': elapsed,
        'stocks': results, 'fii': fii_data,
    })


@app.route('/adv-scan')
def adv_scan():
    return render_template('advscan.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
