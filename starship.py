"""
Legend V6.1 - Unified Trading Engine + Grid Tuning (FULL – bez placeholdera)

ŠTA SADRŽI:
- 15 strategija (trend / meanrev / momentum / volatility / pattern)
- Režimska analiza + dinamičko ponderisanje + konflikt filter
- ML model (GradientBoostingClassifier) sa per-simbol 70/30 time-series split
- Opcioni GRID TUNING (time-series subsplits per simbol) -> best_gbm_params.json
- Metrike: Accuracy, AUC, Confusion Matrix, Precision, Recall, profit_proxy (tuning)
- SCALP MODE (M1): fiksni mali target USD, time-stop, early abort, no-progress->BE
- SWING MODE (M5): ATR SL/TP, partial TP, breakeven, trailing (sa anti-spam pomakom)
- Trailing anti-spam: minimalni pomak ≥ ATR * TRAIL_MIN_MOVE_MULT_ATR + cooldown
- Auto disable (strategije & simboli) – Wilson, expectancy, dnevni limit
- Meta-label skeleton (kandidati i naknadno etiketiranje)
- Persistencija strategy_stats / symbol_stats (pickle)
- CSV logging: signals, trades, meta label candidates/final
- Auto retrain (interval + min gap)
- Fix “if not rates” bug
- Grid tuning rezultati u grid_tuning_results.csv
- TUNE_BEFORE_TRAIN kontrola (prvo tuning pa finalni fit)
- FORCE_ONLY_TUNING opcija (izađe nakon tuninga)
- FAST_MODEL parametri kao fallback
- Weekend crypto-only mode
- SCALP tag u komentarima pozicija (filter menadžment)

PREPORUKA:
1. Prvi tuning: radni dan (više simbola) -> ALWAYS_TRAIN_ON_START=True, TUNE_BEFORE_TRAIN=True, GRID_PARAM_MODE="small".
2. Nakon kreiranja best_gbm_params.json, postavi ALWAYS_TRAIN_ON_START=False radi brzog restarta.
3. Vikendom tuning radi samo na crypto – manje kvalitetno.

POKRETANJE:
    python src/eurusd_multi_strategy_simple.py

NAPOMENA: Ako želiš brži test – smanji TRAIN_DAYS_FX na 120 ili 90.

"""

import os
import time
import csv
import math
import json
import logging
import pickle
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier

import requests
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

import talib
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================
# =========================================================
#  FX – TIGHT & SMART (po strategiji skripte)
# =========================================================

# CSV logging files
SIGNAL_LOG_CSV = "signal_log.csv"
TRADES_LOG_FILE = "trades_log.csv"
META_LABEL_CANDIDATES_FILE = "meta_candidates.csv"
META_LABEL_FINAL_FILE = "meta_final.csv"

EURUSD = {
    "symbol": "EURUSD",
    "sl_pips": 5,              # ← 6→5 (najmanji spread = najmanji SL)
    "tp_pips": 10,             # ← 12→10 (1:2 RR dovoljan za visoku preciznost)
    "trail_pips": 2,           # ← 3→2 (raniji trailing, ali bez šume)
    "trail_step": 0.4,         # ← 0.5→0.4 (finiji korak)
    "breakeven_pips": 1.5,     # ← 2→1.5 (brži BE, ali ne odmah)
    "lot_size": 0.10,
    "max_positions": 2,
    "trade_hours": (1, 22),    # ← izbegni rollover 0-1h
    "trade_days": [0, 1, 2, 3, 4]
}

GBPUSD = {
    "symbol": "GBPUSD",
    "sl_pips": 6,              # ← ostaje (veća volatilnost)
    "tp_pips": 14,             # ← 15→14 (malo manje, češći proboj)
    "trail_pips": 3,
    "trail_step": 0.5,
    "breakeven_pips": 2,
    "lot_size": 0.10,
    "max_positions": 2,
    "trade_hours": (1, 22),
    "trade_days": [0, 1, 2, 3, 4]
}

USDJPY = {
    "symbol": "USDJPY",
    "sl_pips": 5,              # ← 6→5 (manji spread, mirniji)
    "tp_pips": 12,             # ← 15→12 (čišći proboj na JPY)
    "trail_pips": 2,
    "trail_step": 0.4,
    "breakeven_pips": 1.5,
    "lot_size": 0.10,
    "max_positions": 2,
    "trade_hours": (1, 22),
    "trade_days": [0, 1, 2, 3, 4]
}

AUDUSD = {
    "symbol": "AUDUSD",
    "sl_pips": 7,              # ← ostaje (volatilniji od EUR)
    "tp_pips": 13,             # ← 14→12 (manje, češći hit)
    "trail_pips": 2,
    "trail_step": 0.4,
    "breakeven_pips": 2,
    "lot_size": 0.10,
    "max_positions": 2,
    "trade_hours": (1, 22),
    "trade_days": [0, 1, 2, 3, 4]
}

NZDUSD = {
    "symbol": "NZDUSD",
    "sl_pips": 8,
    "tp_pips": 15,             # ← 14→11 (čišći RR)
    "trail_pips": 2,
    "trail_step": 0.4,
    "breakeven_pips": 2,
    "lot_size": 0.10,
    "max_positions": 2,
    "trade_hours": (1, 22),
    "trade_days": [0, 1, 2, 3, 4]
}

USDCAD = {
    "symbol": "USDCAD",
    "sl_pips": 6,
    "tp_pips": 13,             # ← 15→13 (manje, češći proboj)
    "trail_pips": 2,
    "trail_step": 0.4,
    "breakeven_pips": 2,
    "lot_size": 0.10,
    "max_positions": 2,
    "trade_hours": (1, 22),
    "trade_days": [0, 1, 2, 3, 4]
}

USDCHF = {
    "symbol": "USDCHF",
    "sl_pips": 5,              # ← 6→5 (mirniji, manji spread)
    "tp_pips": 11,             # ← 14→11
    "trail_pips": 2,
    "trail_step": 0.4,
    "breakeven_pips": 1.5,
    "lot_size": 0.10,
    "max_positions": 2,
    "trade_hours": (1, 22),
    "trade_days": [0, 1, 2, 3, 4]
}

# =========================================================
#  METALI & CRYPTO – ostaju isti (volatilniji)
# =========================================================
XAUUSDs = {
    "symbol": "XAUUSDs",
    "sl_pips": 180,          # 300 → 180 (≈ 0.9 × ATR(14) na M5)
    "tp_pips": 320,          # 500 → 320 (1.78 RR, češći proboj)
    "trail_pips": 35,        # 50 → 35 (ranije čuvanje profita)
    "trail_step": 8,         # 10 → 8 (finiji korak)
    "breakeven_pips": 30,    # 50 → 30 (brži BE, ali ne odmah)
    "lot_size": 0.03,        # 0.05 → 0.03 (manja ekspozicija)
    "max_positions": 1,
    "trade_hours": (9, 22),  # izbegni azijsku dosadu
    "trade_days": [0, 1, 2, 3, 4],
    # scalp filteri (override)
    "scalp_no_progress_pips": 25,
    "scalp_early_abort_pips": -15
}

BTCUSDT = {
    "symbol": "BTCUSDT",
    "sl_pips": 800,          # 1200 → 800 (≈ 1.1 × ATR(14) na M5)
    "tp_pips": 1300,         # 1800 → 1300 (1.6 RR, češći hit)
    "trail_pips": 200,       # 300 → 200 (ranije čuvanje)
    "trail_step": 70,        # 100 → 70 (manji korak)
    "breakeven_pips": 250,   # 400 → 250 (brži BE)
    "lot_size": 0.008,       # 0.01 → 0.008 (manja ekspozicija)
    "max_positions": 1,
    "trade_hours": (0, 24),
    "trade_days": [0, 1, 2, 3, 4, 5, 6],
    # scalp filteri (override)
    "scalp_no_progress_pips": 100,
    "scalp_early_abort_pips": -60
}
# -------------------------------------------------------------------
#  RJEČNIK KOJI POVEZUJE SIMBOL S NJEGOVIM PARAMETRIMA
# -------------------------------------------------------------------
SYMBOL_PARAMS = {
    "EURUSD":   EURUSD,
    "GBPUSD":   GBPUSD,
    "USDJPY":   USDJPY,
    "XAUUSDs":  XAUUSDs,
    "BTCUSDT":  BTCUSDT,
    "AUDUSD":   AUDUSD,
    "NZDUSD":   NZDUSD,
    "USDCAD":   USDCAD,
    "USDCHF":   USDCHF
}

# Tvoja originalna lista (ostavi kako je bilo)
DEFAULT_SYMBOLS = list(SYMBOL_PARAMS.keys())
DEFAULT_SYMBOLS = ["EURUSD","GBPUSD","USDJPY","XAUUSDs","BTCUSDT","AUDUSD","NZDUSD","USDCAD","USDCHF"]
CRYPTO_SET = {"BTCUSD","BTCUSDT"}

SWING_MODE = True
SCALP_MODE = True

ALWAYS_TRAIN_ON_START = False
AUTO_RETRAIN_ENABLED = False
AUTO_RETRAIN_DAYS = 7
AUTO_RETRAIN_MIN_GAP_MIN = 120

FAST_MODEL = True

# GRID TUNING FLAGS
TUNE_BEFORE_TRAIN = False
GRID_PARAM_MODE = "small"         # "small" ili "medium"
FORCE_ONLY_TUNING = False
FORCE_DEFAULT_PARAMS = False

BEST_PARAM_FILE = "best_gbm_params.json"
GRID_RESULTS_FILE = "grid_tuning_results.csv"

LAST_TRAIN_FILE = "last_train_timestamp.txt"
LAST_RETRAIN_ATTEMPT_FILE = "last_retrain_attempt.txt"

TRAIN_DAYS_FX = 90
TRAIN_DAYS_BTC = 180
LABEL_THRESHOLD = 0.0008

ENABLE_SIGNAL_CSV = True
ENABLE_TRADE_CSV = True
ENABLE_META_LABEL_LOG = True

SIGNAL_LOG_FILE = "signals_log.csv"
TRADES_LOG_FILE = "executed_trades.csv"
META_LABEL_CANDIDATES_FILE = "meta_label_candidates.csv"
META_LABEL_FINAL_FILE = "meta_label_labeled.csv"

AUTO_DISABLE_STRATEGIES = False
AUTO_DISABLE_SYMBOLS = False
STRATEGY_DISABLE_MIN_TRADES = 50
STRATEGY_DISABLE_WLB_HITRATE = 0.42
STRATEGY_DISABLE_EXPECTANCY = 0.0
STRATEGY_DISABLE_COOLDOWN_BARS = 200

SYMBOL_DISABLE_ROLLING = 50
SYMBOL_DISABLE_EXPECTANCY = -0.05
SYMBOL_DISABLE_HITRATE = 0.48
SYMBOL_DISABLE_DAILY_R_LIMIT = -3.0
SYMBOL_DISABLE_COOLDOWN_MIN = 240

META_LABEL_LOOKAHEAD_BARS = 12
META_LABEL_MIN_MFE_FACTOR = 0.6
META_LABEL_MAX_MAE_BEFORE_MFE = 0.8

PARTIAL_TP_ENABLED = True
BREAKEVEN_ENABLED = True
TRAILING_ENABLED = True

SCALP_TIMEFRAME = "M1"
SCALP_CONFIRM_TF = "M5"
SCALP_TARGET_USD_RANGE = (3.0, 7.0)
SCALP_TARGET_PIPS_DEFAULT = 20
SCALP_SL_PIPS_DEFAULT = 20
SCALP_MAX_TIME_BARS = 30
SCALP_NO_PROGRESS_PIPS = 9999
SCALP_EARLY_ABORT_PIPS = 9999

MODEL_FILE = "bot_legend_v6.pkl"
SCALER_FILE = "scaler_legend_v6.pkl"
FEATURES_FILE = "features_legend_v6.pkl"
STATS_PERSIST_FILE = "runtime_stats.pkl"

BASE_DROP = ['time','open','high','low','close','tick_volume','spread','real_volume','volume']
MIN_DATA_BARS = 200
PERF_UPDATE_INTERVAL = 120
POLL_HISTORY_MIN = 2

TRAIL_MIN_MOVE_MULT_ATR = 0.30
ITERATION_LOG_INTERVAL = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_legend_v6_grid_full.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

_last_retrain_check = {"time": None}

# =========================================================
# HELPERI
# =========================================================
def load_env_credentials():
    login=os.getenv("MT5_LOGIN")
    password=os.getenv("MT5_PASSWORD")
    server=os.getenv("MT5_SERVER")
    if (not login or not password or not server) and os.path.exists(".env"):
        try:
            with open(".env","r",encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith("#") or "=" not in line: continue
                    k,v=line.split("=",1)
                    k=k.strip(); v=v.strip().strip('"').strip("'")
                    if k=="MT5_LOGIN" and not login: login=v
                    elif k=="MT5_PASSWORD" and not password: password=v
                    elif k=="MT5_SERVER" and not server: server=v
        except:
            pass
    return login,password,server

def mt5_timeframe(tf_str: str):
    if mt5 is None: return None
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1
    }.get(tf_str.upper(), mt5.TIMEFRAME_M5)

def mt5_fetch_bars(symbol: str, tf_str="M5", days=180):
    if mt5 is None:
        return None
    try:
        tf=mt5_timeframe(tf_str)
        end=datetime.now(timezone.utc)
        start=end - timedelta(days=days)
        rates=mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates)==0:
            rates=mt5.copy_rates_from_pos(symbol, tf, 0, 5000)
        if rates is None or len(rates)==0:
            return None
        df=pd.DataFrame(rates)
        df['time']=pd.to_datetime(df['time'], unit='s', utc=True)
        if 'real_volume' in df.columns and 'tick_volume' not in df.columns:
            df.rename(columns={'real_volume':'tick_volume'}, inplace=True)
        return df[['time','open','high','low','close','tick_volume']]
    except Exception as e:
        logging.warning(f"MT5 fetch failed {symbol}: {e}")
        return None

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    try:
        df['rsi6']=talib.RSI(df['close'],6)
        df['rsi14']=talib.RSI(df['close'],14)
        df['rsi24']=talib.RSI(df['close'],24)
        try:
            macd1,signal1,hist1=talib.MACD(df['close'],fastperiod=8,slowperiod=21,signalperiod=5)
            df['macd1']=macd1; df['macd_signal1']=signal1; df['hist1']=hist1
        except:
            df['macd1']=np.nan; df['macd_signal1']=np.nan; df['hist1']=np.nan
        df['atr']=talib.ATR(df['high'], df['low'], df['close'])
        df['atr_ratio']=df['atr']/df['close']
        df['true_range']=talib.TRANGE(df['high'], df['low'], df['close'])
        try:
            bb_u,bb_m,bb_l=talib.BBANDS(df['close'])
            df['bb_upper']=bb_u; df['bb_middle']=bb_m; df['bb_lower']=bb_l
            df['bb_width']=(df['bb_upper']-df['bb_lower'])/df['bb_middle']
            df['bb_percent']=(df['close']-df['bb_lower'])/(df['bb_upper']-df['bb_lower'])
        except:
            for c in ['bb_upper','bb_middle','bb_lower','bb_width','bb_percent']:
                df[c]=np.nan
        for p in [5,8,12,21,26,50,100,200]:
            df[f'ema{p}']=talib.EMA(df['close'],p)
            df[f'sma{p}']=talib.SMA(df['close'],p)
        df['plus_di']=talib.PLUS_DI(df['high'],df['low'],df['close'])
        df['minus_di']=talib.MINUS_DI(df['high'],df['low'],df['close'])
        df['adx']=talib.ADX(df['high'],df['low'],df['close'])
        df['willr']=talib.WILLR(df['high'],df['low'],df['close'])
        df['cci']=talib.CCI(df['high'],df['low'],df['close'])
        df['momentum']=talib.MOM(df['close'])
        df['roc']=talib.ROC(df['close'])
        try:
            st_k,st_d=talib.STOCH(df['high'],df['low'],df['close'])
            df['stoch_k']=st_k; df['stoch_d']=st_d
        except:
            df['stoch_k']=np.nan; df['stoch_d']=np.nan
        try:
            df['stoch_rsi']=talib.STOCHRSI(df['close'])
        except:
            df['stoch_rsi']=np.nan
        df['volume_avg']=df['tick_volume'].rolling(20).mean()
        df['volume_ratio']=df['tick_volume']/df['volume_avg']
        df['volume_trend']=df['tick_volume'].rolling(10).apply(lambda x:1 if x.iloc[-1]>x.mean() else -1)
        df['hl_ratio']=(df['high']-df['close'])/(df['high']-df['low']+1e-5)
        df['oc_ratio']=(df['close']-df['open'])/(df['open']+1e-5)
        df['body_size']=(df['close']-df['open']).abs()/df['open']
        df['upper_shadow']=(df['high']-df[['open','close']].max(axis=1))/df['open']
        df['lower_shadow']=(df[['open','close']].min(axis=1)-df['low'])/df['open']
        typical=(df['high']+df['low']+df['close'])/3.0
        vol=df['tick_volume'].replace(0,np.nan)
        df['vwap']=(vol*typical).cumsum()/vol.cumsum()
        df['vwap'].ffill() 
        df['price_vs_vwap']=(df['close']-df['vwap'])/df['vwap']
        for w in [10,20,50]:
            m=df['close'].rolling(w).mean(); s=df['close'].rolling(w).std()
            df[f'z_score_{w}']=(df['close']-m)/(s+1e-9)
            df[f'percentile_{w}']=df['close'].rolling(w).rank(pct=True)
        if len(df)>=20:
            df['trend_strength']=df['close'].rolling(20).apply(lambda s:np.polyfit(np.arange(len(s)), s.values,1)[0])
        else:
            df['trend_strength']=np.nan
        df['doji']=((df['close']-df['open']).abs()/(df['high']-df['low']+1e-5)<0.1).astype(int)
        df['hammer']=(
            (df['low']<df[['open','close']].min(axis=1)) &
            ((df['high']-df[['open','close']].max(axis=1)) <
             (df[['open','close']].min(axis=1)-df['low']))
        ).astype(int)
    except Exception as e:
        logging.error(f"add_features error: {e}")
    return df

def build_dataset_mt5_per_symbol(symbols, tf_out="M5", label_th=LABEL_THRESHOLD, nan_col_max_ratio=0.4):
    union_cols=set()
    data_per_symbol=[]
    for sym in symbols:
        days=TRAIN_DAYS_BTC if sym in CRYPTO_SET else TRAIN_DAYS_FX
        raw=mt5_fetch_bars(sym, tf_str=tf_out, days=days)
        if raw is None or len(raw)<500:
            logging.warning(f"Skip {sym}: low history")
            continue
        if ('tick_volume' not in raw.columns) or raw['tick_volume'].sum()==0:
            diffs=(raw['close']!=raw['close'].shift(1)).astype(int)
            raw['tick_volume']=diffs.rolling(5,min_periods=1).sum()
        df=add_features(raw)
        feat_cols=[c for c in df.columns if c not in BASE_DROP]
        if not feat_cols: continue
        nan_ratio=df[feat_cols].isna().mean()
        kept=[c for c in feat_cols if nan_ratio[c]<=nan_col_max_ratio]
        if len(kept)<10:
            logging.warning(f"{sym}: too many NaNs")
            continue
        future_ret=df['close'].shift(-5)/df['close']-1
        labels=(future_ret>label_th).astype(int)
        valid=~labels.isna()
        if valid.sum()<400:
            logging.warning(f"{sym}: insufficient labels")
            continue
        feat_df=df.loc[valid, kept].fillna(0).replace([np.inf,-np.inf],0.0)
        y=labels.loc[valid].values
        union_cols.update(kept)
        logging.info(f"{sym}: samples={len(y)}, kept_features={len(kept)}")
        data_per_symbol.append({"symbol":sym,"df":feat_df,"y":y})
    if not data_per_symbol:
        return None,None
    feature_names=sorted(list(union_cols))
    for d in data_per_symbol:
        d["X"]=d["df"].reindex(columns=feature_names, fill_value=0.0).values
        del d["df"]
    return data_per_symbol, feature_names

# === FAST GRID TUNING PATCH (ZAMJENA POSTOJEĆIH GRID FUNKCIJA) ===
from time import time as _time
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HGB_AVAILABLE = True
except:
    HGB_AVAILABLE = False
import random

FAST_GRID = True
GRID_SYMBOL_LIMIT = 4          # koliko simbola uzeti za tuning (ako None -> sve)
GRID_SPLITS = 1                # broj forward splitova (prije je bilo 3)
GRID_RANDOM_SEARCH = True      # True = random subset kombinacija
GRID_RANDOM_MAX_COMBOS = 12    # max kombinacija kod random search
GRID_DOWNSAMPLE_RATE = 2       # uzmi svaku 2. svijeću (None ili 1 = bez downsample)
EARLY_STOP_AFTER = 6           # nakon ovoliko testiranih provjeri early stop
EARLY_STOP_DELTA = 0.005       # ako razlika između max i median AUC mala -> prekid

def _grid_param_space(mode="small"):
    if mode=="medium":
        space = {
            "n_estimators":[100,140,180],
            "learning_rate":[0.045,0.055,0.07],
            "max_depth":[3,4,5],
            "min_samples_leaf":[25,40,60],
            "subsample":[0.8,0.9],
            "max_features":[None]
        }
    else:
        space = {
            "n_estimators":[80,120,160],
            "learning_rate":[0.05,0.07],
            "max_depth":[3,4],
            "min_samples_leaf":[30,50],
            "subsample":[0.8],
            "max_features":[None]
        }
    return space

def _time_series_subsplits(n, parts=1, min_train_frac=0.5):
    res=[]
    for i in range(parts):
        train_end = int(min_train_frac*n + i*((n*(1-min_train_frac))*0.4/max(1,parts)))
        test_start=train_end
        test_end=min(n, test_start + int(0.10*n))
        if test_end-test_start < int(0.04*n): continue
        res.append((np.arange(0,train_end), np.arange(test_start,test_end)))
    if not res:
        split=int(0.75*n)
        res.append((np.arange(0,split), np.arange(split,n)))
    return res

def _generate_param_combos(space, random_mode=True, max_combos=12):
    keys=list(space.keys())
    full=list(product(*[space[k] for k in keys]))
    if not random_mode or len(full)<=max_combos:
        for combo in full:
            yield {k:v for k,v in zip(keys,combo)}
    else:
        seen=set()
        while len(seen)<max_combos:
            c=tuple(space[k][random.randrange(len(space[k]))] for k in keys)
            if c in seen: continue
            seen.add(c)
            yield {k:v for k,v in zip(keys,c)}

def run_grid_tuning(symbols, tf_out="M5", label_th=LABEL_THRESHOLD, mode="small"):
    logging.info(f"⚡ FAST GRID TUNING start (mode={mode}) ...")
    if GRID_SYMBOL_LIMIT:
        symbols = symbols[:GRID_SYMBOL_LIMIT]
        logging.info(f"▶ Subset simbola za tuning: {symbols}")
    data_per_symbol, feature_names = build_dataset_mt5_per_symbol(symbols, tf_out=tf_out, label_th=label_th)
    if data_per_symbol is None:
        logging.error("Grid tuning dataset failed.")
        return None
    # Downsample
    if GRID_DOWNSAMPLE_RATE and GRID_DOWNSAMPLE_RATE>1:
        for d in data_per_symbol:
            Xd=d["X"][::GRID_DOWNSAMPLE_RATE]
            yd=d["y"][::GRID_DOWNSAMPLE_RATE]
            d["X"]=Xd; d["y"]=yd
        logging.info(f"⏬ Downsample x{GRID_DOWNSAMPLE_RATE} primijenjen.")

    space=_grid_param_space(mode)
    combos=list(_generate_param_combos(space, GRID_RANDOM_SEARCH, GRID_RANDOM_MAX_COMBOS))
    logging.info(f"Param combos (actual searched): {len(combos)} (random={GRID_RANDOM_SEARCH})")

    if not os.path.exists(GRID_RESULTS_FILE):
        with open(GRID_RESULTS_FILE,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","rank","auc","acc","precision","recall","profit_proxy",
                                    "tn","fp","fn","tp","samples","train_time_sec","params"])

    results=[]
    start_all=_time()

    # Precompute splits & scaled data (cache) za ubrzanje
    cache=[]
    for d in data_per_symbol:
        X=d["X"]; y=d["y"]
        splits=_time_series_subsplits(len(X), parts=GRID_SPLITS, min_train_frac=0.55)
        sym_blocks=[]
        for tr,te in splits:
            Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
            scaler=StandardScaler().fit(Xtr)
            sym_blocks.append((scaler.transform(Xtr), scaler.transform(Xte), ytr, yte))
        cache.append(sym_blocks)

    def eval_combo(params):
        all_probs=[]; all_true=[]
        total=0
        model_proto=None
        use_hist = FAST_GRID and HGB_AVAILABLE
        for sym_blocks in cache:
            for (Xtr_s,Xte_s,ytr,yte) in sym_blocks:
                if use_hist:
                    model_proto=HistGradientBoostingClassifier(
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        max_iter=params['n_estimators'],
                        l2_regularization=0.0,
                        random_state=42
                    )
                else:
                    model_proto=GradientBoostingClassifier(
                        n_estimators=params['n_estimators'],
                        learning_rate=params['learning_rate'],
                        max_depth=params['max_depth'],
                        min_samples_leaf=params['min_samples_leaf'],
                        subsample=params['subsample'],
                        max_features=params['max_features'],
                        random_state=42
                    )
                model_proto.fit(Xtr_s,ytr)
                p=model_proto.predict_proba(Xte_s)[:,1]
                all_probs.append(p); all_true.append(yte); total+=len(yte)
        if not all_probs: return None
        pr=np.concatenate(all_probs); yt=np.concatenate(all_true)
        try: auc=roc_auc_score(yt,pr)
        except: auc=np.nan
        acc=accuracy_score(yt,(pr>0.5).astype(int))
        cm=confusion_matrix(yt,(pr>0.5).astype(int))
        tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
        precision=tp/(tp+fp+1e-9); recall=tp/(tp+fn+1e-9)
        profit_proxy=(precision*recall) - ((1-precision)*0.4)
        return {
            "auc":auc,"acc":acc,"precision":precision,"recall":recall,
            "profit_proxy":profit_proxy,"tn":tn,"fp":fp,"fn":fn,"tp":tp,"samples":total
        }

    for i,params in enumerate(combos,1):
        t0=_time()
        metrics=eval_combo(params)
        dt=_time()-t0
        if metrics is None:
            logging.warning(f"Skip combo {params}")
            continue
        logging.info(f"[{i}/{len(combos)}] AUC={metrics['auc']:.4f} ACC={metrics['acc']:.4f} "
                     f"PP={metrics['profit_proxy']:.4f} t={dt:.1f}s params={params}")
        metrics["params"]=params
        metrics["train_time_sec"]=dt
        results.append(metrics)

        # Early stop provjera
        if i>=EARLY_STOP_AFTER:
            aucs=[r['auc'] for r in results if not np.isnan(r['auc'])]
            if len(aucs)>=EARLY_STOP_AFTER:
                best=max(aucs); median=np.median(aucs)
                if best - median < EARLY_STOP_DELTA:
                    logging.info(f"⛔ EARLY STOP: best-med={best-median:.4f} < {EARLY_STOP_DELTA}")
                    break

    if not results:
        logging.error("No valid results from tuning.")
        return None
    results.sort(key=lambda r:(np.nan_to_num(r['auc']), r['profit_proxy']), reverse=True)
    ts=datetime.now(timezone.utc).isoformat()
    with open(GRID_RESULTS_FILE,"a",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        for rank,r in enumerate(results,1):
            w.writerow([ts,rank,f"{r['auc']:.5f}",f"{r['acc']:.5f}",f"{r['precision']:.5f}",
                        f"{r['recall']:.5f}",f"{r['profit_proxy']:.5f}",
                        r['tn'],r['fp'],r['fn'],r['tp'],r['samples'],
                        f"{r['train_time_sec']:.2f}",json.dumps(r['params'])])
    best=results[0]
    with open(BEST_PARAM_FILE,"w",encoding="utf-8") as f:
        json.dump(best['params'], f, indent=2)
    total_time=_time()-start_all
    logging.info(f"✅ FAST GRID DONE. Best={best['params']} AUC={best['auc']:.4f} "
                 f"ACC={best['acc']:.4f} CombosTested={len(results)} TotalTime={total_time/60:.1f}m")
    return best['params']
# === END FAST GRID TUNING PATCH ===

# =========================================================
# TRAINING
# =========================================================
def _load_best_params_or_default():
    if FORCE_DEFAULT_PARAMS:
        return None
    if os.path.exists(BEST_PARAM_FILE):
        try:
            with open(BEST_PARAM_FILE,"r",encoding="utf-8") as f:
                p=json.load(f)
            logging.info(f"📦 Using tuned params: {p}")
            return p
        except:
            pass
    return None

def run_training(symbols, tf_out="M5", label_th=LABEL_THRESHOLD):
    data_per_symbol, feature_names = build_dataset_mt5_per_symbol(symbols, tf_out=tf_out, label_th=label_th)
    if data_per_symbol is None:
        logging.error("Training dataset failed (no symbols).")
        return False
    X_train_list=[]; y_train_list=[]
    X_test_list=[]; y_test_list=[]
    for d in data_per_symbol:
        X_sym=d["X"]; y_sym=d["y"]; n=len(X_sym)
        split_idx=int(n*0.70)
        X_train_list.append(X_sym[:split_idx]); y_train_list.append(y_sym[:split_idx])
        X_test_list.append(X_sym[split_idx:]); y_test_list.append(y_sym[split_idx:])
    X_train=np.vstack(X_train_list); y_train=np.concatenate(y_train_list)
    X_test=np.vstack(X_test_list); y_test=np.concatenate(y_test_list)
    scaler=StandardScaler().fit(X_train)
    X_train_s=scaler.transform(X_train); X_test_s=scaler.transform(X_test)
    pos_rate_train=y_train.mean(); pos_rate_test=y_test.mean()
    logging.info(f"Label distribution: Train pos={pos_rate_train:.3f} Test pos={pos_rate_test:.3f}")

    tuned_params=_load_best_params_or_default()
    if tuned_params is None:
        if FAST_MODEL:
            tuned_params={"n_estimators":140,"learning_rate":0.07,"max_depth":5,"min_samples_leaf":30,
                          "subsample":0.8,"max_features":None}
            logging.info(f"Using FAST default params: {tuned_params}")
        else:
            tuned_params={"n_estimators":220,"learning_rate":0.06,"max_depth":6,"min_samples_leaf":30,
                          "subsample":1.0,"max_features":None}
            logging.info(f"Using FULL default params: {tuned_params}")

    model=GradientBoostingClassifier(random_state=42, **tuned_params)
    model.fit(X_train_s,y_train)
    train_pred=model.predict(X_train_s); test_pred=model.predict(X_test_s)
    train_prob=model.predict_proba(X_train_s)[:,1]; test_prob=model.predict_proba(X_test_s)[:,1]
    acc_train=accuracy_score(y_train, train_pred); acc_test=accuracy_score(y_test, test_pred)
    try:
        auc_train=roc_auc_score(y_train, train_prob); auc_test=roc_auc_score(y_test, test_prob)
    except:
        auc_train=auc_test=np.nan
    cm=confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    logging.info(
        f"Training done: TrainAcc={acc_train:.3f} TestAcc={acc_test:.3f} "
        f"TrainAUC={auc_train:.3f} TestAUC={auc_test:.3f} "
        f"SamplesTrain={len(y_train)} SamplesTest={len(y_test)} Feat={len(feature_names)} "
        f"TestCM=[TN={tn} FP={fp} FN={fn} TP={tp}]"
    )
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_names, FEATURES_FILE)
    now=datetime.now(timezone.utc).isoformat()
    with open(LAST_TRAIN_FILE,"w") as f: f.write(now)
    with open(LAST_RETRAIN_ATTEMPT_FILE,"w") as f: f.write(now)
    return True

def _read_timestamp(path):
    if not os.path.exists(path): return None
    try:
        with open(path,"r") as f: return datetime.fromisoformat(f.read().strip())
    except:
        return None

def need_retrain():
    if not AUTO_RETRAIN_ENABLED: return False
    now=datetime.now(timezone.utc)
    if _last_retrain_check["time"] and (now - _last_retrain_check["time"]).total_seconds()<30:
        return False
    _last_retrain_check["time"]=now
    last_train=_read_timestamp(LAST_TRAIN_FILE)
    last_attempt=_read_timestamp(LAST_RETRAIN_ATTEMPT_FILE)
    if not os.path.exists(MODEL_FILE):
        return True
    if last_train is None:
        if last_attempt and (now - last_attempt).total_seconds()<AUTO_RETRAIN_MIN_GAP_MIN*60:
            return False
        return True
    if (now - last_train).days >= AUTO_RETRAIN_DAYS:
        if last_attempt and (now - last_attempt).total_seconds()<AUTO_RETRAIN_MIN_GAP_MIN*60:
            return False
        return True
    return False

def wilson_lower_bound(wins, total, z=1.96):
    if total==0: return 0
    p=wins/total
    denom=1+z*z/total
    centre=p + z*z/(2*total)
    adj=z*math.sqrt((p*(1-p)+z*z/(4*total))/total)
    lower=(centre - adj)/denom
    return lower

# =========================================================
    # UNIFIED BOT - PATCH v2 COMPLETE
    # Zamijeni CIJELU UnifiedBot klasu sa ovom
    # =========================================================

class UnifiedBot:
    def __init__(self):
        """PATCH v2 - Potpuno integrisana verzija"""
        self.symbols = DEFAULT_SYMBOLS
        self.symbol_configs = {
       "EURUSD": EURUSD,
       "GBPUSD": GBPUSD,
       "USDJPY": USDJPY,
       "XAUUSDs": EURUSD,  # Alias
       "BTCUSDT": BTCUSDT,
       "AUDUSD": AUDUSD,
       "NZDUSD": NZDUSD,
       "USDCAD": USDCAD,
       "USDCHF": USDCHF
    }
        self.magic_number = 123456
        self.base_lot_swing = 0.05
        self.min_ml_confidence = 0.55        
        # === PATCH v2 NOVI PARAMETRI ===
        self.closed_positions_cooldown = {}  # (symbol, direction) -> datetime
        self.micro_lot = 0.01
        self.min_confidence_threshold = 0.7
        
        self.partial_tp_enabled = PARTIAL_TP_ENABLED
        self.partial_ratio = 0.4
        self.partial_atr_mult = 1.2
        self.breakeven_enabled = BREAKEVEN_ENABLED
        self.breakeven_atr_mult = 0.8
        self.be_offset_pts = 3
        self.trailing_stop_enabled = TRAILING_ENABLED
        self.trailing_stop_activation = 0.6
        self.trailing_stop_distance = 0.6
        self.trailing_cooldown_sec = 15

        # 🧠 ŽIVI PARAMETRI
        self.ml_feedback = deque(maxlen=2000)
        self.last_regime = {}
        self.atr_ref = {}
        self.median_spread_24h = {}
        self.daily_corr = {}
        self.last_micro_retrain = -1
        self._current_symbol = None 
        self.telegram_token = "YOUR_TELEGRAM_BOT_TOKEN"
        self.telegram_chat_id = "YOUR_TELEGRAM_CHAT_ID"

        self.last_trail_update = {}
        self.partial_done = {}
        self.be_done = {}
        self.trailing_last_sl = {}

        self.swing_tf = mt5_timeframe("M5")
        self.higher_tf = mt5_timeframe("M15")
        self.scalp_tf = mt5_timeframe(SCALP_TIMEFRAME) if SCALP_MODE else None
        self.scalp_confirm_tf = mt5_timeframe(SCALP_CONFIRM_TF) if SCALP_MODE else None

        self.strategy_weights = {
            'ema_cross': 1.4,               # ← +0.4
            'multi_timeframe': 1.5,         # ← +0.5
            'adx_trend': 1.3,               # ← +0.3
            'micro_indicators': 1.5,        # ← nova, visoka
            'price_action': 1.1,            # ← +0.1
            'rsi_divergence': 0.9,          # ← ostaje
            'macd_signal': 0.8,             # ← -0.1
            'bollinger_bands': 0.8,         # ← -0.2
            'stochastic': 0.7,              # ← -0.1
            'support_resistance': 0.7,      # ← -0.1
            'volume_analysis': 0.6,         # ← -0.2
            'fibonacci': 0.6,               # ← -0.1
            'ichimoku': 0.7,                # ← -0.2
            'vwap': 0.6,                    # ← -0.2
            'momentum': 0.7,                # ← -0.1
            'harmonic_patterns': 0.0,       # ← -0.1
        }
        self.strategy_stats = {
            n: {"trades":0,"wins":0,"losses":0,"pnl":0.0,"r_list":deque(maxlen=400),
                "avg_r":0.0,"sharpe":0.0,"disabled_until":None,"paper":False}
            for n in self.strategy_weights
        }
        self.symbol_stats = {
            s: {"trades":0,"wins":0,"losses":0,"pnl":0.0,"r_list":deque(maxlen=500),
                "disabled_until":None,"paper":False,"day_pnl":0.0,"day_date":None}
            for s in self.symbols
        }

        self.last_bar_time_swing = {}
        self.last_bar_time_scalp = {}
        self.last_perf_update = datetime.now(timezone.utc)
        self.last_closed_deal_time = datetime.now(timezone.utc) - timedelta(days=30)

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_trained = False
        self.scalp_trades = {}
        self.meta_candidates = []
        self.meta_candidate_id = 0

        self.initialize_mt5()
        self.initialize_model()
        self.load_stats_persistence()

        if self.is_weekend():
            crypto = [s for s in self.symbols if s in CRYPTO_SET]
            if crypto:
                self.symbols = crypto
                logging.info("Weekend crypto-only mode.")

        self.init_csv_files()

        self.symbol_strategy_boost = {
            "EURUSD":   {"ema_cross": 1.3, "multi_timeframe": 1.4, "adx_trend": 1.2, "ichimoku": 1.0},
            "GBPUSD":   {"bollinger_bands": 1.3, "stochastic": 1.2, "volume_analysis": 1.15, "rsi_divergence": 1.2},
            "USDJPY":   {"adx_trend": 1.4, "multi_timeframe": 1.3, "fibonacci": 1.15, "support_resistance": 1.1},
            "XAUUSD":   {"bollinger_bands": 1.4, "price_action": 1.3, "support_resistance": 1.2, "multi_timeframe": 1.1, "harmonic_patterns": 1.2},
            "BTCUSDT":  {"bollinger_bands": 1.5, "volume_analysis": 1.4, "harmonic_patterns": 1.2, "adx_trend": 1.2},
            "AUDUSD":   {"rsi_divergence": 1.3, "support_resistance": 1.2, "vwap": 1.2, "ema_cross": 1.1},
            "NZDUSD":   {"rsi_divergence": 1.3, "support_resistance": 1.2, "vwap": 1.3, "multi_timeframe": 1.1},
            "USDCAD":   {"adx_trend": 1.3, "momentum": 1.2, "macd_signal": 1.15, "fibonacci": 1.1},
            "USDCHF":   {"rsi_divergence": 1.3, "support_resistance": 1.3, "vwap": 1.3, "multi_timeframe": 1.1},
        
        }
        
    def multi_timeframe_confirmation(self, symbol: str, direction: int, primary_tf_data: pd.DataFrame) -> dict:
        """
        MULTI-TIMEFRAME ANALIZA ZA KONFIRMACIJU
        - Provjerava alignment preko 3 timeframe-a
        - Vraća confidence boost ako su timeframe-ovi aligned
    
        Returns:
            dict: {
                'aligned': bool,
                'confidence_boost': float (0.0 do 0.3),
                'htf_trend': int (-1, 0, 1),
                'mtf_trend': int,
                'details': str
           }
        """
        result = {
            'aligned': False,
            'confidence_boost': 0.0,
            'htf_trend': 0,
            'mtf_trend': 0,
            'details': 'N/A'
        }
    
        try:
            # Higher Timeframe (H1) - trend direction
            htf_data = self.get_data(symbol, count=200, tf=mt5.TIMEFRAME_H1)
            if htf_data is None or len(htf_data) < 100:
                return result
        
            # Calculate HTF trend (EMA 50 vs EMA 200)
            ema_50_htf = talib.EMA(htf_data['close'].values, 50)[-1]
            ema_200_htf = talib.EMA(htf_data['close'].values, 200)[-1]
        
            if ema_50_htf > ema_200_htf * 1.001:  # 0.1% threshold
                htf_trend = 1
            elif ema_50_htf < ema_200_htf * 0.999:
                htf_trend = -1
            else:
                htf_trend = 0
        
            # Middle Timeframe (M15) - momentum
            mtf_data = self.get_data(symbol, count=200, tf=mt5.TIMEFRAME_M15)
            if mtf_data is None or len(mtf_data) < 100:
                result['htf_trend'] = htf_trend
                return result
        
            # Calculate MTF momentum (MACD)
            macd, signal, _ = talib.MACD(mtf_data['close'].values, 12, 26, 9)
            macd_val = macd[-1]
            signal_val = signal[-1]
        
            if macd_val > signal_val and macd_val > 0:
                mtf_trend = 1
            elif macd_val < signal_val and macd_val < 0:
                mtf_trend = -1
            else:
                mtf_trend = 0
        
            # Lower Timeframe (Primary - već imamo)
            ltf_ema_20 = talib.EMA(primary_tf_data['close'].values, 20)[-1]
            ltf_price = primary_tf_data['close'].iloc[-1]
        
            if ltf_price > ltf_ema_20 * 1.0005:
                ltf_trend = 1
            elif ltf_price < ltf_ema_20 * 0.9995:
                ltf_trend = -1
            else:
                ltf_trend = 0
        
            # CHECK ALIGNMENT
            result['htf_trend'] = htf_trend
            result['mtf_trend'] = mtf_trend
        
            # Perfect alignment (svi timeframe-ovi u istom smjeru)
            if htf_trend == direction and mtf_trend == direction and ltf_trend == direction:
                result['aligned'] = True
                result['confidence_boost'] = 0.25  # 25% boost
                result['details'] = f"Perfect alignment (HTF:{htf_trend}, MTF:{mtf_trend}, LTF:{ltf_trend})"
        
            # Partial alignment (2 od 3)
            elif (htf_trend == direction and mtf_trend == direction) or \
                (htf_trend == direction and ltf_trend == direction) or \
                (mtf_trend == direction and ltf_trend == direction):
                result['aligned'] = True
                result['confidence_boost'] = 0.15  # 15% boost
                result['details'] = f"Partial alignment (HTF:{htf_trend}, MTF:{mtf_trend}, LTF:{ltf_trend})"
        
            # Weak alignment (1 od 3, ali nema suprotnih)
            elif htf_trend == direction or mtf_trend == direction or ltf_trend == direction:
                conflicting = any([
                    htf_trend == -direction,
                    mtf_trend == -direction,
                    ltf_trend == -direction
                ])
                if not conflicting:
                    result['aligned'] = True
                    result['confidence_boost'] = 0.08  # 8% boost
                    result['details'] = f"Weak alignment (HTF:{htf_trend}, MTF:{mtf_trend}, LTF:{ltf_trend})"
                else:
                    result['details'] = f"Conflicting timeframes (HTF:{htf_trend}, MTF:{mtf_trend}, LTF:{ltf_trend})"
        
            else:
                result['details'] = f"No alignment (HTF:{htf_trend}, MTF:{mtf_trend}, LTF:{ltf_trend})"
    
        except Exception as e:
            logging.error(f"MTF confirmation error for {symbol}: {e}")
            result['details'] = f"Error: {e}"
    
        return result
    
    def calculate_optimal_lot_size(self, symbol: str, tier: int, confidence: float, 
                                 sl_distance_pips: float) -> float:
        """
        OPTIMALNO LOT SIZING SA KELLY CRITERION
        - Dinamičko skaliranje baziran na tier, confidence i riziku
        - Kelly criterion za position sizing (konzervativna verzija)
        - Account balance aware
        """
        cfg = self.symbol_configs.get(symbol, self.symbol_configs["EURUSD"])
        base_lot = cfg.get('lot_size', 0.10)
    
        # Uzmi strategy stats za ovaj simbol
        stats = self.strategy_stats.get(symbol, {})
    
        # Izračunaj istorijsku win rate i avg win/loss
        total_trades = sum(s.get('total', 0) for s in stats.values())
        total_wins = sum(s.get('wins', 0) for s in stats.values())
    
        if total_trades >= 20:  # Dovoljan sample za Kelly
            win_rate = total_wins / total_trades
        
            # Procijeni avg win/loss ratio (proxy)
            # U nedostatku detaljnih podataka, koristimo RR ratio
            avg_rr = 2.0  # Default assumption (2:1 RR)
        
            # Kelly formula (konzervativna verzija - polovina Kelly-a)
            # f = (p * b - q) / b, gdje je:
            # p = win rate, q = loss rate, b = win/loss ratio
            q = 1 - win_rate
            kelly_fraction = ((win_rate * avg_rr) - q) / avg_rr
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap na 25%
        
            # Polovina Kelly-a za konzervativnost
            kelly_half = kelly_fraction * 0.5
        else:
            # Nedovoljan historijski podatak - fixed sizing
            kelly_half = 0.10  # 10% baseline
    
        # TIER MULTIPLIER
        tier_multipliers = {
            1: 1.5,   # Vrhunski setup - povećaj lot
            2: 1.0,   # Standard lot
            3: 0.7    # Slabiji setup - smanji lot
        }
        tier_mult = tier_multipliers.get(tier, 1.0)
    
        # CONFIDENCE MULTIPLIER (dodatni scaling)
        # Visoka confidence = veća pozicija
        if confidence >= 0.80:
            conf_mult = 1.3
        elif confidence >= 0.70:
            conf_mult = 1.15
        elif confidence >= 0.60:
            conf_mult = 1.0
        else:
            conf_mult = 0.85
    
        # RISK-BASED SCALING (veći SL = manji lot)
        # Normaliziraj prema "normalnom" SL-u od 15 pips
        normal_sl_pips = 15.0
        risk_adjustment = min(1.5, normal_sl_pips / max(5.0, sl_distance_pips))
    
        # KOMBINIRAJ SVE FAKTORE
        optimal_lot = base_lot * kelly_half * tier_mult * conf_mult * risk_adjustment
    
        # ENFORCE LIMITS
        min_lot = 0.01
        max_lot = base_lot * 3.0  # Nikad više od 3x base lot
    
        optimal_lot = max(min_lot, min(max_lot, optimal_lot))
    
        # Zaokruži na 0.01 (standard lot step)
        optimal_lot = round(optimal_lot, 2)
    
        logging.info(f"💰 Optimal lot for {symbol} tier {tier}: {optimal_lot:.2f} "
                f"(Kelly: {kelly_half:.2f}, Tier: {tier_mult:.1f}x, Conf: {conf_mult:.1f}x, Risk: {risk_adjustment:.2f}x)")
    
        return optimal_lot
    
    def get_point(self, symbol):
        """Vraća point value za simbol"""
        try:
            info = mt5.symbol_info(symbol)
            if info:
                return info.point
            return 0.0001  # Default za FX
        except:
            return 0.0001


    # === POSTOJEĆE METODE (ostaju iste) ===
    def save_stats_persistence(self):
        try:
            data = {"strategy_stats":self.strategy_stats,"symbol_stats":self.symbol_stats}
            with open(STATS_PERSIST_FILE,"wb") as f:
                pickle.dump(data,f)
        except Exception as e:
            logging.error(f"save_stats_persistence error: {e}")
            
        # ========== 0. per-symbol params ==========
    def _symbol_params(self, symbol: str):
        """Vraća dict s parametrima za simbol; fallback EURUSD."""
        return SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS["EURUSD"])
    
    def _is_trade_time(self, symbol):
        """Vraća True ako je dopušteno trgovati po vremenu."""
        p = self._symbol_params(symbol)
        now = datetime.now(timezone.utc)

        # dan u tjednu
        if now.weekday() not in p.get("trade_days", [0, 1, 2, 3, 4]):
            return False

        # sati
        start_h, end_h = p.get("trade_hours", (0, 24))
        if not (start_h <= now.hour < end_h):
            return False

        # TODO: kasnije proširiti s avoid_news
        return True
    
    def load_stats_persistence(self):
        if not os.path.exists(STATS_PERSIST_FILE): return
        try:
            with open(STATS_PERSIST_FILE,"rb") as f:
                data = pickle.load(f)
            if "strategy_stats" in data:
                for k,v in data["strategy_stats"].items():
                    if k in self.strategy_stats: self.strategy_stats[k].update(v)
            if "symbol_stats" in data:
                for k,v in data["symbol_stats"].items():
                    if k in self.symbol_stats: self.symbol_stats[k].update(v)
            logging.info("📦 Loaded persisted stats.")
        except Exception as e:
            logging.error(f"load_stats_persistence error: {e}")

    def initialize_mt5(self):
        if mt5 is None:
            logging.error("MetaTrader5 modul nije instaliran.")
            raise SystemExit(1)
        if not mt5.initialize():
            login,password,server = load_env_credentials()
            if login and password and server:
                try: login_i = int(login)
                except: login_i = login
                if not mt5.initialize(login=login_i,password=password,server=server):
                    logging.error("MT5 init failed (credentials).")
                    raise SystemExit(1)
            else:
                logging.error("MT5 init failed (no credentials).")
                raise SystemExit(1)
        logging.info("MT5 connected.")
        for s in self.symbols:
            try: mt5.symbol_select(s, True)
            except: pass

    def initialize_model(self):
        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            if os.path.exists(FEATURES_FILE):
                self.feature_names = joblib.load(FEATURES_FILE)
            self.model_trained = True
            logging.info("🧠 ML model učitan.")
        except Exception as e:
            logging.warning(f"Model nije učitan: {e}")
            self.model_trained = False

    def init_csv_files(self):
        if ENABLE_SIGNAL_CSV and not os.path.exists(SIGNAL_LOG_FILE):
            with open(SIGNAL_LOG_FILE,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp","symbol","mode","meta_score","ml_prob","final_decision",
                                        "tier","conflict","cat_scores","top_strategies"])
        if ENABLE_TRADE_CSV and not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp","symbol","ticket","direction","lot","entry_price",
                                        "sl","tp","mode","tier","strategies"])
        if ENABLE_META_LABEL_LOG:
            if not os.path.exists(META_LABEL_CANDIDATES_FILE):
                with open(META_LABEL_CANDIDATES_FILE,"w",newline="",encoding="utf-8") as f:
                    csv.writer(f).writerow(["cand_id","symbol","timestamp","direction","entry_price","sl","tp",
                                            "mode","target_pips","sl_pips","bar_index"])
            if not os.path.exists(META_LABEL_FINAL_FILE):
                with open(META_LABEL_FINAL_FILE,"w",newline="",encoding="utf-8") as f:
                    csv.writer(f).writerow(["cand_id","symbol","direction","label","mfe_pips","mae_pips",
                                            "target_pips","sl_pips","bars_elapsed"])

    def is_weekend(self):
        return datetime.now(timezone.utc).weekday() >= 5

    def get_data(self, symbol, count=800, tf=None):
        try:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None or len(rates)==0: return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            return df
        except Exception as e:
            logging.error(f"get_data error {symbol}: {e}")
            return None

    def calc_lot_from_tier(self, tier):
        p = self._symbol_params(self._current_symbol)
        return p["lot_size"]

    def _sanitize_comment(self, text: str) -> str:
        """Sanitize comment text for MT5 compatibility."""
        try:
            text_str = str(text) if text is not None else ""
            sanitized = re.sub(r'[^A-Za-z0-9 _-]', '_', text_str)
            return sanitized[:31]
        except Exception as e:
            logging.warning(f"Comment sanitization error: {e}")
            return "sanitized_comment"
        
        # ===================================================================
        # 1. DYNAMIC LOT – po volatilnosti i win-rate-u
        # ===================================================================
    def calc_dynamic_lot(self, symbol, base_lot):
        stats = self.symbol_stats.get(symbol, {})
        atr = self.last_regime.get(symbol, {}).get("atr", 0.0001)
        wr = stats.get('wins', 0) / max(1, stats.get('trades', 0))
        vol_mult = min(1.5, max(0.3, 0.8 / (atr / 0.0001 + 0.01)))
        perf_mult = 0.5 + wr
        return round(base_lot * vol_mult * perf_mult, 2)

        # ===================================================================
        # 2. TELEGRAM NOTIFIKACIJE
        # ===================================================================
    def telegram_send(self, msg):
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {"chat_id": self.telegram_chat_id, "text": msg}
            requests.post(url, data=payload, timeout=2)
        except:
            pass

        # ===================================================================
        # 3. MICRO-RETRAIN – svakih 6 sati na feedback-u
        # ===================================================================
    def micro_retrain_on_feedback(self):
        if len(self.ml_feedback) < 200:
            return
        df = pd.DataFrame(self.ml_feedback)
        # Jednostavan target: profit > 0
        df['label'] = (df['profit'] > 0).astype(int)
        # Feature: direction + regime + bars_held
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse=False)
        X = enc.fit_transform(df[['direction', 'regime']].values)
        y = df['label'].values
        from sklearn.ensemble import GradientBoostingClassifier
        micro_model = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
        micro_model.fit(X, y)
        # Sačuvaj
        joblib.dump(micro_model, "micro_model.pkl")
        logging.info("✅ Micro-retrain done – model updated")

        # Ostatak postojećih metoda nastavi koristiti kako jesu...
    def modify_position(self, position, new_sl, new_tp=None):
        try:
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return False

            info = mt5.symbol_info(position.symbol)
            if info:
                point        = info.point
                stops_level  = max(info.trade_stops_level, 10)
                min_dist     = stops_level * point
                current_price= tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid
                digits       = info.digits

                if position.type == mt5.ORDER_TYPE_BUY:
                    if new_sl >= current_price - min_dist:
                        new_sl = current_price - min_dist - point
                else:
                    if new_sl <= current_price + min_dist:
                        new_sl = current_price + min_dist + point
                new_sl = round(new_sl, digits)

            if position.sl is not None and abs(new_sl - position.sl) < info.point:
                return True

            comment = self._sanitize_comment(f"Trail-{position.comment}")

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "magic": position.magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC
            }
            if new_tp is not None:
                req["tp"] = new_tp

            res = mt5.order_send(req)
            if res is None:
                last_error = mt5.last_error()
                logging.error(f"Modify failed with None response for {position.symbol}, last_error: {last_error}")
                return False

            if res.retcode != mt5.TRADE_RETCODE_DONE:
                last_error = mt5.last_error()
                logging.error(f"Modify fail {position.symbol}: retcode={res.retcode}, comment={res.comment}, last_error={last_error}")
                return False

            logging.info(f"🔄 SL updated {position.symbol}: {new_sl}")
            return True

        except Exception as e:
            logging.error(f"modify_position error: {e}")
            return False

    def close_position_market(self, position, comment="ManualClose"):
        try:
            tick=mt5.symbol_info_tick(position.symbol)
            if not tick: return False
            opposite=mt5.ORDER_TYPE_SELL if position.type==mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price=tick.bid if opposite==mt5.ORDER_TYPE_SELL else tick.ask
            req={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": opposite,
                "position": position.ticket,
                "price": price,
                "deviation": 30,
                "magic": position.magic,
                "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC
            }
            res=mt5.order_send(req)
            return res and res.retcode==mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logging.error(f"close_position_market error: {e}")
            return False

    def close_partial_position(self, position, ratio=0.4):
        """Zatvara dio pozicije za partial TP"""
        if self.partial_done.get(position.ticket, False):
            return False
    
        try:
            info = mt5.symbol_info(position.symbol)
            if not info:
                return False
        
            step = info.volume_step if info.volume_step > 0 else 0.01
            close_vol = round(position.volume * ratio / step) * step
            close_vol = max(info.volume_min, min(close_vol, position.volume - info.volume_min + 1e-9))
        
            if close_vol <= 0:
                return False
        
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return False
        
            opposite = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if opposite == mt5.ORDER_TYPE_SELL else tick.ask
        
            # Filling mode - RETURN za sve simbole
            type_filling = mt5.ORDER_FILLING_FOK
        
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": float(close_vol),
                "type": opposite,
                "position": position.ticket,
                "price": price,
                "deviation": 30,
                "magic": self.magic_number,
                "comment": "PartialTP",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling
            }
        
            res = mt5.order_send(req)
        
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                self.partial_done[position.ticket] = True
                logging.info(f"🏁 Partial TP {position.symbol}: closed {close_vol} lots")
                return True
            else:
                comment = res.comment if res else "No response"
                logging.warning(f"Partial TP fail {position.symbol}: {comment}")
                return False
            
        except Exception as e:
            logging.error(f"close_partial_position error: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    def manage_swing_positions(self):
        """
        Enhanced swing trailing sa multi-stage pristupom
        """
        try:
            positions = mt5.positions_get()
            if not positions:
                return
        
            for p in positions:
                # Filter non-swing positions
                if p.magic != self.magic_number:
                    continue
                if p.comment.startswith("SCALP-"):
                    continue
            
                # Cooldown check
                now = datetime.now(timezone.utc)
                if p.ticket in self.last_trail_update:
                    elapsed = (now - self.last_trail_update[p.ticket]).total_seconds()
                    if elapsed < self.trailing_cooldown_sec:
                        continue
            
                # Get market data
                info = mt5.symbol_info(p.symbol)
                if not info:
                    continue
            
                tick = mt5.symbol_info_tick(p.symbol)
                if not tick:
                    continue
            
                digits = info.digits
                point = info.point
                cp = tick.ask if p.type == mt5.ORDER_TYPE_BUY else tick.bid
            
                # Get ATR
                df_sym = self.get_data(p.symbol, count=140, tf=self.swing_tf)
                if df_sym is None or len(df_sym) < 40:
                    continue
            
                df_sym = add_features(df_sym)
                atr = df_sym['atr'].iloc[-1] if 'atr' in df_sym.columns else 10 * point
                atr_pts = atr / point
            
                # Calculate profit in points
                if p.type == mt5.ORDER_TYPE_BUY:
                    profit_pts = (cp - p.price_open) / point
                else:
                    profit_pts = (p.price_open - cp) / point
            
                # === STAGE 1: BREAKEVEN (1.0×ATR profit) ===
                if self.breakeven_enabled and not self.be_done.get(p.ticket, False):
                    breakeven_trigger = 1.0 * atr_pts  # ← PROMIJENJENO sa 0.8 na 1.0
                
                    if profit_pts >= breakeven_trigger:
                        # Move to entry + small buffer
                        be_offset = max(3, atr_pts * 0.05)  # 3 pts minimum ili 5% ATR
                    
                        if p.type == mt5.ORDER_TYPE_BUY:
                            be_price = p.price_open + be_offset * point
                        else:
                            be_price = p.price_open - be_offset * point
                    
                        be_price = round(be_price, digits)
                    
                        # Check if better than current SL
                        should_update = (
                            p.sl is None or
                            (p.type == mt5.ORDER_TYPE_BUY and be_price > p.sl) or
                            (p.type == mt5.ORDER_TYPE_SELL and be_price < p.sl)
                        )
                    
                        if should_update and self.modify_position(p, be_price):
                            self.be_done[p.ticket] = True
                            self.last_trail_update[p.ticket] = now
                            logging.info(f"🔒 BREAKEVEN: {p.symbol} @ {profit_pts:.1f} pts profit")
                            continue
            
                # === STAGE 2: PARTIAL TP (1.5×ATR profit) ===
                if self.partial_tp_enabled and not self.partial_done.get(p.ticket, False):
                    partial_trigger = 1.5 * atr_pts  # ← PROMIJENJENO sa 1.2 na 1.5
                
                    if profit_pts >= partial_trigger:
                        if self.close_partial_position(p, self.partial_ratio):
                            self.last_trail_update[p.ticket] = now
                            logging.info(f"🏁 PARTIAL TP: {p.symbol} @ {profit_pts:.1f} pts")
                            continue
            
                # === STAGE 3: CONSERVATIVE TRAILING (2.0×ATR profit) ===
                if self.trailing_stop_enabled:
                    conservative_trigger = 2.0 * atr_pts
                    conservative_distance = 1.2 * atr  # Široki trailing
                
                    if profit_pts >= conservative_trigger and profit_pts < 4.0 * atr_pts:
                        if p.type == mt5.ORDER_TYPE_BUY:
                            target_sl = cp - conservative_distance
                        else:
                            target_sl = cp + conservative_distance
                    
                        target_sl = round(target_sl, digits)
                    
                        # Check minimum move (0.3×ATR)
                        min_move = 0.3 * atr
                    
                        should_update = False
                        if p.sl is not None:
                            sl_diff = abs(target_sl - p.sl)
                            if sl_diff >= min_move:
                                if p.type == mt5.ORDER_TYPE_BUY and target_sl > p.sl:
                                    should_update = True
                                elif p.type == mt5.ORDER_TYPE_SELL and target_sl < p.sl:
                                    should_update = True
                        else:
                            should_update = True
                    
                        if should_update and self.modify_position(p, target_sl):
                            self.last_trail_update[p.ticket] = now
                            distance_pts = conservative_distance / point
                            logging.info(f"🐢 CONSERVATIVE TRAIL: {p.symbol} @ {profit_pts:.1f} pts, distance={distance_pts:.1f} pts")
                            continue
            
                # === STAGE 4: AGGRESSIVE TRAILING (4.0×ATR profit) ===
                if self.trailing_stop_enabled:
                    aggressive_trigger = 4.0 * atr_pts
                    aggressive_distance = 0.7 * atr  # Uži trailing
                
                    if profit_pts >= aggressive_trigger:
                        if p.type == mt5.ORDER_TYPE_BUY:
                            target_sl = cp - aggressive_distance
                        else:
                            target_sl = cp + aggressive_distance
                    
                        target_sl = round(target_sl, digits)
                    
                        # Check minimum move (0.2×ATR)
                        min_move = 0.2 * atr
                    
                        should_update = False
                        if p.sl is not None:
                            sl_diff = abs(target_sl - p.sl)
                            if sl_diff >= min_move:
                                if p.type == mt5.ORDER_TYPE_BUY and target_sl > p.sl:
                                    should_update = True
                                elif p.type == mt5.ORDER_TYPE_SELL and target_sl < p.sl:
                                    should_update = True
                        else:
                            should_update = True
                    
                        if should_update and self.modify_position(p, target_sl):
                            self.last_trail_update[p.ticket] = now
                            distance_pts = aggressive_distance / point
                            logging.info(f"🚀 AGGRESSIVE TRAIL: {p.symbol} @ {profit_pts:.1f} pts, distance={distance_pts:.1f} pts")
                            continue
    
        except Exception as e:
            logging.error(f"manage_swing_positions error: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")


    def manage_scalps(self):
        """
        Enhanced scalp management - manje agresivno early abort
        """
        if not SCALP_MODE or not self.scalp_trades:
            return
    
        to_remove = []
    
        for oid, st in list(self.scalp_trades.items()):
            positions = mt5.positions_get(ticket=oid)
        
            if not positions:
                to_remove.append(oid)
                continue
        
            p = positions[0]
            info = mt5.symbol_info(st['symbol'])
        
            if not info:
                continue
        
            tick = mt5.symbol_info_tick(st['symbol'])
        
            if not tick:
                continue
        
            pip_val, dig, pip_size = self.get_pip_value_and_digits(st['symbol'])
            cp = tick.bid if p.type == mt5.ORDER_TYPE_SELL else tick.ask
        
            # Calculate P&L
            if p.type == mt5.ORDER_TYPE_BUY:
                diff = cp - st['entry_price']
            else:
                diff = st['entry_price'] - cp
        
            diff_pips = diff / pip_size
        
            # Track max favorable/adverse
            st['max_fav'] = max(st['max_fav'], diff_pips)
            st['max_adv'] = max(st['max_adv'], -diff_pips if diff_pips < 0 else 0)
        
            # Calculate elapsed time
            bars_elapsed = int((datetime.now(timezone.utc) - st['opened']).total_seconds() / 60)
        
            # === EARLY ABORT (manje agresivno) ===
            # Samo ako ide jako loše u prva 3 minuta
            if bars_elapsed <= 3 and st['max_adv'] > 8:  # ← PROMIJENJENO sa 5 na 8 pips
                self.close_position_market(p, "EarlyAbort")
                logging.info(f"⛔ Early abort {st['symbol']} adv={st['max_adv']:.1f}p after {bars_elapsed}min")
                to_remove.append(oid)
                continue
        
            # === TIME STOP (produljen rok) ===
            if bars_elapsed >= 30:  # ← PROMIJENJENO sa 20 na 30 min
                self.close_position_market(p, "TimeStop")
                logging.info(f"⌛ Time-stop {st['symbol']} fav={st['max_fav']:.1f}p after {bars_elapsed}min")
                to_remove.append(oid)
                continue
        
            # === BREAKEVEN (optimizovano) ===
            # Na +6 pips (ranije je bilo +8)
            breakeven_trigger = 6  # ← PROMIJENJENO sa 8 na 6 pips
        
            if diff_pips >= breakeven_trigger:
                # Check if not already at breakeven
                if p.sl is None or abs(p.sl - st['entry_price']) > 1e-5:
                    be_price = st['entry_price'] + (2 * pip_size) if p.type == mt5.ORDER_TYPE_BUY else st['entry_price'] - (2 * pip_size)
                
                    if self.modify_position(p, round(be_price, dig)):
                        logging.info(f"🔒 Scalp BE: {st['symbol']} @ {diff_pips:.1f}p (after {bars_elapsed}min)")
        
            # === TRAILING (dodano!) ===
            # Na +12 pips profit, aktiviraj trailing
            trailing_trigger = 12  # pips
            trailing_distance = 6  # pips
        
            if diff_pips >= trailing_trigger:
                if p.type == mt5.ORDER_TYPE_BUY:
                    trail_sl = cp - (trailing_distance * pip_size)
                    if p.sl is None or trail_sl > p.sl:
                        if self.modify_position(p, round(trail_sl, dig)):
                            logging.info(f"🔄 Scalp trail: {st['symbol']} @ {diff_pips:.1f}p, distance={trailing_distance}p")
                else:
                    trail_sl = cp + (trailing_distance * pip_size)
                    if p.sl is None or trail_sl < p.sl:
                        if self.modify_position(p, round(trail_sl, dig)):
                            logging.info(f"🔄 Scalp trail: {st['symbol']} @ {diff_pips:.1f}p, distance={trailing_distance}p")
    
        # Cleanup closed positions
        for r in to_remove:
            self.scalp_trades.pop(r, None)

    def update_performance_stats(self):
        # Ova funkcija ostaje ista kao u originalnoj verziji
        now=datetime.now(timezone.utc)
        

    def extract_features(self, df):
        df = add_features(df)
        if self.feature_names is None:
            cols = [c for c in df.columns if c not in BASE_DROP]
        else:
            cols = self.feature_names
            for c in cols:
                if c not in df.columns: df[c] = np.nan
        row = df[cols].iloc[-1].fillna(0).replace([np.inf,-np.inf],0).values.reshape(1,-1)
        if self.scaler is not None:
            try: row = self.scaler.transform(row)
            except: pass
        return row

    def _pack(self,name,category,direction,confidence,reasons):
        return {"name":name,"category":category,"raw_direction":direction,"confidence":confidence,"reasons":reasons}

    # === PATCH v2: NOVE METODE ===
    
    def _trend_filter_boost(self, symbol, direction, df):
        """Trend boost: 1.5x u smjeru, 0.7x protiv"""
        try:
            ema21 = df['ema21'].iloc[-1] if 'ema21' in df.columns else None
            ema50 = df['ema50'].iloc[-1] if 'ema50' in df.columns else None
            
            if ema21 is None or ema50 is None or pd.isna(ema21) or pd.isna(ema50):
                return 1.0
                
            if ema21 > ema50:
                return 1.5 if direction == 'buy' else 0.7
            elif ema21 < ema50:
                return 1.5 if direction == 'sell' else 0.7
            else:
                return 1.0
        except:
            return 1.0

    def strategy_micro_indicators(self, df):
        """PATCH v2: Mikro indikatori RSI(3), Stoch(3,1,1), Mom(2), Z(3)"""
        try:
            if len(df) < 5:
                return self._pack("micro_indicators", "momentum", 0, 0.0, ["len<5"])
                
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            rsi3 = talib.RSI(close_prices, timeperiod=3)[-1]
            stochK, stochD = talib.STOCH(high_prices, low_prices, close_prices, 
                                         fastk_period=3, slowk_period=1, slowd_period=1)
            stochK = stochK[-1]
            mom2 = (df['close'].iloc[-1] - df['close'].iloc[-3]) / (df['close'].iloc[-3] + 1e-9)
            recent_close = df['close'].iloc[-3:]
            z3 = (df['close'].iloc[-1] - recent_close.mean()) / (recent_close.std() + 1e-9)
            
            d = 0
            reasons = []
            confidence = 0.0
            
            if (rsi3 < 30) and (stochK < 20) and (mom2 > 0) and (z3 < -1):
                d = 1
                reasons.append(f"micro_buy")
                confidence = 0.85
            elif (rsi3 > 70) and (stochK > 80) and (mom2 < 0) and (z3 > 1):
                d = -1
                reasons.append(f"micro_sell")
                confidence = 0.85
                
            return self._pack("micro_indicators", "momentum", d, confidence, reasons)
        except Exception as e:
            return self._pack("micro_indicators", "momentum", 0, 0.0, [f"err:{e}"])

    def _check_direction_cooldown(self, symbol, direction):
        """PATCH v2: Provjeri cooldown"""
        key = (symbol, direction)
        cooldown_until = self.closed_positions_cooldown.get(key)
        if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
            return False
        return True
        
    def _set_closed_position_cooldown(self, symbol, direction, minutes=10):
        """PATCH v2: Postavi cooldown"""
        key = (symbol, direction)
        self.closed_positions_cooldown[key] = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        logging.info(f"Cooldown {symbol} {direction}: {minutes}min")

    def check_closed_positions(self):
        """PATCH v2: Skenira zatvorene pozicije"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)
            deals = mt5.history_deals_get(start_time, end_time)
            
            if not deals:
                return
                
            for deal in deals:
                deal_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
                time_diff = (end_time - deal_time).total_seconds()
                
                if (deal.magic == self.magic_number and 
                    deal.entry == mt5.DEAL_ENTRY_OUT and
                    time_diff < 60):
                    
                    symbol = deal.symbol
                    direction = 'buy' if deal.type == mt5.DEAL_TYPE_SELL else 'sell'
                    key = (symbol, direction)
                    if key not in self.closed_positions_cooldown:
                        self._set_closed_position_cooldown(symbol, direction)
        except:
            pass

    def can_open_position(self, symbol, direction):
        """PATCH v2: Max 2 pozicije, isti smjer, cooldown"""
        if not self._check_direction_cooldown(symbol, direction):
            return False
            
        try:
            open_positions = mt5.positions_get(symbol=symbol)
        except:
            open_positions = None
            
        if not open_positions:
            return True
            
        my_positions = [p for p in open_positions if p.magic == self.magic_number]
        
        if len(my_positions) >= 2:
            return False
            
        for pos in my_positions:
            pos_direction = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
            if pos_direction != direction:
                return False
                
        return True

    # === POSTOJEĆE STRATEGIJE (ostaju iste) ===
    def strategy_ema_cross(self, df):
        try:
            ema8=df['ema8'].iloc[-3:]; ema21=df['ema21'].iloc[-3:]; ema50=df['ema50'].iloc[-3:]
            d=0;r=[]
            if ema8.iloc[-1]>ema21.iloc[-1] and ema8.iloc[-2]<=ema21.iloc[-2] and ema21.iloc[-1]>ema50.iloc[-1]:
                d=1;r.append("bull_cross")
            elif ema8.iloc[-1]<ema21.iloc[-1] and ema8.iloc[-2]>=ema21.iloc[-2] and ema21.iloc[-1]<ema50.iloc[-1]:
                d=-1;r.append("bear_cross")
            return self._pack("ema_cross","trend",d,0.8 if d else 0.0,r)
        except: return self._pack("ema_cross","trend",0,0.0,["err"])
        
    def strategy_micro_entry(self, df):
        """
        Mikro-ulazni filter: RSI(3), Stoch(3,1,1), Mom(2), Z-score(3)
        Vraća 1 (buy), -1 (sell), 0 (neutral)
        """
        try:
            if len(df) < 5:
                return 0
            close = df['close'].values
            high  = df['high'].values
            low   = df['low'].values

            rsi3  = talib.RSI(close, 3)[-1]
            stochK, _ = talib.STOCH(high, low, close,
                                    fastk_period=3, slowk_period=1, slowd_period=1)
            stochK = stochK[-1]
            mom2  = (close[-1] - close[-3]) / (close[-3] + 1e-9)
            z3    = (close[-1] - close[-3:].mean()) / (close[-3:].std() + 1e-9)

            if rsi3 < 30 and stochK < 20 and mom2 > 0 and z3 < -1.5:
                return 1
            if rsi3 > 70 and stochK > 80 and mom2 < 0 and z3 > 1.5:
                return -1
            return 0
        except Exception as e:
            return 0
        
    def strategy_rsi_divergence(self, df):
        try:
            rsi=df['rsi14'].iloc[-5:]; price=df['close'].iloc[-5:]
            d=0;r=[]
            if len(rsi)<5: return self._pack("rsi_divergence","meanrev",0,0,[])
            if rsi.iloc[-1]>rsi.iloc[-3] and price.iloc[-1]<price.iloc[-3] and rsi.iloc[-1]<45:
                d=1;r.append("bull_div")
            elif rsi.iloc[-1]<rsi.iloc[-3] and price.iloc[-1]>price.iloc[-3] and rsi.iloc[-1]>55:
                d=-1;r.append("bear_div")
            return self._pack("rsi_divergence","meanrev",d,0.7 if d else 0.0,r)
        except: return self._pack("rsi_divergence","meanrev",0,0.0,["err"])
        
    def strategy_macd_signal(self, df):
        try:
            hist=df['hist1'].iloc[-3:]; macd=df['macd1'].iloc[-3:]
            d=0;r=[]
            if hist.iloc[-1]>0 and hist.iloc[-2]<=0 and macd.iloc[-1]>macd.iloc[-2]:
                d=1;r.append("macd_bull")
            elif hist.iloc[-1]<0 and hist.iloc[-2]>=0 and macd.iloc[-1]<macd.iloc[-2]:
                d=-1;r.append("macd_bear")
            return self._pack("macd_signal","momentum",d,0.6 if d else 0.0,r)
        except: return self._pack("macd_signal","momentum",0,0.0,["err"])
        
    def strategy_bollinger_bands(self, df):
        try:
            bb_w=df['bb_width'].iloc[-10:]; bb_p=df['bb_percent'].iloc[-3:]
            d=0;r=[]
            if bb_w.iloc[-1]<bb_w.mean()*0.75:
                if bb_p.iloc[-1]>0.8 and bb_p.iloc[-2]<=0.8: d=1;r.append("upper_break")
                elif bb_p.iloc[-1]<0.2 and bb_p.iloc[-2]>=0.2: d=-1;r.append("lower_break")
            return self._pack("bollinger_bands","volatility",d,0.7 if d else 0.0,r)
        except: return self._pack("bollinger_bands","volatility",0,0.0,["err"])
        
    def strategy_stochastic(self, df):
        try:
            k=df['stoch_k'].iloc[-3:]; d_line=df['stoch_d'].iloc[-3:]
            d=0;r=[]
            if k.iloc[-1]>d_line.iloc[-1] and k.iloc[-2]<=d_line.iloc[-2] and k.iloc[-1]<35:
                d=1;r.append("stoch_bull_cross")
            elif k.iloc[-1]<d_line.iloc[-1] and k.iloc[-2]>=d_line.iloc[-2] and k.iloc[-1]>65:
                d=-1;r.append("stoch_bear_cross")
            return self._pack("stochastic","momentum",d,0.6 if d else 0.0,r)
        except: return self._pack("stochastic","momentum",0,0.0,["err"])
        
    def strategy_adx_trend(self, df):
        try:
            adx=df['adx'].iloc[-1]; pdi=df['plus_di'].iloc[-1]; mdi=df['minus_di'].iloc[-1]
            d=0;r=[]
            if adx>20:
                if pdi>mdi: d=1;r.append("adx_bull")
                elif mdi>pdi: d=-1;r.append("adx_bear")
            return self._pack("adx_trend","trend",d,0.7 if d else 0.0,r)
        except: return self._pack("adx_trend","trend",0,0.0,["err"])
        
    def strategy_support_resistance(self, df):
        try:
            high_20=df['high'].rolling(20).max().iloc[-1]
            low_20=df['low'].rolling(20).min().iloc[-1]
            cp=df['close'].iloc[-1]; prev=df['close'].iloc[-2]
            d=0;r=[]
            if abs(cp-low_20)/cp<0.003 and cp>prev: d=1;r.append("support_bounce")
            elif abs(cp-high_20)/cp<0.003 and cp<prev: d=-1;r.append("resistance_reject")
            return self._pack("support_resistance","meanrev",d,0.55 if d else 0.0,r)
        except: return self._pack("support_resistance","meanrev",0,0.0,["err"])
        
    def strategy_volume_analysis(self, df):
        try:
            vr=df['volume_ratio'].iloc[-3:]; pc=df['close'].pct_change().iloc[-3:]
            d=0;r=[]
            if vr.iloc[-1]>1.2:
                if pc.iloc[-1]>0: d=1;r.append("vol_up_move")
                elif pc.iloc[-1]<0: d=-1;r.append("vol_down_move")
            return self._pack("volume_analysis","momentum",d,0.55 if d else 0.0,r)
        except: return self._pack("volume_analysis","momentum",0,0.0,["err"])
        
    def strategy_fibonacci(self, df):
        try:
            high_50=df['high'].rolling(50).max().iloc[-1]
            low_50=df['low'].rolling(50).min().iloc[-1]
            cp=df['close'].iloc[-1]
            f618=low_50+0.618*(high_50-low_50); f382=low_50+0.382*(high_50-low_50)
            d=0;r=[]
            if abs(cp-f618)/cp<0.005 and cp>df['close'].iloc[-3]: d=1;r.append("f618_long")
            elif abs(cp-f382)/cp<0.005 and cp<df['close'].iloc[-3]: d=-1;r.append("f382_short")
            return self._pack("fibonacci","meanrev",d,0.55 if d else 0.0,r)
        except: return self._pack("fibonacci","meanrev",0,0.0,["err"])
        
    def strategy_ichimoku(self, df):
        try:
            tenkan=(df['high'].rolling(9).max()+df['low'].rolling(9).min())/2
            kijun=(df['high'].rolling(26).max()+df['low'].rolling(26).min())/2
            if len(tenkan.dropna())<2 or len(kijun.dropna())<2:
                return self._pack("ichimoku","trend",0,0.0,[])
            d=0;r=[]
            if tenkan.iloc[-1]>kijun.iloc[-1] and tenkan.iloc[-2]<=kijun.iloc[-2] and df['close'].iloc[-1]>max(tenkan.iloc[-1],kijun.iloc[-1]):
                d=1;r.append("tenkan_kijun_bull")
            elif tenkan.iloc[-1]<kijun.iloc[-1] and tenkan.iloc[-2]>=kijun.iloc[-2] and df['close'].iloc[-1]<min(tenkan.iloc[-1],kijun.iloc[-1]):
                d=-1;r.append("tenkan_kijun_bear")
            return self._pack("ichimoku","trend",d,0.65 if d else 0.0,r)
        except: return self._pack("ichimoku","trend",0,0.0,["err"])
        
    def strategy_vwap(self, df):
        try:
            v=df['price_vs_vwap'].iloc[-3:]
            d=0;r=[]
            if v.iloc[-1]<-0.003 and v.iloc[-1]>v.iloc[-2]:
                d=1;r.append("vwap_revert_up")
            elif v.iloc[-1]>0.003 and v.iloc[-1]<v.iloc[-2]:
                d=-1;r.append("vwap_revert_down")
            return self._pack("vwap","meanrev",d,0.5 if d else 0.0,r)
        except: return self._pack("vwap","meanrev",0,0.0,["err"])
        
    def strategy_momentum(self, df):
        try:
            m=df['momentum'].iloc[-5:]; r=df['roc'].iloc[-5:]
            d=0; rr=[]
            if m.iloc[-1]>0 and r.iloc[-1]>0: d=1; rr.append("mom_up")
            elif m.iloc[-1]<0 and r.iloc[-1]<0: d=-1; rr.append("mom_down")
            return self._pack("momentum","momentum",d,0.55 if d else 0.0,rr)
        except: return self._pack("momentum","momentum",0,0.0,["err"])
        
    def strategy_price_action(self, df):
        try:
            d=0; rr=[]
            hammer=df['hammer'].iloc[-1]; doji=df['doji'].iloc[-1]
            if hammer and df['close'].iloc[-1]>df['open'].iloc[-1]:
                d=1; rr.append("hammer_bull")
            elif doji:
                d=-1; rr.append("doji_soft_sell")
            if (df['close'].iloc[-1]>df['open'].iloc[-1] and df['close'].iloc[-2]<df['open'].iloc[-2] and
                df['close'].iloc[-1]>df['open'].iloc[-2] and df['open'].iloc[-1]<df['close'].iloc[-2]):
                d=1; rr.append("bull_engulf")
            if (df['close'].iloc[-1]<df['open'].iloc[-1] and df['close'].iloc[-2]>df['open'].iloc[-2] and
                df['close'].iloc[-1]<df['open'].iloc[-2] and df['open'].iloc[-1]>df['close'].iloc[-2]):
                d=-1; rr.append("bear_engulf")
            if "bull_engulf" in rr: d=1
            if "bear_engulf" in rr: d=-1
            return self._pack("price_action","pattern",d,0.6 if d else 0.0,rr)
        except: return self._pack("price_action","pattern",0,0.0,["err"])
        
    def strategy_harmonic_patterns(self, df):
        try:
            pc=df['close'].pct_change().iloc[-8:]
            if len(pc)<8: return self._pack("harmonic_patterns","pattern",0,0.0,[])
            ab=pc.iloc[1:3].sum(); cd=pc.iloc[5:7].sum()
            d=0;r=[]
            if abs(abs(cd)-abs(ab))/(abs(ab)+1e-9)<0.3:
                if cd<0<ab: d=1;r.append("h_bull")
                elif cd>0>ab: d=-1;r.append("h_bear")
            return self._pack("harmonic_patterns","pattern",d,0.5 if d else 0.0,r)
        except: return self._pack("harmonic_patterns","pattern",0,0.0,["err"])
        
    def strategy_multi_timeframe(self, df, symbol):
        try:
            df_h=self.get_data(symbol, count=250, tf=self.higher_tf)
            if df_h is None or len(df_h)<80:
                return self._pack("multi_timeframe","trend",0,0.0,[])
            df_h=add_features(df_h)
            ema_cur=1 if df['ema21'].iloc[-1]>df['ema50'].iloc[-1] else -1
            ema_h=1 if df_h['ema21'].iloc[-1]>df_h['ema50'].iloc[-1] else -1
            rsi_c=df['rsi14'].iloc[-1]; rsi_h=df_h['rsi14'].iloc[-1]
            d=0;r=[]
            if ema_cur==1 and ema_h==1 and rsi_c<60 and rsi_h<65: d=1;r.append("mtf_bull")
            elif ema_cur==-1 and ema_h==-1 and rsi_c>40 and rsi_h>35: d=-1;r.append("mtf_bear")
            return self._pack("multi_timeframe","trend",d,0.7 if d else 0.0,r)
        except: return self._pack("multi_timeframe","trend",0,0.0,["err"])

    def collect_signals(self, df, symbol):
        """PATCH v2: Uključuje mikro indikatore"""
        funcs = [
            self.strategy_ema_cross,
            self.strategy_rsi_divergence,
            self.strategy_macd_signal,
            self.strategy_bollinger_bands,
            self.strategy_stochastic,
            self.strategy_adx_trend,
            self.strategy_support_resistance,
            self.strategy_volume_analysis,
            self.strategy_fibonacci,
            self.strategy_ichimoku,
            self.strategy_vwap,
            self.strategy_momentum,
            self.strategy_price_action,
            self.strategy_harmonic_patterns,
            lambda d: self.strategy_multi_timeframe(d, symbol),
            lambda d: self.strategy_micro_entry(d),
            self.strategy_micro_indicators  # PATCH v2
        ]
        out = []
        for fn in funcs:
            try:
                r = fn(df)
                if not isinstance(r, dict) or 'confidence' not in r:
                    r = self._pack(fn.__name__, "misc", 0, 0.0, ["invalid"])
                out.append(r)
            except Exception as e:
                out.append(self._pack(fn.__name__, "misc", 0, 0.0, [f"err:{e}"]))

        # debug ispis
        active = [s for s in out if s['raw_direction'] != 0]
        top_list = [f"{s['name']}({s['confidence']:.2f})" for s in
                    sorted(active, key=lambda x: x['confidence'], reverse=True)[:3]]
        logging.info(f"SIGNALS {symbol}: {len(active)}/{len(out)} | top3: {top_list}")
        
        buy_cnt  = sum(1 for s in out if s['raw_direction'] ==  1)
        sell_cnt = sum(1 for s in out if s['raw_direction'] == -1)
        logging.info(f"DIRECTION-COUNT {symbol}: BUY={buy_cnt} SELL={sell_cnt}")
        return out
        
    def detect_regime(self, df):
        regime={}
        try:
            ema200=df.get('ema200', df['close'].rolling(200).mean())
            if ema200.isna().sum()>10:
                regime['trend']=0.3
            else:
                slope=(ema200.iloc[-1]-ema200.iloc[-10])/(10+1e-9)
                adx=df['adx'].iloc[-1] if 'adx' in df.columns else 20
                slope_norm=abs(slope)/(df['close'].iloc[-1]*0.001)
                regime['trend']=float(np.clip(slope_norm*0.5 + (adx/40)*0.5,0,1))
            bbw=df['bb_width'].iloc[-30:].mean()
            if 'bb_width' in df.columns:
                ref=df['bb_width'].iloc[-300:].median() if len(df)>310 else bbw
                comp=bbw/(ref+1e-9)
            else: comp=1.0
            regime['range']=float(np.clip(1 - regime['trend']*0.6 + (1-min(comp,2))*0.4,0,1))
            atr_ratio=df['atr_ratio'].iloc[-1] if 'atr_ratio' in df.columns else 0.001
            median_atr=df['atr_ratio'].iloc[-300:].median() if len(df)>310 else atr_ratio
            vol_ratio=atr_ratio/(median_atr+1e-9)
            regime['high_vol']=float(np.clip(vol_ratio-1,0,1))
            regime['low_vol']=float(np.clip(1-vol_ratio,0,1))
            regime['squeeze']=float(np.clip(0.9-comp,0,1))
        except Exception as e:
            logging.warning(f"Regime fail: {e}")
            regime={"trend":0.3,"range":0.3,"high_vol":0.2,"low_vol":0.2,"squeeze":0.0}
        return regime

    def compute_adaptive_weights(self, symbol: str, regime: dict) -> dict:
        """
        PROFESIONALNO ADAPTIVNO PONDERISANJE
        - Kombinuje istorijske performanse sa režimskom prilagodbom
        - Eksponencijalni decay za starije rezultate
        - Minimalno zadržavanje baznih težina
        """
        stats = self.strategy_stats.get(symbol, {})
        regime_type = regime.get('label', 'neutral')
    
        adaptive = {}
        decay_factor = 0.95  # Eksponencijalni decay za starije podatke
        min_weight = 0.15    # Minimalna težina (ne dozvoli potpuno isključivanje)
        max_weight = 2.0     # Maksimalna težina (anti-overfitting)
    
        for strat_name, base_weight in self.strategy_weights.items():
            stat = stats.get(strat_name, {})
        
            # 1. Istorijske performanse (Wilson score + winrate)
            total = stat.get('total', 0)
            wins = stat.get('wins', 0)
        
            if total >= 10:  # Dovoljan sample
                # Wilson score (statistički robustniji od prostog winrate)
                z = 1.96  # 95% confidence
                phat = wins / total
                denominator = 1 + z**2 / total
                centre_adjusted = phat + z**2 / (2 * total)
                adjusted_sd = math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
            
                wilson_lower = (centre_adjusted - z * adjusted_sd) / denominator
                perf_mult = max(0.5, min(2.0, wilson_lower * 2.0))  # Scale 0.5-2.0
            
                # Primijeni eksponencijalni decay (noviji tradeovi važniji)
                decay = decay_factor ** (total / 20)  # Decay svaka 20 tradeova
                perf_mult = 1.0 + (perf_mult - 1.0) * decay
            else:
                perf_mult = 1.0  # Neutralno ako nema dovoljno podataka
        
            # 2. Režimska prilagodba (trend vs mean-reversion strategije)
            regime_mult = 1.0
        
            # Trend strategije u trending režimu
            trend_strats = ['macd_trend', 'adx_trend', 'ema_cross', 'supertrend', 
                       'parabolic_sar', 'ichimoku_trend']
            # Mean-reversion strategije u ranging režimu
            meanrev_strats = ['rsi_meanrev', 'bollinger_meanrev', 'stochastic_meanrev',
                         'cci_meanrev', 'williams_meanrev']
        
            if regime_type == 'trending':
                if strat_name in trend_strats:
                    regime_mult = 1.4
                elif strat_name in meanrev_strats:
                    regime_mult = 0.6
            elif regime_type == 'ranging':
                if strat_name in meanrev_strats:
                    regime_mult = 1.4
                elif strat_name in trend_strats:
                    regime_mult = 0.6
            elif regime_type == 'volatile':
                # U volatilnom režimu favorizuj momentum strategije
                if 'momentum' in strat_name or strat_name in ['rsi_divergence', 'volume_profile']:
                    regime_mult = 1.3
        
            # 3. Kombiniraj sve multiplikatore
            final_weight = base_weight * perf_mult * regime_mult
        
            # 4. Primijeni granice
            final_weight = max(min_weight, min(max_weight, final_weight))
        
            adaptive[strat_name] = final_weight
    
        # Normalizuj težine da suma bude konzistentna
        total_weight = sum(adaptive.values())
        if total_weight > 0:
            norm_factor = len(adaptive) / total_weight
            adaptive = {k: v * norm_factor for k, v in adaptive.items()}
    
        return adaptive


    def evaluate_auto_disables(self):
        if AUTO_DISABLE_STRATEGIES:
            for name, st in self.strategy_stats.items():
                if st['disabled_until'] and datetime.now(timezone.utc)<st['disabled_until']:
                    continue
                if st['trades']>=STRATEGY_DISABLE_MIN_TRADES:
                    wr=st['wins']/max(1,st['trades']); exp=st['avg_r']
                    wlb=wilson_lower_bound(st['wins'], st['trades'])
                    if (wlb<STRATEGY_DISABLE_WLB_HITRATE) and (exp<STRATEGY_DISABLE_EXPECTANCY):
                        st['disabled_until']=datetime.now(timezone.utc)+timedelta(minutes=STRATEGY_DISABLE_COOLDOWN_BARS)
                        st['paper']=True
                        logging.warning(f"⚠️ Strategy {name} disabled (wr={wr:.2f} wlb={wlb:.2f} exp={exp:.2f})")
        if AUTO_DISABLE_SYMBOLS:
            for sym, ss in self.symbol_stats.items():
                if ss['disabled_until'] and datetime.now(timezone.utc)<ss['disabled_until']:
                    continue
                trades=ss['trades']
                if trades>=SYMBOL_DISABLE_ROLLING:
                    wr=ss['wins']/max(1,trades)
                    exp=np.mean(ss['r_list']) if ss['r_list'] else 0
                    if exp<SYMBOL_DISABLE_EXPECTANCY and wr<SYMBOL_DISABLE_HITRATE:
                        ss['disabled_until']=datetime.now(timezone.utc)+timedelta(minutes=SYMBOL_DISABLE_COOLDOWN_MIN)
                        ss['paper']=True
                        logging.warning(f"⛔ Symbol {sym} disabled (wr={wr:.2f}, exp={exp:.2f})")
                today=datetime.utcnow().date()
                if ss['day_date']!=today:
                    ss['day_date']=today; ss['day_pnl']=0.0
                if ss['day_pnl']<=SYMBOL_DISABLE_DAILY_R_LIMIT:
                    ss['disabled_until']=datetime.now(timezone.utc)+timedelta(minutes=SYMBOL_DISABLE_COOLDOWN_MIN)
                    ss['paper']=True
                    logging.warning(f"🚫 Daily loss limit {sym} – disabling.")

    def aggregate_signals(self, signals: list, regime: dict, symbol: str) -> dict:
        logging.info(f"Strategy contributions: {[(s['name'], s['confidence'] * self.strategy_weights.get(s['name'], 0)) for s in signals if s['raw_direction'] != 0]}")
        """
        PROFESIONALNA AGREGACIJA SA QUALITY SCORING
        - Multi-level konflikt detekcija
        - Quality score baziran na konzistentnosti
        - Režimska validacija signala
        """
        if not signals:
            return {"confidence": 0, "direction": 0, "conflict": True, "quality": 0}
    
        # Adaptivne težine
        adaptive_weights = self.compute_adaptive_weights(symbol, regime)
    
        # Razdvoji signale po smjeru
        long_signals = []
        short_signals = []
        neutral_signals = []
    
        for sig in signals:
            raw_dir = sig['raw_direction']
            conf = sig['confidence']
            name = sig['name']
        
            weighted_conf = conf * adaptive_weights.get(name, 0.5)
        
            if raw_dir > 0:
                long_signals.append({'name': name, 'conf': weighted_conf, 'raw_conf': conf})
            elif raw_dir < 0:
                short_signals.append({'name': name, 'conf': weighted_conf, 'raw_conf': conf})
            else:
                neutral_signals.append({'name': name, 'conf': conf})
    
        # Izračunaj snagu svake strane
        long_strength = sum(s['conf'] for s in long_signals)
        short_strength = sum(s['conf'] for s in short_signals)
    
        # KONFLIKT DETEKCIJA (multi-level)
        conflict = False
        conflict_severity = 0.0
    
        if long_signals and short_signals:
            # Procenat signala u suprotnom smjeru
            total_active = len(long_signals) + len(short_signals)
            minority_pct = min(len(long_signals), len(short_signals)) / total_active
        
            # Snaga manjine u odnosu na većinu
            strength_ratio = min(long_strength, short_strength) / max(long_strength, short_strength, 0.001)
        
            # Konflikt ako ima značajnu manjinu ili snaga je balansirana
            if minority_pct > 0.25 or strength_ratio > 0.4:
                conflict = True
                conflict_severity = max(minority_pct, strength_ratio)
    
        # QUALITY SCORE (0-1): mjera konzistentnosti signala
        quality = 0.0
    
        if long_signals or short_signals:
            # 1. Uniformnost confidence (niža std dev = veći quality)
            all_confs = [s['raw_conf'] for s in long_signals + short_signals]
            if len(all_confs) > 1:
                conf_std = np.std(all_confs)
                conf_mean = np.mean(all_confs)
                uniformity = max(0, 1 - (conf_std / (conf_mean + 0.01)))
            else:
                uniformity = 1.0
        
            # 2. Konzistentnost smjera (% signala u dominantnom smjeru)
            if long_signals or short_signals:
                consistency = max(len(long_signals), len(short_signals)) / (len(long_signals) + len(short_signals))
            else:
                consistency = 0.0
        
            # 3. Broj aktivnih strategija (više strategija = veći quality)
            num_active = len(long_signals) + len(short_signals)
            coverage = min(1.0, num_active / 10.0)  # Optimalno 10+ strategija
        
            # Kombiniraj komponente
            quality = (uniformity * 0.35 + consistency * 0.45 + coverage * 0.20)
        
            # Penalizuj za konflikt
            if conflict:
                quality *= (1 - conflict_severity * 0.5)
    
        # FINALNI PRAVAC I CONFIDENCE
        if long_strength > short_strength:
            direction = 1
            base_conf = long_strength / (long_strength + short_strength + 0.001)
        elif short_strength > long_strength:
            direction = -1
            base_conf = short_strength / (long_strength + short_strength + 0.001)
        else:
            direction = 0
            base_conf = 0.0
    
        # Prilagodi confidence sa quality score
        final_confidence = base_conf * (0.7 + 0.3 * quality)  # Quality bonus do 30%
    
        # Režimska validacija (dodatna penalizacija)
        regime_type = regime.get('label', 'neutral')
        regime_penalty = 1.0
    
        if regime_type == 'low_volatility':
            regime_penalty = 0.85  # Opreznije u niskoj volatilnosti
        elif regime_type == 'very_volatile':
            regime_penalty = 0.80  # Opreznije u ekstremnoj volatilnosti
    
        final_confidence *= regime_penalty
    
        return {
            "confidence": final_confidence,
            "direction": direction,
            "conflict": conflict,
            "conflict_severity": conflict_severity,
            "quality": quality,
            "long_strength": long_strength,
            "short_strength": short_strength,
            "num_long": len(long_signals),
            "num_short": len(short_signals),
            "regime": regime_type
        }


    def combine_with_ml(self, agg: dict, ml_prob: float) -> dict:
        """
        Kombinuje ML predikciju sa agregiranim signalima.
        Vraća dictionary sa direction i confidence.
    
        Args:
            agg: Agregirani signali (dict sa 'confidence', 'direction', 'conflict')
            ml_prob: ML probabilnost (0-1)
    
        Returns:
            dict: {'direction': int, 'confidence': float, ...}
        """
        # ML direction: -1 do +1
        ml_dir = 2 * ml_prob - 1
    
        # Uzmi direction i confidence iz agregacije
        agg_direction = agg.get('direction', 0)
        agg_confidence = agg.get('confidence', 0.5)
    
        # Konflikt penalizacija
        penalty = 0.7 if agg.get('conflict', False) else 1.0
    
        # Kombiniraj ML i agregaciju
        # Ako ML i agregacija se slažu u smjeru -> boost
        # Ako se ne slažu -> penalizuj
        if (ml_dir > 0.1 and agg_direction == 1) or (ml_dir < -0.1 and agg_direction == -1):
            # Agreement - boost confidence
            final_confidence = agg_confidence * 1.15 * penalty
            direction = agg_direction
        elif (ml_dir < -0.1 and agg_direction == 1) or (ml_dir > 0.1 and agg_direction == -1):
            # Disagreement - heavy penalty
            final_confidence = agg_confidence * 0.6 * penalty
            direction = agg_direction  # Prefer signals over ML
        else:
            # ML neutral - rely on signals
            final_confidence = agg_confidence * penalty
            direction = agg_direction
    
        # Cap confidence
        final_confidence = min(0.95, max(0, final_confidence))
    
        return {
            'direction': direction,
            'confidence': final_confidence,
            'ml_dir': ml_dir,
            'ml_prob': ml_prob,
            'agg_confidence': agg_confidence,
            'penalty_applied': penalty < 1.0
        }

    def decide_trade(self, combined: dict, agg: dict, ml_prob: float, 
                     symbol: str, df: pd.DataFrame, signals: list) -> dict:
        """
        Donosi odluku o trgovanju na osnovu kombinovanih signala.

        Args:
            combined: Dict iz combine_with_ml() sa 'direction' i 'confidence'
            agg: Agregirani signali sa 'conflict', 'quality', itd.
            ml_prob: ML verovatnoća (0-1)
            symbol: Trgovački simbol
            df: DataFrame sa podacima
            signals: Lista svih signala

        Returns:
            Dict sa trgovačkim parametrima ili None ako nema signala
        """
        # ═══════════════════════════════════════════════════════════════
        # VALIDACIJA UNOSA
        # ═══════════════════════════════════════════════════════════════
        if not isinstance(combined, dict):
            logging.error(f"{symbol}: combine_with_ml returned non-dict: {type(combined)}")
            return None

        if 'direction' not in combined or 'confidence' not in combined:
            logging.error(f"{symbol}: Missing keys in combined: {combined.keys()}")
            return None

        # ═══════════════════════════════════════════════════════════════
        # IZVLAČENJE OSNOVNIH PODATAKA
        # ═══════════════════════════════════════════════════════════════
        direction = combined.get('direction', 0)
        confidence = combined.get('confidence', 0)

        # Brzi izlaz ako nema signala
        if direction == 0 or confidence < 0.45:
            return None

        # ═══════════════════════════════════════════════════════════════
        # QUALITY SKOR IZ AGREGACIJE
        # ═══════════════════════════════════════════════════════════════
        quality = agg.get('quality', 0.5)
        conflict = agg.get('conflict', False)
        conflict_severity = agg.get('conflict_severity', 0.0)

        # ═══════════════════════════════════════════════════════════════
        # DODELA TIER-A (na osnovu kvaliteta)
        # ═══════════════════════════════════════════════════════════════
        # Tier 1: Vrhunski setup (visok conf + visok quality + bez konflikta)
        # Tier 2: Dobar setup (dobar conf + OK quality + mali konflikt)
        # Tier 3: Prosečan setup (minimalni zahtevi)

        tier = 3  # Podrazumevani
        min_conf_threshold = 0.45

        if confidence >= 0.75 and quality >= 0.70 and not conflict:
            tier = 1
            min_conf_threshold = 0.65
        elif confidence >= 0.65 and quality >= 0.55 and conflict_severity < 0.3:
            tier = 2
            min_conf_threshold = 0.55
        elif confidence >= 0.50 and quality >= 0.40:
            tier = 3
            min_conf_threshold = 0.45
        else:
            return None  # Nedovoljan kvalitet

        # ═══════════════════════════════════════════════════════════════
        # PROVERA SLAganja SA ML
        # ═══════════════════════════════════════════════════════════════
        ml_agreement = False

        if ml_prob >= 0.52 and direction == 1:
            ml_agreement = True
        elif ml_prob <= 0.48 and direction == -1:
            ml_agreement = True

        # Za tier 1, zahteva se saglasnost ML-a
        if tier == 1 and not ml_agreement:
            tier = 2  # Snižavanje tier-a
            logging.debug(f"{symbol}: Tier 1 snižen na 2 (nema saglasnosti ML-a)")

        # ═══════════════════════════════════════════════════════════════
        # PROVERA VOLATILNOSTI
        # ═══════════════════════════════════════════════════════════════
        if 'atr' in df.columns:
            try:
                recent_volatility = df['close'].pct_change().tail(20).std()
                avg_volatility = df['close'].pct_change().tail(100).std()
                vol_ratio = recent_volatility / (avg_volatility + 0.0001)
        
                # Previsoka volatilnost = skip tier 3
                if vol_ratio > 2.5 and tier == 3:
                    logging.debug(f"{symbol}: Skipped - high volatility (ratio: {vol_ratio:.2f})")
                    return None
            except Exception as e:
                logging.warning(f"{symbol}: Volatility check failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # SPREAD CHECK (za FX simbole)
        # ═══════════════════════════════════════════════════════════════
        if mt5:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    spread_points = symbol_info.spread
                    point = self.get_point(symbol)
                    spread_pips = spread_points * point / point  # Pips
            
                    # Ako je spread abnormalno visok, skip ili downgrade
                    if spread_pips > 5 and tier >= 2:
                        tier = 3
                        logging.debug(f"{symbol}: Tier downgraded due to spread ({spread_pips:.1f} pips)")
                    elif spread_pips > 8:
                        logging.debug(f"{symbol}: Skipped - excessive spread ({spread_pips:.1f} pips)")
                        return None
            except Exception as e:
                logging.warning(f"{symbol}: Spread check failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # FINAL VALIDATION
        # ═══════════════════════════════════════════════════════════════
        if confidence < min_conf_threshold:
           return None

        # ═══════════════════════════════════════════════════════════════
        # RETURN DECISION
        # ═══════════════════════════════════════════════════════════════
        decision = {
           "direction": direction,
           "confidence": confidence,
           "tier": tier,
           "quality": quality,
           "ml_prob": ml_prob,
           "ml_agreement": ml_agreement,
           "conflict": conflict,
           "conflict_severity": conflict_severity
        }

        # Log odluke
        logging.info(f"✅ {symbol} DECISION: dir={direction}, conf={confidence:.2f}, "
        f"tier={tier}, quality={quality:.2f}, ml_agree={ml_agreement}")

        return decision

    def is_spread_ok(self, symbol):
        try:
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if not info or not tick:
                return True
            spread_points = (tick.ask - tick.bid) / info.point
            stops_level = getattr(info, 'trade_stops_level', 10)
            threshold = max(3 * stops_level, 8)
            return spread_points <= threshold
        except:
            return True

    def is_safe_to_trade(self, symbol, df=None):
        if not self.is_spread_ok(symbol):
            return True
        try:
            if df is not None and len(df) > 30:
                atr = talib.ATR(df['high'], df['low'], df['close']).iloc[-1]
                atr_ratio = atr / df['close'].iloc[-1]
                if atr_ratio > 0.65:
                    return False
        except:
            pass
        ss = self.symbol_stats.get(symbol)
        if ss and ss.get('disabled_until') and datetime.now(timezone.utc) < ss['disabled_until']:
            return False
        return True

    def get_pip_value_and_digits(self, symbol):
        info=mt5.symbol_info(symbol)
        if not info: return 10.0, 5, 0.00001
        digits=info.digits
        if "JPY" in symbol: pip_val=9.0
        elif symbol.startswith("XAU"): pip_val=1.0
        elif symbol.startswith("BTC"): pip_val=10.0
        else: pip_val=10.0
        point=info.point
        pip_size=point*(10 if digits>=5 else 1)
        return pip_val, digits, pip_size

    def compute_scalp_params(self, symbol, direction, df):
        pip_val, digits, pip_size = self.get_pip_value_and_digits(symbol)

        # >>> PER-SYMBOL PIPS <<<
        p           = self._symbol_params(symbol)
        target_pips = p["tp_pips"]
        sl_pips     = p["sl_pips"]
        # >>> END PER-SYMBOL <<<

        target_usd = np.random.uniform(*SCALP_TARGET_USD_RANGE)
        lot = target_usd / (pip_val * target_pips)

        info = mt5.symbol_info(symbol)
        if info:
            step = info.volume_step if info.volume_step > 0 else 0.01
            lot = max(info.volume_min, min(info.volume_max, round(lot / step) * step))

        price = df['close'].iloc[-1]
        min_dist = max(info.trade_stops_level, 10) * pip_size
        
        if symbol.startswith("BTC"):
            min_dist = max(min_dist, 1500 * pip_size)
        
        # >>> BTC extra distance <<<
        if symbol.startswith("BTC"):
            min_dist = max(min_dist, 1200 * pip_size)   # 120 USD min

        if direction == "buy":
            sl = price - max(sl_pips * pip_size, min_dist)
            tp = price + target_pips * pip_size
        else:
            sl = price + max(sl_pips * pip_size, min_dist)
            tp = price - target_pips * pip_size

        return float(lot), round(sl, digits), round(tp, digits), target_pips, sl_pips

    def swing_calc_stops(self, df: pd.DataFrame, symbol: str, direction: int) -> tuple:
        """
        PROFESIONALNI STOP CALCULATION
        - ATR-based sa volatility adjustment
        - Psihološki nivoi (round numbers)
        - Support/Resistance integracija
        - Dinamički RR ratio baziran na market conditions
        """
        cfg = self.symbol_configs.get(symbol, self.symbol_configs["EURUSD"])
        point = self.get_point(symbol)
        spread = mt5.symbol_info(symbol).spread * point if mt5 else 0

        # 1. ATR CALCULATION (Multi-period za robusnost)
        atr_14 = df['atr'].iloc[-1] if 'atr' in df.columns else 0.0001
        atr_7 = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 7)[-1]
        atr_21 = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 21)[-1]

        # Weighted ATR (više težine na kraći period - brža reakcija)
        atr = (atr_7 * 0.5 + atr_14 * 0.3 + atr_21 * 0.2)

        # 2. VOLATILITY ADJUSTMENT
        recent_volatility = df['close'].pct_change().tail(20).std()
        avg_volatility = df['close'].pct_change().std()
        vol_ratio = recent_volatility / (avg_volatility + 0.0001)

        # Prilagodi ATR multiplier based na volatility
        if vol_ratio > 1.5:  # Visoka volatilnost
            atr_mult_sl = 2.0  # Širi SL
            atr_mult_tp = 2.5  # Manji TP mult (konzervativnije)
            rr_target = 1.5    # Niži RR target
        elif vol_ratio < 0.7:  # Niska volatilnost
            atr_mult_sl = 1.2  # Tijesni SL
            atr_mult_tp = 3.5  # Veći TP mult (agresivnije)
            rr_target = 2.5    # Viši RR target
        else:  # Normalna volatilnost
            atr_mult_sl = 1.5
            atr_mult_tp = 3.0
            rr_target = 2.0

        # 3. SUPPORT/RESISTANCE ADJUSTMENT
        close_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()

        # Za LONG poziciju
        if direction == 1:
            # SL ispod recent swing low, ali ne dalje od ATR limita
            swing_sl = recent_low - spread
            atr_sl = close_price - (atr * atr_mult_sl)
            sl_price = max(swing_sl, atr_sl)  # Uzmi bliži (tiješni) SL
    
            # TP iznad resistance + ATR
            resistance = recent_high
            atr_tp = close_price + (atr * atr_mult_tp)
            tp_price = max(resistance + atr * 0.5, atr_tp)
    
        # Za SHORT poziciju
        else:
            # SL iznad recent swing high, ali ne dalje od ATR limita
            swing_sl = recent_high + spread
            atr_sl = close_price + (atr * atr_mult_sl)
            sl_price = min(swing_sl, atr_sl)  # Uzmi bliži (tijesni) SL
    
            # TP ispod support - ATR
            support = recent_low
            atr_tp = close_price - (atr * atr_mult_tp)
            tp_price = min(support - atr * 0.5, atr_tp)

        # 4. MINIMUM RR RATIO ENFORCEMENT
        actual_risk = abs(close_price - sl_price)
        actual_reward = abs(tp_price - close_price)
        actual_rr = actual_reward / (actual_risk + 0.0001)
    
        # Ako je RR ratio prenizak, proširi TP
        if actual_rr < rr_target:
            if direction == 1:
                tp_price = close_price + (actual_risk * rr_target)
            else:
                tp_price = close_price - (actual_risk * rr_target)
    
        # 5. FINAL VALIDATION
        min_sl = atr * 0.8  # Minimum SL distanca
        max_sl = atr * 3.0  # Maximum SL distanca (anti-runaway)
    
        sl_distance = abs(close_price - sl_price)
    
        if sl_distance < min_sl:
            if direction == 1:
                sl_price = close_price - min_sl
            else:
                sl_price = close_price + min_sl
        elif sl_distance > max_sl:
            if direction == 1:
                sl_price = close_price - max_sl
            else:
                sl_price = close_price + max_sl
    
        # Rekalibruj TP nakon SL korekcije
        actual_risk = abs(close_price - sl_price)
        if direction == 1:
            tp_price = close_price + (actual_risk * rr_target)
        else:
            tp_price = close_price - (actual_risk * rr_target)
    
        return sl_price, tp_price
    
    def round_to_psychological(price, direction_adjust):
        """Zaokruži na psihološke nivoe (50 pips)"""
        pips = price / pip_size
        rounded_pips = round(pips / 50) * 50
        return rounded_pips * pip_size
    
        # Za SL, zaokruži u "sigurniju" stranu
        # Za TP, zaokruži u "ambiciozniju" stranu
        # sl_price = round_to_psychological(sl_price, direction * -1)
        # tp_price = round_to_psychological(tp_price, direction)
    
        # 5. MINIMUM RR RATIO ENFORCEMENT
        actual_risk = abs(close_price - sl_price)
        actual_reward = abs(tp_price - close_price)
        actual_rr = actual_reward / (actual_risk + 0.0001)
    
        # Ako je RR ratio prenizak, proširi TP
        if actual_rr < rr_target:
            if direction == 1:
                tp_price = close_price + (actual_risk * rr_target)
            else:
                tp_price = close_price - (actual_risk * rr_target)
    
        # 6. FINAL VALIDATION
        min_sl = atr * 0.8  # Minimum SL distanca
        max_sl = atr * 3.0  # Maximum SL distanca (anti-runaway)
    
        sl_distance = abs(close_price - sl_price)
    
        if sl_distance < min_sl:
            if direction == 1:
                sl_price = close_price - min_sl
            else:
                sl_price = close_price + min_sl
        elif sl_distance > max_sl:
            if direction == 1:
                sl_price = close_price - max_sl
            else:
                sl_price = close_price + max_sl
    
        # Rekalibruj TP nakon SL korekcije
        actual_risk = abs(close_price - sl_price)
        if direction == 1:
            tp_price = close_price + (actual_risk * rr_target)
        else:
            tp_price = close_price - (actual_risk * rr_target)
    
        return sl_price, tp_price

    def _safe_sl_tp(self, symbol, direction, price, sl, tp):
        """Osigurava da SL/TP nisu previše blizu cijene (respektuje broker limits)"""
        try:
            info = mt5.symbol_info(symbol)
            if not info:
                return sl, tp
        
            min_distance = info.trade_stops_level * info.point
        
            # Ako broker nema minimum distance, postavi razuman default
            if min_distance == 0:
                min_distance = 10 * info.point  # 10 points minimum
        
            if direction == "buy" or direction == 1:
                # SL ispod price
                if sl > 0 and (price - sl) < min_distance:
                    sl = price - min_distance
                # TP iznad price
                if tp > 0 and (tp - price) < min_distance:
                    tp = price + min_distance
            else:  # sell
                # SL iznad price
                if sl > 0 and (sl - price) < min_distance:
                    sl = price + min_distance
                # TP ispod price
                if tp > 0 and (price - tp) < min_distance:
                    tp = price - min_distance
        
            return sl, tp
        
        except Exception as e:
            logging.error(f"_safe_sl_tp error {symbol}: {e}")
            return sl, tp


    def place_order(self, symbol, direction, sl, tp, tier, lot=None, context="", scalp=False, micro=False):
        """
        Otvara trading poziciju sa validacijom i error handlingom.
    
        Args:
            symbol: Trading simbol (EURUSD, GBPUSD, itd.)
            direction: 1/"buy" ili -1/"sell"
            sl: Stop loss cijena
            tp: Take profit cijena
            tier: Tier nivo (1-3)
            lot: Lot size (None = auto calculate)
            context: Kontekst (strategije koje su dale signal)
            scalp: Da li je scalp trade
            micro: Da li je micro trade (PATCH v2)
    
        Returns:
            MT5 order result ili None ako neuspješno
        """
        try:
            # Normalizuj direction
            if direction == 1 or direction == "buy":
                direction = "buy"
            elif direction == -1 or direction == "sell":
                direction = "sell"
            else:
                logging.error(f"Invalid direction: {direction}")
                return None
        
            # Prefix za komentar
            if micro:
                lot = self.micro_lot
                prefix = "MICRO-"
            elif scalp:
                prefix = "SCALP-"
            else:
                prefix = "SE2-"
        
            # Terminal validation
            terminal_info = mt5.terminal_info()
            if not terminal_info or not terminal_info.trade_allowed:
                logging.warning(f"Trading not allowed in terminal for {symbol}")
                return None
        
            # Account validation
            account_info = mt5.account_info()
            if account_info is None:
                logging.warning(f"Account not authorized for {symbol}")
                return None
        
            # Symbol selection
            if not mt5.symbol_select(symbol, True):
                logging.warning(f"Failed to select symbol {symbol}")
                return None
        
            # Get tick and symbol info
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if not tick or not info:
                logging.warning(f"Failed to get tick/info for {symbol}")
                return None
        
            # Check if trading enabled for symbol
            if hasattr(info, 'trade_mode') and info.trade_mode == 0:
                logging.warning(f"Trading disabled for symbol {symbol}")
                return None
        
            # Calculate lot size if not provided
            if lot is None:
                if hasattr(self, 'calc_lot_from_tier'):
                    base = self.calc_lot_from_tier(tier)
                    lot = self.calc_dynamic_lot(symbol, base)
                else:
                    lot = round(self.base_lot_swing, 2)
        
            # Normalize lot to symbol's volume step
            if hasattr(info, 'volume_step') and info.volume_step > 0:
                lot = round(lot / info.volume_step) * info.volume_step
        
            # Ensure lot is within min/max limits
            if hasattr(info, 'volume_min') and lot < info.volume_min:
                lot = info.volume_min
            if hasattr(info, 'volume_max') and lot > info.volume_max:
                lot = info.volume_max
        
            # Get execution price
            price = tick.ask if direction == "buy" else tick.bid
        
            # Validate and adjust SL/TP
            sl, tp = self._safe_sl_tp(symbol, direction, price, sl, tp)
        
            # Round SL/TP to proper digits
            if hasattr(info, 'digits'):
                sl = round(sl, info.digits)
                tp = round(tp, info.digits)
        
            # Build comment
            raw_comment = f"{prefix}{direction[:1].upper()}T{tier}"
            if context:
                clean_context = context[:10].replace(":", "_").replace(",", "_")
                raw_comment += f"_{clean_context}"
            comment = self._sanitize_comment(raw_comment)[:31]  # MT5 max 31 chars
        
            # Determine fill type - RETURN mode (najkompatibilniji za sve brokere)
            
            type_filling = mt5.ORDER_FILLING_FOK
                    
            # Build request
            
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 30,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }
        
            # Send order
            res = mt5.order_send(req)
        
            # Retry without comment if failed
            if res is None:
                last_error = mt5.last_error()
                logging.warning(f"Order failed with None response for {symbol}, last_error: {last_error}, retrying without comment")
                req_retry = req.copy()
                req_retry.pop("comment", None)
                res = mt5.order_send(req_retry)
            
                if res is None:
                    last_error = mt5.last_error()
                    logging.error(f"Order retry also failed for {symbol}, last_error: {last_error}")
                    return None
        
            # Check result
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                last_error = mt5.last_error()
                logging.error(f"Order fail {symbol} {direction}: retcode={res.retcode}, comment={res.comment}, last_error={last_error}")
                return None
        
            # Success!
            mode_str = 'MICRO' if micro else ('SCALP' if scalp else 'SWING')
            logging.info(f"🎯 {symbol} {direction.upper()} lot={lot} @ {price:.5f} SL:{sl:.5f} TP:{tp:.5f} tier={tier} ({mode_str})")
        
            # Telegram notification (optional)
            try:
                self.telegram_send(f"🎯 {symbol} {direction.upper()} lot={lot} @ {price:.5f} SL={sl:.5f} TP={tp:.5f}")
            except:
                pass
        
            return res
        
        except Exception as e:
            logging.error(f"place_order error {symbol}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None

    def calc_lot_from_tier(self, tier):
        mult={1:0.5,2:1.0,3:1.3}.get(tier,1.0)
        return round(self.base_lot_swing*mult,2)
        # čitamo base-lot iz parametra
        p = self._symbol_params(self._current_symbol)   # definiramo u 4. koraku
        return p["lot_size"]


    def _sanitize_comment(self, text: str) -> str:
        """Sanitize comment text for MT5 compatibility.
        
        Args:
            text: Comment text to sanitize
            
        Returns:
            Sanitized comment string (max 31 chars, safe characters only)
        """
        try:
            # Convert to str safely
            text_str = str(text) if text is not None else ""
            
            # Replace any character not in [A-Za-z0-9 space underscore hyphen] with underscore
            sanitized = re.sub(r'[^A-Za-z0-9 _-]', '_', text_str)
            
            # Truncate to max 31 characters (ASCII length)
            return sanitized[:31]
        except Exception as e:
            logging.warning(f"Comment sanitization error: {e}")
            return "sanitized_comment"

    def modify_position(self, position, new_sl, new_tp=None):
        try:
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return False

            # === NOVO: ispravi SL da bude dovoljno daleko ===
            info = mt5.symbol_info(position.symbol)
            if info:
                point        = info.point
                stops_level  = max(info.trade_stops_level, 10)
                min_dist     = stops_level * point
                current_price= tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid
                digits       = info.digits

                if position.type == mt5.ORDER_TYPE_BUY:
                    if new_sl >= current_price - min_dist:
                        new_sl = current_price - min_dist - point
                else:  # SELL
                    if new_sl <= current_price + min_dist:
                        new_sl = current_price + min_dist + point
                new_sl = round(new_sl, digits)

            # Ako se SL nije promenio – ne šalji
            if position.sl is not None and abs(new_sl - position.sl) < info.point:
                return True

            comment = self._sanitize_comment(f"Trail-{position.comment}")

            # === Kraj nove provjere ===

            comment = self._sanitize_comment(f"Trail-{position.comment}")

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "magic": position.magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC
            }
            if new_tp is not None:
                req["tp"] = new_tp

            res = mt5.order_send(req)
            if res is None:
                last_error = mt5.last_error()
                logging.error(f"Modify failed with None response for {position.symbol}, last_error: {last_error}")
                return False

            if res.retcode != mt5.TRADE_RETCODE_DONE:
                last_error = mt5.last_error()
                logging.error(f"Modify fail {position.symbol}: retcode={res.retcode}, comment={res.comment}, last_error={last_error}")
                return False

            logging.info(f"🔄 SL updated {position.symbol}: {new_sl}")
            return True

        except Exception as e:
            logging.error(f"modify_position error: {e}")
            return False

    def close_position_market(self, position, comment="ManualClose"):
        try:
            tick=mt5.symbol_info_tick(position.symbol)
            if not tick: return False
            opposite=mt5.ORDER_TYPE_SELL if position.type==mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price=tick.bid if opposite==mt5.ORDER_TYPE_SELL else tick.ask
            req={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": opposite,
                "position": position.ticket,
                "price": price,
                "deviation": 30,
                "magic": position.magic,
                "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC
            }
            res=mt5.order_send(req)
            return res and res.retcode==mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logging.error(f"close_position_market error: {e}")
            return False

    def close_partial_position(self, position, ratio=0.4):
        if self.partial_done.get(position.ticket, False): return False
        try:
            info=mt5.symbol_info(position.symbol)
            if not info: return False
            step=info.volume_step if info.volume_step>0 else 0.01
            close_vol=round(position.volume*ratio/step)*step
            close_vol=max(info.volume_min, min(close_vol, position.volume-info.volume_min+1e-9))
            if close_vol<=0: return False
            tick=mt5.symbol_info_tick(position.symbol)
            if not tick: return False
            opposite=mt5.ORDER_TYPE_SELL if position.type==mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price=tick.bid if opposite==mt5.ORDER_TYPE_SELL else tick.ask
            req={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": float(close_vol),
                "type": opposite,
                "position": position.ticket,
                "price": price,
                "deviation": 30,
                "magic": self.magic_number,
                "comment": "PartialTP",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN
            }
            res=mt5.order_send(req)
            if res.retcode==mt5.TRADE_RETCODE_DONE:
                self.partial_done[position.ticket]=True
                logging.info(f"🏁 Partial {position.symbol}: {close_vol}")
                return True
            else:
                logging.warning(f"Partial fail {position.symbol}: {res.comment}")
                return False
        except Exception as e:
            logging.error(f"close_partial_position error: {e}")
            return False

    def manage_swing_positions(self):
        try:
            positions=mt5.positions_get()
            if not positions: return
            for p in positions:
                if p.magic!=self.magic_number: continue
                if p.comment.startswith("SCALP-"): continue
                now=datetime.now(timezone.utc)
                if p.ticket in self.last_trail_update and (now-self.last_trail_update[p.ticket]).total_seconds()<self.trailing_cooldown_sec:
                    continue
                info=mt5.symbol_info(p.symbol)
                if not info: continue
                tick=mt5.symbol_info_tick(p.symbol)
                if not tick: continue
                digits=info.digits; point=info.point
                cp=tick.ask if p.type==mt5.ORDER_TYPE_BUY else tick.bid
                df_sym=self.get_data(p.symbol, count=140, tf=self.swing_tf)
                if df_sym is None or len(df_sym)<40: continue
                df_sym=add_features(df_sym)
                atr=df_sym['atr'].iloc[-1] if 'atr' in df_sym.columns else 10*point
                atr_pts=atr/point
                profit_pts=(cp-p.price_open)/point if p.type==mt5.ORDER_TYPE_BUY else (p.price_open - cp)/point

                if self.breakeven_enabled and not self.be_done.get(p.ticket,False) and profit_pts>=(self.breakeven_atr_mult*atr_pts):
                    be_price=p.price_open + self.be_offset_pts*point if p.type==mt5.ORDER_TYPE_BUY else p.price_open - self.be_offset_pts*point
                    be_price=round(be_price,digits)
                    if p.sl is None or (p.type==mt5.ORDER_TYPE_BUY and be_price>p.sl) or (p.type==mt5.ORDER_TYPE_SELL and be_price<p.sl):
                        if self.modify_position(p, be_price):
                            self.be_done[p.ticket]=True
                            self.last_trail_update[p.ticket]=now
                            continue

                if self.partial_tp_enabled and not self.partial_done.get(p.ticket,False) and profit_pts>=(self.partial_atr_mult*atr_pts):
                    if self.close_partial_position(p,self.partial_ratio):
                        self.last_trail_update[p.ticket]=now
                        continue

                if self.trailing_stop_enabled and profit_pts>=(self.trailing_stop_activation*atr_pts):
                    target_sl=(cp - self.trailing_stop_distance*atr) if p.type==mt5.ORDER_TYPE_BUY else (cp + self.trailing_stop_distance*atr)
                    target_sl=round(target_sl,digits)
                    curr_sl=p.sl
                    update=False
                    if curr_sl is not None:
                        if p.type==mt5.ORDER_TYPE_BUY and target_sl - curr_sl < atr*TRAIL_MIN_MOVE_MULT_ATR:
                            update=False
                        elif p.type==mt5.ORDER_TYPE_SELL and curr_sl - target_sl < atr*TRAIL_MIN_MOVE_MULT_ATR:
                            update=False
                        else:
                            if p.type==mt5.ORDER_TYPE_BUY and target_sl>curr_sl: update=True
                            if p.type==mt5.ORDER_TYPE_SELL and target_sl<curr_sl: update=True
                    else:
                        update=True
                    if update and self.modify_position(p,target_sl):
                        self.last_trail_update[p.ticket]=now
        except Exception as e:
            logging.error(f"manage_swing_positions error: {e}")

    def manage_scalps(self):
        if not SCALP_MODE or not self.scalp_trades: return
        to_remove=[]
        for oid, st in list(self.scalp_trades.items()):
            positions=mt5.positions_get(ticket=oid)
            if not positions:
                to_remove.append(oid); continue
            p=positions[0]
            info=mt5.symbol_info(st['symbol'])
            if not info: continue
            tick=mt5.symbol_info_tick(st['symbol'])
            if not tick: continue
            pip_val,dig,pip_size=self.get_pip_value_and_digits(st['symbol'])
            cp=tick.bid if p.type==mt5.ORDER_TYPE_SELL else tick.ask
            diff=(cp-st['entry_price']) if p.type==mt5.ORDER_TYPE_BUY else (st['entry_price']-cp)
            diff_pips=diff/pip_size
            st['max_fav']=max(st['max_fav'], diff_pips)
            st['max_adv']=max(st['max_adv'], -diff_pips if diff_pips<0 else 0)
            bars_elapsed=int((datetime.now(timezone.utc)-st['opened']).total_seconds()/60)

            # === NOVI SETTING – da "diše" ===
            if bars_elapsed <= 3 and st['max_adv'] > 5:  # raniji abort tek na -5p
                self.close_position_market(p, "EarlyAbort")
                logging.info(f"⛔ Early abort {st['symbol']} adv={st['max_adv']:.1f}p")
                to_remove.append(oid); continue

            if bars_elapsed >= 20:  # duži rok
                self.close_position_market(p, "TimeStop")
                logging.info(f"⌛ Time-stop {st['symbol']} fav={st['max_fav']:.1f}p")
                to_remove.append(oid); continue

            # BE tek na +8 pips
            if diff_pips >= 8 and (p.sl is None or abs(p.sl - st['entry_price']) > 1e-5):
                be_price = st['entry_price']
                if self.modify_position(p, round(be_price, dig)):
                    logging.info(f"⏳ Scalp BE set {st['symbol']} (after +8p)")

        for r in to_remove:
            self.scalp_trades.pop(r, None)
    def update_performance_stats(self):
        now=datetime.now(timezone.utc)
        if (now - self.last_perf_update).total_seconds()<PERF_UPDATE_INTERVAL:
            return
        from_time=self.last_closed_deal_time; to_time=now
        deals=mt5.history_deals_get(from_time, to_time)
        if deals:
            for d in deals:
                if d.magic!=self.magic_number: continue
                profit=d.profit+d.swap+d.commission
                symbol=d.symbol; comment=d.comment
                base_strats=[]
                if ":" in comment:
                    raw=comment.split(":",1)[1]
                    parts=[p.strip() for p in raw.split(",") if p.strip() in self.strategy_stats]
                    base_strats=parts[:3]
                if not base_strats:
                    base_strats=["ema_cross","multi_timeframe","momentum"]
                r_mult=1 if profit>=0 else -1
                for sn in base_strats:
                    st=self.strategy_stats.get(sn)
                    if not st: continue
                    st['trades']+=1
                    if profit>=0: st['wins']+=1
                    else: st['losses']+=1
                    st['pnl']+=profit
                    st['r_list'].append(r_mult)
                    st['avg_r']=np.mean(st['r_list'])
                    if len(st['r_list'])>5:
                        st['sharpe']=st['avg_r']/(np.std(st['r_list'])+1e-9)
                sym_stat=self.symbol_stats.get(symbol)
                if sym_stat:
                    sym_stat['trades']+=1
                    if profit>=0: sym_stat['wins']+=1
                    else: sym_stat['losses']+=1
                    sym_stat['pnl']+=profit
                    sym_stat['r_list'].append(r_mult)
                    today=datetime.utcnow().date()
                    if sym_stat['day_date']!=today:
                        sym_stat['day_date']=today; sym_stat['day_pnl']=0.0
                    sym_stat['day_pnl']+=r_mult
        self.last_closed_deal_time=now - timedelta(minutes=POLL_HISTORY_MIN)
        self.last_perf_update=now
        self.evaluate_auto_disables()
        self.save_stats_persistence()

    def process_meta_candidates(self):
        if not ENABLE_META_LABEL_LOG or not self.meta_candidates: return
        now=datetime.now(timezone.utc)
        finished=[]
        for c in self.meta_candidates:
            elapsed_minutes=(now - c['timestamp']).total_seconds()/60
            if elapsed_minutes < META_LABEL_LOOKAHEAD_BARS: continue
            df=self.get_data(c['symbol'], count=400, tf=mt5_timeframe("M1"))
            if df is None: continue
            entry_time=c['timestamp']
            df_sel=df[df['time']>=entry_time]
            if len(df_sel)<2: continue
            ahead=df_sel.iloc[:META_LABEL_LOOKAHEAD_BARS]
            if len(ahead)==0: continue
            dir_mult=1 if c['direction']=="buy" else -1
            entry_price=c['entry_price']
            pip_val,dig,pip_size=self.get_pip_value_and_digits(c['symbol'])
            prices=ahead['close'].values
            diffs=(prices-entry_price)*dir_mult
            mfe=np.max(diffs)/pip_size
            mae=np.min(diffs)/pip_size
            mae=abs(min(mae,0))
            label=1 if (mfe >= META_LABEL_MIN_MFE_FACTOR * c['target_pips'] and mae <= META_LABEL_MAX_MAE_BEFORE_MFE * c['sl_pips']) else 0
            with open(META_LABEL_FINAL_FILE,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([c['cand_id'], c['symbol'], c['direction'], label,
                                        f"{mfe:.2f}", f"{mae:.2f}", c['target_pips'], c['sl_pips'],
                                        int(elapsed_minutes)])
            finished.append(c)
        for c in finished:
            self.meta_candidates.remove(c)

    def log_signal_csv(self, symbol, mode, decision, agg, signals):
        """Log signal sa fallback za missing keys"""
        try:
            row = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'mode': mode,
                'decision_dir': decision.get('direction', 0) if decision else 0,
                'decision_conf': decision.get('confidence', 0) if decision else 0,
                'agg_conf': agg.get('confidence', 0),
                'agg_dir': agg.get('direction', 0),
                'conflict': agg.get('conflict', False),
                'quality': agg.get('quality', 0),
                'cat_scores': agg.get('cat_scores', ''),  # ← FALLBACK
                'meta_score': decision.get('meta_score', 0) if decision else 0,  # ← FALLBACK
                'active_signals': len([s for s in signals if s['raw_direction'] != 0])
            }
        
            with open(SIGNAL_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logging.error(f"log_signal_csv error: {e}")

    def log_trade_csv(self, symbol, res, direction, lot, sl, tp, mode, tier, strategies):
        if not ENABLE_TRADE_CSV or not res: return
        try:
            with open(TRADES_LOG_FILE,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).isoformat(),
                    symbol, res.order, direction, lot,
                    getattr(res,'price',0),
                    sl, tp, mode, tier, strategies
                ])
        except Exception as e:
            logging.error(f"log_trade_csv error: {e}")

    def log_meta_candidate(self, symbol, direction, entry_price, sl, tp, mode, target_pips, sl_pips):
        if not ENABLE_META_LABEL_LOG: return
        try:
            cid=self.meta_candidate_id; self.meta_candidate_id+=1
            rec={"cand_id":cid,"symbol":symbol,"direction":direction,"entry_price":entry_price,
                 "sl":sl,"tp":tp,"mode":mode,"target_pips":target_pips,"sl_pips":sl_pips,
                 "timestamp":datetime.now(timezone.utc)}
            self.meta_candidates.append(rec)
            with open(META_LABEL_CANDIDATES_FILE,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([
                    cid,symbol,rec['timestamp'].isoformat(),direction,entry_price,sl,tp,
                    mode,target_pips,sl_pips,0
                ])
        except Exception as e:
            logging.error(f"log_meta_candidate error: {e}")

    def start(self):
        """Glavni trading loop - POTPUNO ISPRAVLJENA INDENTACIJA"""
        logging.info("🚀 Unified Engine (V6.1) STARTED")
        iteration = 0
        try:
            while True:
                iteration += 1
                self.update_performance_stats()
                self.process_meta_candidates()

                # 🧠 MICRO-RETRAIN SVAKIH 6 SATI
                now = datetime.now(timezone.utc)
                if now.hour % 6 == 0 and now.hour != self.last_micro_retrain:
                    self.last_micro_retrain = now.hour
                    self.micro_retrain_on_feedback()

                if SWING_MODE:
                    self.manage_swing_positions()
                if SCALP_MODE:
                    self.manage_scalps()

                positions_all = mt5.positions_get() or []
                swing_positions = [p for p in positions_all if p.magic == self.magic_number and not p.comment.startswith("SCALP-")]
                scalp_positions = [p for p in positions_all if p.magic == self.magic_number and p.comment.startswith("SCALP-")]

                for symbol in self.symbols:
                    if not self._is_trade_time(symbol):
                        continue
                    logging.info(f"LOOP {symbol} start")
                    self._current_symbol = symbol 
                    try:
                        # ═══════════════════════════════════════════════════════════
                        # SWING MODE
                        # ═══════════════════════════════════════════════════════════
                        if SWING_MODE:
                            df_sw = self.get_data(symbol, count=800, tf=self.swing_tf)
                            logging.info(f"DATA {symbol}: bars={len(df_sw) if df_sw is not None else 0}")
                            if df_sw is not None and len(df_sw) >= MIN_DATA_BARS:
                                last_bar = df_sw['time'].iloc[-1]
                                if symbol not in self.last_bar_time_swing or last_bar > self.last_bar_time_swing[symbol]:
                                    self.last_bar_time_swing[symbol] = last_bar
                                    df_feat = add_features(df_sw)
                                    if self.is_safe_to_trade(symbol, df_sw):
                                        signals = self.collect_signals(df_feat, symbol)
                                    
                                        # DEBUG za određene simbole
                                        if symbol in ("EURUSD", "USDJPY", "XAUUSDs"):
                                            active_dbg = [s for s in signals if s['raw_direction'] != 0]
                                            top_dbg = [f"{s['name']}({s['confidence']:.2f})" for s in 
                                                      sorted(active_dbg, key=lambda x: x['confidence'], reverse=True)[:3]]
                                            logging.info(f"DEBUG {symbol}: {len(active_dbg)}/{len(signals)} | top3: {top_dbg}")
                                    
                                        regime = self.detect_regime(df_feat)
                                        agg = self.aggregate_signals(signals, regime, symbol)
                                    
                                        # ML prediction
                                        try:
                                            if self.model_trained:
                                                features = self.extract_features(df_sw)
                                                proba = self.model.predict_proba(features)[0]
                                                ml_prob = float(proba[1])
                                            else:
                                                ml_prob = 0.5
                                        except Exception as e:
                                            logging.error(f"ML predict error {symbol}: {e}")
                                            ml_prob = 0.5
                                    
                                        combined = self.combine_with_ml(agg, ml_prob)
                                        decision = self.decide_trade(combined, agg, ml_prob, symbol, df_feat, signals)
                                        self.log_signal_csv(symbol, "swing", decision, agg, signals)
                                    
                                        # Trade execution
                                        if decision and ((ml_prob >= self.min_ml_confidence) or (decision['confidence'] >= 0.55)):
                                            sym_open = [p for p in swing_positions if p.symbol == symbol]
                                            if len(sym_open) < 3:
                                                # Prepare context
                                                contrib = [(s['name'], s['confidence'] * self.strategy_weights.get(s['name'], 0.5))
                                                          for s in signals if s['raw_direction'] != 0]
                                                contrib.sort(key=lambda x: x[1], reverse=True)
                                                top_context = ",".join([c[0] for c in contrib[:3]])
                                            
                                                # Calculate stops
                                                sl, tp = self.swing_calc_stops(df_feat, symbol, decision['direction'])
                                            
                                                # Place order (reduce lot if conflict)
                                                if agg['conflict'] and decision['tier'] < 3:
                                                    old = self.base_lot_swing
                                                    self.base_lot_swing *= 0.7
                                                    logging.info(f"🚀 ATTEMPTING ORDER: {symbol} dir={decision['direction']} tier={decision['tier']} conf={decision['confidence']:.2f}")
                                                    res = self.place_order(symbol, decision['direction'], sl, tp, 
                                                                          decision['tier'], None, top_context, scalp=False)
                                                    self.base_lot_swing = old
                                                else:
                                                    res = self.place_order(symbol, decision['direction'], sl, tp, 
                                                                          decision['tier'], None, top_context, scalp=False)
                                            
                                                # Log if successful
                                                if res:
                                                    self.log_trade_csv(symbol, res, decision['direction'], res.volume, 
                                                                      sl, tp, "swing", decision['tier'], top_context)
                                                    self.log_meta_candidate(symbol, decision['direction'], 
                                                                           getattr(res, 'price', 0), sl, tp, "swing", 0, 0)

                        # ═══════════════════════════════════════════════════════════
                        # SCALP MODE
                        # ═══════════════════════════════════════════════════════════
                        if SCALP_MODE:
                            df_sc = self.get_data(symbol, count=800, tf=self.scalp_tf)
                            if df_sc is not None and len(df_sc) >= 300:
                                last_bar = df_sc['time'].iloc[-1]
                                if symbol not in self.last_bar_time_scalp or last_bar > self.last_bar_time_scalp[symbol]:
                                    self.last_bar_time_scalp[symbol] = last_bar
                                    df_feat = add_features(df_sc)
                                    if self.is_safe_to_trade(symbol, df_sc):
                                        signals = self.collect_signals(df_feat, symbol)
                                        regime = self.detect_regime(df_feat)
                                        agg = self.aggregate_signals(signals, regime, symbol)
                                    
                                        # ML prediction
                                        try:
                                            if self.model_trained:
                                                features = self.extract_features(df_sc)
                                                proba = self.model.predict_proba(features)[0]
                                                ml_prob = float(proba[1])
                                            else:
                                                ml_prob = 0.5
                                        except Exception as e:
                                            logging.error(f"ML predict error {symbol}: {e}")
                                            ml_prob = 0.5
                                    
                                        combined = self.combine_with_ml(agg, ml_prob)
                                        decision = self.decide_trade(combined, agg, ml_prob, symbol, df_feat, signals)
                                        self.log_signal_csv(symbol, "scalp", decision, agg, signals)
                                    
                                        # Trade execution
                                        if decision and ((ml_prob >= self.min_ml_confidence) or (decision['confidence'] >= 0.55)):
                                            sym_scalps = [p for p in scalp_positions if p.symbol == symbol]
                                            if len(sym_scalps) < 5:
                                                # Prepare context
                                                contrib = [(s['name'], s['confidence'] * self.strategy_weights.get(s['name'], 0.5))
                                                          for s in signals if s['raw_direction'] != 0]
                                                contrib.sort(key=lambda x: x[1], reverse=True)
                                                top_context = ",".join([c[0] for c in contrib[:3]])
                                            
                                                # Calculate scalp params
                                                lot, sl, tp, target_pips, sl_pips = self.compute_scalp_params(
                                                    symbol, decision['direction'], df_feat)
                                            
                                                # Place order
                                                res = self.place_order(symbol, decision['direction'], sl, tp, 
                                                                      decision['tier'], lot, top_context, scalp=True)
                                            
                                                # Track and log if successful
                                                if res:
                                                    self.scalp_trades[res.order] = {
                                                        "symbol": symbol,
                                                        "direction": decision['direction'],
                                                        "opened": datetime.now(timezone.utc),
                                                        "entry_price": getattr(res, 'price', 0),
                                                        "target_pips": target_pips,
                                                        "sl_pips": sl_pips,
                                                        "max_fav": 0.0,
                                                        "max_adv": 0.0
                                                    }
                                                    self.log_trade_csv(symbol, res, decision['direction'], lot, 
                                                                      sl, tp, "scalp", decision['tier'], top_context)
                                                    self.log_meta_candidate(symbol, decision['direction'], 
                                                                           getattr(res, 'price', 0), sl, tp, 
                                                                           "scalp", target_pips, sl_pips)

                        # Periodic logging
                        if iteration % ITERATION_LOG_INTERVAL == 0:
                            logging.info(f"ℹ️ {symbol} iteration={iteration}")

                    except Exception as e:
                        logging.error(f"Symbol loop error {symbol}: {e}")
                        continue

                # Sleep between iterations
                time.sleep(2)

        except KeyboardInterrupt:
            logging.info("🛑 User stop (Ctrl+C)")
        except Exception as e:
            logging.error(f"Fatal loop error: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            time.sleep(5)
        finally:
            try:
                self.save_stats_persistence()
                logging.info("💾 Stats saved")
            except:
                pass
            try:
                mt5.shutdown()
                logging.info("🔄 MT5 shutdown complete")
            except:
                pass
# =========================================================
# MAIN
# =========================================================
def main():
    if mt5 is None:
        logging.error("MetaTrader5 nije instaliran: pip install MetaTrader5")
        return
    if not mt5.initialize():
        for s in DEFAULT_SYMBOLS:
            rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, 10)
            print(f"{s}: {len(rates) if rates else 0} bars")
        login,password,server=load_env_credentials()
        if login and password and server:
            try: login_i=int(login)
            except: login_i=login
            if not mt5.initialize(login=login_i,password=password,server=server):
                logging.error("MT5 init failed (training stage).")
                return
        else:
            logging.error("MT5 init failed & no credentials.")
            return
    for s in DEFAULT_SYMBOLS:
        try: mt5.symbol_select(s, True)
        except: pass

    trained=False
    if ALWAYS_TRAIN_ON_START or not os.path.exists(MODEL_FILE):
        logging.info("▶️ Training required (initial or missing model).")
        if TUNE_BEFORE_TRAIN and not FORCE_DEFAULT_PARAMS:
            params=run_grid_tuning(DEFAULT_SYMBOLS, tf_out="M5", label_th=LABEL_THRESHOLD, mode=GRID_PARAM_MODE)
            if FORCE_ONLY_TUNING:
                logging.info("FORCE_ONLY_TUNING=True -> završavam nakon tuninga.")
                mt5.shutdown()
                return
        ok=run_training(DEFAULT_SYMBOLS, tf_out="M5", label_th=LABEL_THRESHOLD)
        trained=ok
    else:
        logging.info("Skipping initial training (ALWAYS_TRAIN_ON_START=False)")

    mt5.shutdown()

    bot = UnifiedBot()
    bot.start()

if __name__=="__main__":
    main()
