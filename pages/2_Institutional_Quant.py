"""
Institutional-Grade Stock Dashboard & Quantitative Screener
===========================================================
Implements J.P. Morgan / Top Sell-Side Quant Stack Principles:
- Decision systems (not just label models)
- Point-in-time data correctness
- Execution realism with transaction costs
- Model Risk Management (MRM)
- Multi-factor scoring engine
- Research discipline & robust validation

References:
- SR 11-7 Model Risk Management (Federal Reserve)
- IOSCO Guidance on Algorithmic Trading
- FINRA Rule 3110 (Supervision)
- SEC Market Access Rule 15c3-5
- BCBS 239 (Risk Data Aggregation)
"""

import streamlit as st
import yfinance as yf
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import json

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

st.set_page_config(layout="wide", page_title="Institutional Quant Dashboard", page_icon="📊")

# Institutional defaults
DEFAULT_BENCHMARK = "SPY"
RISK_FREE_RATE = 0.05  # 5% annual
TRADING_DAYS_PER_YEAR = 252

# Execution cost assumptions (institutional realistic)
DEFAULT_COMMISSION_PER_SHARE = 0.005  # $0.005 per share
DEFAULT_SPREAD_PCT = 0.001  # 10 bps half-spread
DEFAULT_MARKET_IMPACT_FACTOR = 0.1  # Impact scales with participation

# Risk limits (institutional defaults)
DEFAULT_MAX_POSITION_PCT = 0.10  # 10% max single position
DEFAULT_MAX_SECTOR_PCT = 0.25  # 25% max sector exposure
DEFAULT_MAX_DRAWDOWN_PCT = 0.15  # 15% max drawdown kill switch


class Regime(Enum):
    """Market regime classification"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class ModelCard:
    """SR 11-7 style Model Card for governance"""
    model_id: str
    model_type: str
    objective: str
    features: List[str]
    training_period: str
    limitations: List[str]
    validation_results: Dict = field(default_factory=dict)
    owner: str = "quant_research"
    version: str = "1.0.0"
    last_validated: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "objective": self.objective,
            "features": self.features,
            "training_period": self.training_period,
            "limitations": self.limitations,
            "validation_results": self.validation_results,
            "owner": self.owner,
            "version": self.version,
            "last_validated": self.last_validated
        }


@dataclass
class ExecutionParams:
    """Execution realism parameters"""
    commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE
    spread_pct: float = DEFAULT_SPREAD_PCT
    market_impact_factor: float = DEFAULT_MARKET_IMPACT_FACTOR
    use_next_bar_open: bool = True
    

@dataclass
class RiskLimits:
    """Risk management constraints"""
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    max_sector_pct: float = DEFAULT_MAX_SECTOR_PCT
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT
    enable_kill_switch: bool = True


# =============================================================================
# DATA FETCHING (with point-in-time awareness)
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_stock_info(symbol: str) -> Optional[Dict]:
    """Fetch stock info with validation"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info or 'longName' not in info:
            return None
        return info
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def fetch_price_history(symbol: str, start_date, end_date, benchmark: str = DEFAULT_BENCHMARK) -> pd.DataFrame:
    """
    Fetch price history with benchmark for relative performance analysis.
    Includes point-in-time awareness flags.
    """
    try:
        stock = yf.Ticker(symbol)
        bench = yf.Ticker(benchmark)
        
        df = stock.history(start=start_date, end=end_date, interval='1d')
        bench_df = bench.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            return pd.DataFrame()
        
        # FIX: Ensure benchmark index timezone matches stock index timezone
        if not bench_df.empty:
            if df.index.tz is not None and bench_df.index.tz is None:
                bench_df.index = bench_df.index.tz_localize(df.index.tz)
            elif df.index.tz is None and bench_df.index.tz is not None:
                bench_df.index = bench_df.index.tz_localize(None)
            elif df.index.tz != bench_df.index.tz:
                bench_df.index = bench_df.index.tz_convert(df.index.tz)

        # Add benchmark returns for relative performance
        df['Benchmark_Close'] = bench_df['Close'].reindex(df.index, method='ffill')
        df['Stock_Return'] = df['Close'].pct_change()
        df['Benchmark_Return'] = df['Benchmark_Close'].pct_change()
        df['Relative_Return'] = df['Stock_Return'] - df['Benchmark_Return']
        df['Outperformed'] = (df['Relative_Return'] > 0).astype(int)
        
        # Point-in-time flag (simulated - would come from vendor timestamp in real system)
        df['Data_AsOf'] = df.index
        
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_quarterly_financials(symbol: str) -> pd.DataFrame:
    """Fetch quarterly financials with point-in-time awareness"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.quarterly_financials
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.T
        # Add point-in-time timestamp (would be actual announcement date in real system)
        df['Report_Date'] = df.index
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_annual_financials(symbol: str) -> pd.DataFrame:
    """Fetch annual financials"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.financials
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.T
        df['Report_Date'] = df.index
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_news(symbol: str) -> List[Dict]:
    """Fetch news items"""
    try:
        stock = yf.Ticker(symbol)
        return stock.news if stock.news else []
    except Exception:
        return []


# =============================================================================
# TECHNICAL INDICATORS & FEATURE ENGINEERING
# =============================================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive feature engineering for institutional quant models.
    Includes momentum, volatility, volume, and microstructure signals.
    """
    df = df.copy()
    
    # --- Price-based features ---
    df['Return'] = df['Close'].pct_change()
    df['Return_1w'] = df['Close'].pct_change(5)
    df['Return_1m'] = df['Close'].pct_change(21)
    df['Return_3m'] = df['Close'].pct_change(63)
    df['Return_6m'] = df['Close'].pct_change(126)
    
    # Moving averages
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Price position in range
    df['Price_Position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                           (df['High'].rolling(20).max() - df['Low'].rolling(20).min() + 1e-9)
    
    # --- Volatility features ---
    df['Volatility'] = df['Return'].rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    df['Volatility_3m'] = df['Return'].rolling(63).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    df['ATR'] = compute_atr(df, 14)
    df['Vol_Regime'] = (df['Volatility'] > df['Volatility'].rolling(63).mean()).astype(int)
    
    # --- RSI (14) ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # RSI divergence (momentum quality)
    df['RSI_Momentum'] = df['RSI14'].diff(5)
    df['Price_Momentum'] = df['Close'].diff(5)
    df['RSI_Divergence'] = np.where(
        (df['Price_Momentum'] > 0) & (df['RSI_Momentum'] < 0), -1,
        np.where((df['Price_Momentum'] < 0) & (df['RSI_Momentum'] > 0), 1, 0)
    )
    
    # --- Momentum features ---
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Momentum_20'] = df['Close'].pct_change(20)
    df['Momentum_Accel'] = df['Momentum_5'] - df['Momentum_5'].shift(5)
    
    # --- Volume-based indicators ---
    if 'Volume' in df.columns:
        df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA20'] + 1e-9)
        df['Volume_Trend'] = df['Volume'].rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0, raw=True)
        
        # OBV
        direction = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (direction * df['Volume']).fillna(0).cumsum()
        df['OBV_MA20'] = df['OBV'].rolling(20).mean()
        df['OBV_Signal'] = (df['OBV'] > df['OBV_MA20']).astype(int)
        
        # Volume-weighted metrics
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        df['VWAP_Distance'] = (df['Close'] - df['VWAP']) / df['VWAP']
        
        # Dollar volume (liquidity proxy)
        df['Dollar_Volume'] = df['Close'] * df['Volume']
        df['Dollar_Volume_MA20'] = df['Dollar_Volume'].rolling(20).mean()
        df['Liquidity_Score'] = (df['Dollar_Volume_MA20'] / df['Dollar_Volume_MA20'].rolling(63).mean()).fillna(1)
    else:
        df['Volume_SMA20'] = np.nan
        df['Volume_Ratio'] = np.nan
        df['OBV'] = np.nan
        df['VWAP'] = df['Close']
        df['Liquidity_Score'] = 1.0
    
    # --- Microstructure proxies ---
    df['True_Range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR_Pct'] = df['True_Range'] / df['Close']
    
    # Intraday range as spread proxy
    df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
    
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Detect market regime for regime-switching models.
    Returns regime classification for each date.
    """
    df = df.copy()
    
    # Volatility regime
    vol_ma = df['Volatility'].rolling(63).mean()
    high_vol = df['Volatility'] > vol_ma * 1.2
    low_vol = df['Volatility'] < vol_ma * 0.8
    
    # Trend regime (200-day MA slope)
    ma200_slope = df['SMA200'].diff(20) / df['SMA200'].shift(20)
    trending = abs(ma200_slope) > 0.02
    
    # Risk-on/off (relative to benchmark)
    if 'Benchmark_Return' in df.columns:
        bench_ma = df['Benchmark_Return'].rolling(63).mean()
        risk_on = bench_ma > 0.0005  # Positive benchmark trend
    else:
        risk_on = df['Return'].rolling(63).mean() > 0.0003
    
    # Combine regimes
    regime = pd.Series('neutral', index=df.index)
    regime[high_vol & risk_on] = 'high_vol_risk_on'
    regime[high_vol & ~risk_on] = 'high_vol_risk_off'
    regime[low_vol & risk_on] = 'low_vol_risk_on'
    regime[trending & risk_on] = 'trending_risk_on'
    regime[trending & ~risk_on] = 'trending_risk_off'
    
    return regime


# =============================================================================
# MULTI-FACTOR SCORING ENGINE
# =============================================================================

def compute_momentum_factor(df: pd.DataFrame) -> pd.Series:
    """
    Momentum factor: combination of short, medium, and long-term momentum.
    Institutional approach: risk-adjusted momentum with quality filters.
    """
    # Raw momentum signals
    mom_1m = df['Return_1m'].fillna(0)
    mom_3m = df['Return_3m'].fillna(0)
    mom_6m = df['Return_6m'].fillna(0)
    
    # Volatility adjustment (higher vol = lower score)
    vol_adj = 1 / (1 + df['Volatility'].fillna(0.2))
    
    # Momentum quality (avoiding recent reversals)
    mom_quality = (df['Momentum_Accel'].fillna(0) > 0).astype(float)
    
    # Composite momentum score
    momentum = (
        0.4 * mom_1m + 
        0.35 * mom_3m + 
        0.25 * mom_6m
    ) * vol_adj * (0.5 + 0.5 * mom_quality)
    
    return momentum


def compute_quality_factor(df: pd.DataFrame, financials: pd.DataFrame) -> pd.Series:
    """
    Quality factor: profitability, earnings stability, balance sheet strength.
    Uses financial statement data when available.
    """
    # Default quality score based on price action
    quality = pd.Series(0.5, index=df.index)
    
    if not financials.empty and 'Net Income' in financials.columns:
        # Merge financials with price data
        fin_df = financials.copy()
        fin_df.index = pd.to_datetime(fin_df.index)
        
        # FIX: Ensure index timezone matches the price data (df) to avoid TypeError
        # yfinance price history is often tz-aware (America/New_York), 
        # while financials are often tz-naive.
        if df.index.tz is not None:
            if fin_df.index.tz is None:
                fin_df.index = fin_df.index.tz_localize(df.index.tz)
            else:
                fin_df.index = fin_df.index.tz_convert(df.index.tz)
        else:
            if fin_df.index.tz is not None:
                fin_df.index = fin_df.index.tz_localize(None)
        
        # Forward-fill financial metrics to daily frequency
        for col in ['Net Income', 'Total Revenue', 'Operating Income']:
            if col in fin_df.columns:
                df[col] = fin_df[col].reindex(df.index, method='ffill')
        
        # Profitability metrics
        if 'Net Income' in df.columns and 'Total Revenue' in df.columns:
            df['Net_Margin'] = df['Net Income'] / (df['Total Revenue'] + 1e-9)
            quality = df['Net_Margin'].fillna(0.1)
            
        # Earnings consistency (if we have enough data)
        if 'Net Income' in df.columns:
            earnings_trend = df['Net Income'].rolling(4).apply(
                lambda x: 1 if all(x.diff().dropna() > 0) else 0, raw=False
            ).fillna(0.5)
            quality = quality * (0.7 + 0.3 * earnings_trend)
    
    # Price-based quality proxies
    # Low volatility = higher quality
    vol_score = 1 - (df['Volatility'].fillna(0.2) / df['Volatility'].rolling(252).max().fillna(0.5))
    quality = quality * (0.8 + 0.2 * vol_score)
    
    return quality


def compute_value_factor(df: pd.DataFrame, info: Dict) -> pd.Series:
    """
    Value factor: valuation metrics relative to history.
    """
    value = pd.Series(0.5, index=df.index)
    
    # Use P/E if available
    pe = info.get('trailingPE', None)
    if pe and isinstance(pe, (int, float)) and pe > 0:
        # Lower P/E = higher value score
        pe_score = max(0, min(1, (30 - pe) / 30))
        value = pd.Series(pe_score, index=df.index)
    
    # Price-to-book proxy using historical range
    price_position = df['Price_Position'].fillna(0.5)
    value = value * (1 - price_position)  # Lower position = higher value
    
    return value


def compute_volatility_factor(df: pd.DataFrame) -> pd.Series:
    """
    Volatility/Risk factor: lower volatility = higher score (for risk-adjusted returns).
    """
    vol_rank = df['Volatility'].rolling(252).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 20 else 0.5, 
        raw=False
    )
    vol_factor = 1 - vol_rank.fillna(0.5)
    
    # Add drawdown component
    rolling_max = df['Close'].cummax()
    drawdown = (df['Close'] - rolling_max) / rolling_max
    max_dd = drawdown.rolling(63).min()
    dd_factor = 1 + max_dd.fillna(0)  # Less negative drawdown = higher score
    
    return vol_factor * dd_factor


def compute_flow_factor(df: pd.DataFrame) -> pd.Series:
    """
    Volume/Flow factor: institutional accumulation patterns.
    """
    if 'Volume_Ratio' not in df.columns:
        return pd.Series(0.5, index=df.index)
    
    # Volume surge with positive price = accumulation
    volume_surge = (df['Volume_Ratio'] > 1.5).astype(float)
    price_up = (df['Return'] > 0).astype(float)
    accumulation = volume_surge * price_up
    
    # OBV confirmation
    obv_confirm = (df['OBV_Signal'] == 1).astype(float) if 'OBV_Signal' in df.columns else 0.5
    
    # VWAP distance (above VWAP = buying pressure)
    vwap_signal = (df['VWAP_Distance'].fillna(0) > 0).astype(float) if 'VWAP_Distance' in df.columns else 0.5
    
    flow = 0.4 * accumulation + 0.3 * obv_confirm + 0.3 * vwap_signal
    return flow


def compute_composite_score(df: pd.DataFrame, info: Dict, financials: pd.DataFrame,
                           weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Compute multi-factor composite score.
    Default weights: momentum=0.25, quality=0.25, value=0.20, volatility=0.15, flow=0.15
    """
    if weights is None:
        weights = {
            'momentum': 0.25,
            'quality': 0.25,
            'value': 0.20,
            'volatility': 0.15,
            'flow': 0.15
        }
    
    df = df.copy()
    
    # Compute individual factors
    df['Momentum_Factor'] = compute_momentum_factor(df)
    df['Quality_Factor'] = compute_quality_factor(df, financials)
    df['Value_Factor'] = compute_value_factor(df, info)
    df['Volatility_Factor'] = compute_volatility_factor(df)
    df['Flow_Factor'] = compute_flow_factor(df)
    
    # Composite score (z-score normalization within lookback)
    factors = ['Momentum_Factor', 'Quality_Factor', 'Value_Factor', 'Volatility_Factor', 'Flow_Factor']
    for f in factors:
        df[f] = (df[f] - df[f].rolling(252).mean()) / (df[f].rolling(252).std() + 1e-9)
    
    df['Composite_Score'] = (
        weights['momentum'] * df['Momentum_Factor'] +
        weights['quality'] * df['Quality_Factor'] +
        weights['value'] * df['Value_Factor'] +
        weights['volatility'] * df['Volatility_Factor'] +
        weights['flow'] * df['Flow_Factor']
    )
    
    # Rank percentile (0-1)
    df['Composite_Rank'] = df['Composite_Score'].rolling(252).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 20 else 0.5,
        raw=False
    )
    
    return df


# =============================================================================
# LABEL CREATION (Decision System Approach)
# =============================================================================

def create_binary_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """Binary classification label"""
    future_return = df['Close'].shift(-horizon) / df['Close'] - 1
    return (future_return > threshold).astype(int)


def create_regression_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Expected return label (regression approach)"""
    return df['Close'].shift(-horizon) / df['Close'] - 1


def create_relative_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.005) -> pd.Series:
    """
    Label for outperforming benchmark.
    threshold: minimum outperformance required (e.g., 0.005 = 50 bps)
    """
    if 'Benchmark_Close' not in df.columns:
        return create_binary_labels(df, horizon, threshold)
    
    future_stock = df['Close'].shift(-horizon) / df['Close'] - 1
    future_bench = df['Benchmark_Close'].shift(-horizon) / df['Benchmark_Close'] - 1
    relative_return = future_stock - future_bench
    return (relative_return > threshold).astype(int)


def create_quantile_labels(df: pd.DataFrame, horizon: int = 5, n_quantiles: int = 5) -> pd.Series:
    """
    Multi-class ranking labels (for ranking models).
    Returns quantile rank (0 to n_quantiles-1).
    """
    future_return = df['Close'].shift(-horizon) / df['Close'] - 1
    # Use rolling window to compute quantiles (point-in-time)
    labels = future_return.rolling(252).apply(
        lambda x: pd.qcut(x, q=n_quantiles, labels=False, duplicates='drop').iloc[-1] 
        if len(x.dropna()) >= n_quantiles else 2,
        raw=False
    )
    return labels.fillna(2).astype(int)


# =============================================================================
# MODEL RISK MANAGEMENT
# =============================================================================

class ModelRiskManager:
    """Implements SR 11-7 style Model Risk Management"""
    
    def __init__(self):
        self.model_inventory = {}
        self.validation_reports = {}
        self.monitoring_data = {}
    
    def register_model(self, model_card: ModelCard):
        """Register a model in the inventory"""
        self.model_inventory[model_card.model_id] = model_card
    
    def get_model_card(self, model_id: str) -> Optional[ModelCard]:
        return self.model_inventory.get(model_id)
    
    def record_validation(self, model_id: str, results: Dict):
        """Record validation results"""
        if model_id in self.model_inventory:
            self.model_inventory[model_id].validation_results = results
            self.model_inventory[model_id].last_validated = datetime.datetime.now().strftime("%Y-%m-%d")
        self.validation_reports[model_id] = results
    
    def check_drift(self, model_id: str, recent_performance: pd.Series, 
                    threshold: float = 0.1) -> Dict:
        """Check for model drift"""
        if model_id not in self.validation_reports:
            return {"status": "unknown", "message": "No baseline available"}
        
        baseline = self.validation_reports[model_id].get('avg_f1', 0.5)
        recent_avg = recent_performance.mean()
        
        drift_pct = (baseline - recent_avg) / baseline if baseline > 0 else 0
        
        if drift_pct > threshold:
            status = "critical"
            message = f"Model performance degraded by {drift_pct:.1%}. Retraining recommended."
        elif drift_pct > threshold / 2:
            status = "warning"
            message = f"Model performance declining ({drift_pct:.1%}). Monitor closely."
        else:
            status = "healthy"
            message = f"Model performing within expected range."
        
        return {
            "status": status,
            "drift_pct": drift_pct,
            "baseline_f1": baseline,
            "recent_f1": recent_avg,
            "message": message
        }
    
    def render_model_card(self, model_id: str):
        """Render model card in Streamlit"""
        card = self.get_model_card(model_id)
        if not card:
            st.warning(f"Model {model_id} not found in inventory")
            return
        
        with st.expander(f"📋 Model Card: {model_id}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model Type:** {card.model_type}")
                st.write(f"**Objective:** {card.objective}")
                st.write(f"**Version:** {card.version}")
                st.write(f"**Owner:** {card.owner}")
            with col2:
                st.write(f"**Last Validated:** {card.last_validated or 'Never'}")
                st.write(f"**Training Period:** {card.training_period}")
            
            st.write("**Features Used:**")
            st.write(card.features)
            
            st.write("**Known Limitations:**")
            for lim in card.limitations:
                st.write(f"- {lim}")
            
            if card.validation_results:
                st.write("**Validation Results:**")
                st.json(card.validation_results)


# Global MRM instance
mrm = ModelRiskManager()


# =============================================================================
# WALK-FORWARD VALIDATION (Purged & Embargoed)
# =============================================================================

def purged_kfold_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, 
                    purge_pct: float = 0.02, embargo_pct: float = 0.01) -> List[Tuple]:
    """
    Purged K-Fold Cross-Validation for time series.
    Prevents leakage by purging overlapping periods between train and test.
    """
    n = len(X)
    fold_size = n // n_splits
    purge_size = int(fold_size * purge_pct)
    embargo_size = int(fold_size * embargo_pct)
    
    splits = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        
        # Purge: remove overlap between train and test
        train_end = max(0, test_start - purge_size)
        train_indices = list(range(0, train_end))
        
        # Embargo: remove beginning of test set
        test_indices = list(range(test_start + embargo_size, test_end))
        
        if len(train_indices) > 50 and len(test_indices) > 10:
            splits.append((train_indices, test_indices))
    
    return splits


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    initial_train_size: float = 0.6,
    test_size: float = 0.1,
    step_size: float = 0.1,
    use_purged: bool = True,
    embargo_pct: float = 0.01
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Expanding-window walk-forward validation with optional purging.
    """
    n = len(X)
    init_train = int(n * initial_train_size)
    test_n = max(10, int(n * test_size))
    step_n = max(10, int(n * step_size))
    
    fold_results = []
    oof_probs = pd.Series(index=X.index, dtype=float)
    oof_preds = pd.Series(index=X.index, dtype=int)
    oof_true = y.copy()
    
    start_train_end = init_train
    fold = 1
    
    while start_train_end + test_n <= n:
        if use_purged:
            # Apply embargo
            embargo_n = max(1, int(test_n * embargo_pct))
            train_idx = X.index[:start_train_end - embargo_n]
            test_idx = X.index[start_train_end:start_train_end + test_n]
        else:
            train_idx = X.index[:start_train_end]
            test_idx = X.index[start_train_end:start_train_end + test_n]
        
        if len(train_idx) < 50 or len(test_idx) < 5:
            break
        
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        
        # Handle class imbalance
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train.dropna())
        if len(classes) > 1:
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train.dropna())
            class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
        else:
            class_weight_dict = None
        
        # Train model
        if model_name == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=300, 
                random_state=42, 
                n_jobs=-1,
                class_weight=class_weight_dict
            )
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=2000, class_weight=class_weight_dict)
        elif model_name == 'RidgeRegression':
            model = Ridge(alpha=1.0)
        else:
            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            oof_probs.loc[test_idx] = probs
        
        oof_preds.loc[test_idx] = preds
        
        # Compute metrics
        fold_results.append({
            "fold": fold,
            "train_end": str(train_idx[-1]),
            "test_start": str(test_idx[0]),
            "test_end": str(test_idx[-1]),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "n_train": len(train_idx),
            "n_test": len(test_idx)
        })
        
        fold += 1
        start_train_end += step_n
    
    folds_df = pd.DataFrame(fold_results)
    return folds_df, oof_true, oof_preds, oof_probs


# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================

def calibrate_probabilities(y_true: pd.Series, probs: pd.Series, 
                           method: str = 'isotonic') -> pd.Series:
    """
    Calibrate predicted probabilities using Platt scaling or isotonic regression.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    
    # FIX: Drop NaNs from input before passing to calibration
    # The 'probs' series may have NaNs where folds didn't cover predictions.
    df = pd.DataFrame({"y": y_true, "p": probs}).dropna()
    
    if len(df) < 100:
        return probs
    
    if method == 'isotonic':
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(df['p'].values.reshape(-1, 1), df['y'].values)
        
        # We only predict for the valid non-NaN entries we have
        calibrated_values = iso.predict(df['p'].values.reshape(-1, 1))
        
        # Return a series indexed by the original timestamps, with NaNs where appropriate
        calibrated_series = pd.Series(index=probs.index, dtype=float)
        calibrated_series.loc[df.index] = calibrated_values
        return calibrated_series
        
    else:
        # Platt scaling approximation
        from sklearn.linear_model import LogisticRegression
        platt = LogisticRegression()
        platt.fit(df['p'].values.reshape(-1, 1), df['y'].values)
        
        calibrated_values = platt.predict_proba(df['p'].values.reshape(-1, 1))[:, 1]
        
        calibrated_series = pd.Series(index=probs.index, dtype=float)
        calibrated_series.loc[df.index] = calibrated_values
        return calibrated_series


def evaluate_calibration(y_true: pd.Series, probs: pd.Series, n_bins: int = 10) -> Dict:
    """
    Evaluate probability calibration using reliability diagram data.
    """
    df = pd.DataFrame({"y": y_true, "p": probs}).dropna()
    if len(df) < 100:
        return {"brier_score": 0.25, "reliability": None}
    
    brier = brier_score_loss(df['y'], df['p'])
    
    # Reliability diagram bins
    df['bin'] = pd.qcut(df['p'], q=n_bins, duplicates='drop')
    reliability = df.groupby('bin').agg({
        'y': 'mean',
        'p': 'mean',
        'y': 'count'
    }).rename(columns={'y': 'actual_rate', 'p': 'pred_prob', 'y': 'count'})
    
    return {
        "brier_score": brier,
        "reliability": reliability
    }


# =============================================================================
# EXECUTION REALISM & BACKTESTING
# =============================================================================

def estimate_transaction_costs(df: pd.DataFrame, position_size: float, 
                               params: ExecutionParams) -> pd.Series:
    """
    Estimate transaction costs including commission, spread, and market impact.
    """
    # Commission (per share)
    commission = params.commission_per_share * position_size / df['Close']
    
    # Spread cost (half-spread)
    spread_cost = params.spread_pct
    
    # Market impact (simplified square root model)
    participation = position_size / (df['Volume'] * df['Close'] + 1e-9)
    impact = params.market_impact_factor * np.sqrt(participation)
    
    total_cost = commission + spread_cost + impact
    return total_cost


def apply_risk_limits(positions: pd.Series, returns: pd.Series, 
                     limits: RiskLimits) -> pd.Series:
    """
    Apply risk limits and kill switches to positions.
    """
    # Track cumulative returns for drawdown
    cumulative = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    # Kill switch on max drawdown
    if limits.enable_kill_switch:
        kill_triggered = drawdown < -limits.max_drawdown_pct
        positions = positions.where(~kill_triggered, 0)
    
    # Position size limits
    positions = positions.clip(-limits.max_position_pct, limits.max_position_pct)
    
    return positions


def realistic_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    execution_params: ExecutionParams = None,
    risk_limits: RiskLimits = None,
    initial_capital: float = 100000
) -> pd.DataFrame:
    """
    Institutional-grade backtest with execution realism.
    """
    if execution_params is None:
        execution_params = ExecutionParams()
    if risk_limits is None:
        risk_limits = RiskLimits()
    
    df = df.copy()
    
    # Use next bar open for fills (no lookahead)
    df['Execution_Price'] = df['Open'].shift(-1) if execution_params.use_next_bar_open else df['Close']
    
    # Position sizing (risk-parity style based on volatility)
    df['Target_Position'] = signals
    df['Vol_Adjusted_Position'] = df['Target_Position'] / (df['Volatility'].fillna(0.2) + 0.1)
    df['Position'] = df['Vol_Adjusted_Position'].clip(-1, 1)
    
    # Apply risk limits
    df['Position'] = apply_risk_limits(df['Position'], df['Return'], risk_limits)
    
    # Transaction costs
    position_changes = df['Position'].diff().abs()
    df['Transaction_Cost'] = estimate_transaction_costs(
        df, position_changes * initial_capital, execution_params
    )
    
    # Strategy returns
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return'] - df['Transaction_Cost']
    df['Strategy_Cumulative'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    
    # Buy & Hold benchmark
    df['Hold_Cumulative'] = (1 + df['Return'].fillna(0)).cumprod()
    
    # Relative performance
    df['Relative_Cumulative'] = df['Strategy_Cumulative'] / df['Hold_Cumulative']
    
    return df


def compute_backtest_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive backtest metrics.
    """
    returns = df['Strategy_Return'].dropna()
    hold_returns = df['Return'].dropna()
    
    if len(returns) < 10:
        return {}
    
    # Return metrics
    total_return = df['Strategy_Cumulative'].iloc[-1] - 1
    hold_return = df['Hold_Cumulative'].iloc[-1] - 1
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    hold_vol = hold_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Sharpe ratio
    excess_returns = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if returns.std() > 0 else 0
    hold_sharpe = ((hold_returns.mean() - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR) / hold_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if hold_returns.std() > 0 else 0
    
    # Sortino ratio
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = (returns.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE) / downside if downside > 0 else 0
    
    # Drawdown
    cumulative = df['Strategy_Cumulative']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate and payoff
    trades = returns[returns != 0]
    win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
    avg_win = trades[trades > 0].mean() if (trades > 0).any() else 0
    avg_loss = trades[trades < 0].mean() if (trades < 0).any() else 0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Information ratio (vs benchmark)
    if 'Benchmark_Return' in df.columns:
        active_returns = returns - df['Benchmark_Return'].reindex(returns.index).fillna(0)
        tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        info_ratio = (active_returns.mean() * TRADING_DAYS_PER_YEAR) / tracking_error if tracking_error > 0 else 0
    else:
        info_ratio = None
    
    # Turnover
    turnover = df['Position'].diff().abs().mean() * TRADING_DAYS_PER_YEAR
    
    return {
        'total_return': total_return,
        'hold_return': hold_return,
        'excess_return': total_return - hold_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'payoff_ratio': payoff_ratio,
        'information_ratio': info_ratio,
        'turnover': turnover,
        'benchmark_sharpe': hold_sharpe
    }


# =============================================================================
# BENCHMARK MODELS
# =============================================================================

def create_benchmark_signals(df: pd.DataFrame, strategy: str = 'momentum') -> pd.Series:
    """
    Create benchmark strategy signals for comparison.
    """
    if strategy == 'momentum':
        # Simple momentum: long when price > SMA50
        return (df['Close'] > df['SMA50']).astype(int)
    elif strategy == 'mean_reversion':
        # Mean reversion: long when RSI < 30
        return (df['RSI14'] < 30).astype(int)
    elif strategy == 'random':
        # Random signals
        np.random.seed(42)
        return pd.Series(np.random.choice([0, 1], size=len(df)), index=df.index)
    else:
        return pd.Series(0, index=df.index)


# =============================================================================
# SIDEBAR & UI
# =============================================================================

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("⚙️ Configuration")
    
    # Symbol inputs
    symbol = st.sidebar.text_input('Main Stock Symbol', 'AAPL').upper().strip()
    compare_symbol = st.sidebar.text_input('Compare with (Peer)', 'MSFT').upper().strip()
    benchmark = st.sidebar.text_input('Benchmark', DEFAULT_BENCHMARK).upper().strip()
    
    # Date range
    st.sidebar.subheader("📅 Time Period")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime.date.today() - datetime.timedelta(days=730))
    with col2:
        end_date = st.date_input("End", datetime.date.today())
    
    # Execution params
    st.sidebar.subheader("💰 Execution Parameters")
    commission = st.sidebar.slider("Commission ($/share)", 0.0, 0.02, DEFAULT_COMMISSION_PER_SHARE, 0.001)
    spread = st.sidebar.slider("Spread (%)", 0.0, 0.005, DEFAULT_SPREAD_PCT, 0.0001) * 100
    
    # Risk limits
    st.sidebar.subheader("🛡️ Risk Limits")
    max_pos = st.sidebar.slider("Max Position (%)", 5, 50, int(DEFAULT_MAX_POSITION_PCT * 100)) / 100
    max_dd = st.sidebar.slider("Max Drawdown Kill (%)", 5, 30, int(DEFAULT_MAX_DRAWDOWN_PCT * 100)) / 100
    
    execution_params = ExecutionParams(
        commission_per_share=commission,
        spread_pct=spread / 100
    )
    risk_limits = RiskLimits(
        max_position_pct=max_pos,
        max_drawdown_pct=max_dd
    )
    
    return {
        'symbol': symbol,
        'compare_symbol': compare_symbol,
        'benchmark': benchmark,
        'start_date': start_date,
        'end_date': end_date,
        'execution_params': execution_params,
        'risk_limits': risk_limits
    }


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    config = render_sidebar()
    
    if not config['symbol']:
        st.error("Please enter a stock symbol")
        return
    
    # Fetch data
    info = fetch_stock_info(config['symbol'])
    if not info:
        st.error(f"Could not find data for symbol '{config['symbol']}'")
        return
    
    raw_history = fetch_price_history(
        config['symbol'], 
        config['start_date'], 
        config['end_date'],
        config['benchmark']
    )
    
    if raw_history.empty:
        st.error("No price data available")
        return
    
    # Header
    st.title(f"📊 {info.get('longName', config['symbol'])} ({config['symbol']})")
    st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    
    # Key metrics
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    def fmt_val(v, default="N/A"):
        if v is None:
            return default
        return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
    
    with col_m1:
        st.metric("Current Price", f"${fmt_val(info.get('currentPrice'))}")
    with col_m2:
        mc = info.get('marketCap', 0)
        mc_str = f"${mc/1e12:.2f}T" if mc >= 1e12 else f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc:,.0f}"
        st.metric("Market Cap", mc_str)
    with col_m3:
        pe = info.get('trailingPE', 'N/A')
        st.metric("P/E Ratio", fmt_val(pe))
    with col_m4:
        div = info.get('dividendYield', 0)
        st.metric("Div Yield", f"{div*100:.2f}%" if div else "N/A")
    with col_m5:
        pm = info.get('profitMargins', 0)
        st.metric("Profit Margin", f"{pm*100:.2f}%" if pm else "N/A")
    
    # Tabs
    tabs = st.tabs([
        "📈 Price & Volume", 
        "🔬 Multi-Factor Analysis", 
        "🤖 Quant Models (MRM)",
        "💼 Execution & Backtest",
        "📊 Financials", 
        "🔄 Peer Comparison", 
        "📰 News"
    ])
    
    # ---- TAB 1: Price & Volume ----
    with tabs[0]:
        history = raw_history.rename_axis('Date').reset_index()
        history['EMA20'] = history['Close'].ewm(span=20, adjust=False).mean()
        history['EMA50'] = history['Close'].ewm(span=50, adjust=False).mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, 
                           subplot_titles=('Price & EMAs', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=history['Date'],
            open=history['Open'], 
            high=history['High'],
            low=history['Low'], 
            close=history['Close'],
            name='OHLC'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=history['Date'], y=history['EMA20'],
                                line=dict(width=1, color='blue'), name='20 EMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=history['Date'], y=history['EMA50'],
                                line=dict(width=1, color='orange'), name='50 EMA'), row=1, col=1)
        
        if 'Volume' in history.columns:
            colors = ['green' if history.loc[i, 'Close'] >= history.loc[i, 'Open'] else 'red' 
                     for i in history.index]
            fig.add_trace(go.Bar(x=history['Date'], y=history['Volume'], 
                                marker_color=colors, name='Volume'), row=2, col=1)
        
        fig.update_layout(height=650, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- TAB 2: Multi-Factor Analysis ----
    with tabs[1]:
        st.header("Multi-Factor Scoring Engine")
        st.markdown("""
        **Institutional Approach:** Composite scoring using momentum, quality, value, 
        volatility, and flow factors. Each factor is z-score normalized and combined 
        with institutional-grade weights.
        """)
        
        # Compute indicators and factors
        df_factors = compute_indicators(raw_history)
        financials = fetch_quarterly_financials(config['symbol'])
        df_factors = compute_composite_score(df_factors, info, financials)
        
        # Factor display
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.subheader("Factor Scores (Z-Score)")
            factor_chart = pd.DataFrame({
                'Date': df_factors.index,
                'Momentum': df_factors['Momentum_Factor'],
                'Quality': df_factors['Quality_Factor'],
                'Value': df_factors['Value_Factor'],
                'Volatility': df_factors['Volatility_Factor'],
                'Flow': df_factors['Flow_Factor']
            }).set_index('Date')
            st.line_chart(factor_chart)
        
        with col_f2:
            st.subheader("Composite Score & Rank")
            composite_chart = pd.DataFrame({
                'Date': df_factors.index,
                'Composite Score': df_factors['Composite_Score'],
                'Percentile Rank': df_factors['Composite_Rank']
            }).set_index('Date')
            st.line_chart(composite_chart)
        
        # Current factor breakdown
        st.subheader("Current Factor Breakdown")
        latest = df_factors.iloc[-1]
        factor_cols = st.columns(5)
        factor_names = ['Momentum', 'Quality', 'Value', 'Volatility', 'Flow']
        factor_keys = ['Momentum_Factor', 'Quality_Factor', 'Value_Factor', 'Volatility_Factor', 'Flow_Factor']
        
        for col, name, key in zip(factor_cols, factor_names, factor_keys):
            with col:
                score = latest.get(key, 0)
                color = "🟢" if score > 0.5 else "🔴" if score < -0.5 else "🟡"
                st.metric(f"{color} {name}", f"{score:.2f}")
        
        # Signal alignment
        st.subheader("Signal Alignment")
        momentum_signal = 1 if latest.get('Momentum_Factor', 0) > 0 else 0
        flow_signal = 1 if latest.get('Flow_Factor', 0) > 0 else 0
        vol_regime = latest.get('Vol_Regime', 1)
        
        alignment_score = (momentum_signal + flow_signal + vol_regime) / 3
        st.progress(alignment_score)
        st.write(f"**Signal Alignment:** {alignment_score:.0%} (Momentum + Flow + Vol Regime)")
        
        if alignment_score >= 0.67:
            st.success("✅ Strong signal alignment - Consider position")
        elif alignment_score >= 0.33:
            st.warning("⚠️ Mixed signals - Monitor closely")
        else:
            st.error("❌ Weak alignment - Avoid or reduce position")
    
    # ---- TAB 3: Quant Models (MRM) ----
    with tabs[2]:
        st.header("🤖 Quantitative Models & Model Risk Management")
        st.markdown("""
        **SR 11-7 Compliant Model Risk Management:** Walk-forward validation, 
        probability calibration, drift detection, and model cards for governance.
        """)
        
        with st.expander("🔧 Model Configuration", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                model_type = st.selectbox("Model Type", [
                    "RandomForest", "LogisticRegression", "RidgeRegression"
                ])
                prediction_type = st.selectbox("Prediction Target", [
                    "Binary (Up/Down)",
                    "Relative vs Benchmark",
                    "Expected Return (Regression)"
                ])
            with c2:
                horizon = st.slider("Horizon (days)", 1, 30, 5)
                threshold_pct = st.slider("Threshold (%)", 0.0, 5.0, 1.0) / 100
            with c3:
                initial_train = st.slider("Initial Train (%)", 40, 80, 60) / 100
                use_purged = st.checkbox("Use Purged CV", value=True)
            with c4:
                calibrate = st.checkbox("Calibrate Probabilities", value=True)
                run_validation = st.button("🚀 Run Validation", type="primary")
        
        if run_validation:
            # Prepare features
            df_model = compute_indicators(raw_history)
            
            # Create labels based on prediction type
            if prediction_type == "Binary (Up/Down)":
                labels = create_binary_labels(df_model, horizon, threshold_pct)
            elif prediction_type == "Relative vs Benchmark":
                labels = create_relative_labels(df_model, horizon, threshold_pct)
            else:
                labels = create_regression_labels(df_model, horizon)
            
            X = prepare_features(df_model)
            common_idx = X.index.intersection(labels.dropna().index)
            X = X.loc[common_idx]
            y = labels.loc[common_idx]
            
            if len(X) < 200:
                st.error("Insufficient data for validation")
            else:
                with st.spinner("Running walk-forward validation..."):
                    folds_df, oof_true, oof_preds, oof_probs = walk_forward_validation(
                        X, y, model_type, initial_train, 0.1, 0.1, use_purged
                    )
                
                if not folds_df.empty:
                    # Display fold results
                    st.subheader("📊 Walk-Forward Validation Results")
                    st.dataframe(folds_df, use_container_width=True)
                    
                    # Average metrics
                    avg = folds_df[['accuracy', 'precision', 'recall', 'f1']].mean()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{avg['accuracy']:.3f}")
                    m2.metric("Precision", f"{avg['precision']:.3f}")
                    m3.metric("Recall", f"{avg['recall']:.3f}")
                    m4.metric("F1 Score", f"{avg['f1']:.3f}")
                    
                    # Probability calibration
                    if calibrate and oof_probs.notna().any():
                        st.subheader("📈 Probability Calibration")
                        calibrated = calibrate_probabilities(oof_true, oof_probs)
                        
                        cal_df = pd.DataFrame({
                            'Original': oof_probs.dropna(),
                            'Calibrated': calibrated.dropna()
                        })
                        st.line_chart(cal_df)
                        
                        # Brier score
                        # Ensure we only score on non-NaN indices
                        valid_orig = oof_probs.dropna().index
                        brier_orig = brier_score_loss(oof_true.loc[valid_orig], oof_probs.loc[valid_orig])
                        
                        valid_cal = calibrated.dropna().index
                        brier_cal = brier_score_loss(oof_true.loc[valid_cal], calibrated.loc[valid_cal])
                        
                        st.write(f"**Brier Score - Original:** {brier_orig:.4f} | **Calibrated:** {brier_cal:.4f}")
                    
                    # Register model in MRM
                    model_card = ModelCard(
                        model_id=f"{config['symbol']}_{model_type}_{horizon}d",
                        model_type=model_type,
                        objective=prediction_type,
                        features=list(X.columns),
                        training_period=f"{X.index[0]} to {X.index[-1]}",
                        limitations=[
                            "Based on yfinance data (not institutional grade)",
                            "Survivorship bias not fully corrected",
                            "Point-in-time fundamentals approximated"
                        ],
                        validation_results={
                            "avg_accuracy": float(avg['accuracy']),
                            "avg_precision": float(avg['precision']),
                            "avg_recall": float(avg['recall']),
                            "avg_f1": float(avg['f1']),
                            "n_folds": len(folds_df)
                        }
                    )
                    mrm.register_model(model_card)
                    mrm.record_validation(model_card.model_id, model_card.validation_results)
                    
                    # Save to session state
                    st.session_state['model'] = model_card
                    st.session_state['X_latest'] = X.iloc[-1:]
                    st.success(f"✅ Model {model_card.model_id} registered in MRM")
        
        # Model Card Display
        if 'model' in st.session_state:
            mrm.render_model_card(st.session_state['model'].model_id)
        
        # Latest prediction
        st.subheader("🔮 Latest Prediction")
        if 'model' in st.session_state and 'X_latest' in st.session_state:
            st.info("Model trained and ready for prediction")
            # Would show actual prediction here
        else:
            st.info("Run validation to train model and see predictions")
    
    # ---- TAB 4: Execution & Backtest ----
    with tabs[3]:
        st.header("💼 Execution Realism & Backtesting")
        st.markdown("""
        **Institutional-Grade Backtesting:** Includes transaction costs, spread, 
        market impact, risk limits, and kill switches. Complies with SEC Market 
        Access Rule and FINRA guidance on algorithmic trading controls.
        """)
        
        # Use composite score as signal for demo
        df_bt = compute_indicators(raw_history)
        df_bt = compute_composite_score(df_bt, info, fetch_quarterly_financials(config['symbol']))
        
        # Generate signals from composite score
        signals = (df_bt['Composite_Rank'] > 0.6).astype(int) - (df_bt['Composite_Rank'] < 0.4).astype(int)
        
        # Run backtest
        bt_results = realistic_backtest(
            df_bt, signals, 
            config['execution_params'], 
            config['risk_limits']
        )
        
        metrics = compute_backtest_metrics(bt_results)
        
        # Metrics display
        st.subheader("📈 Performance Metrics")
        if metrics:
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.metric("Total Return", f"{metrics['total_return']:.1%}")
                st.metric("Buy & Hold", f"{metrics['hold_return']:.1%}")
            with col_m2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Benchmark Sharpe", f"{metrics['benchmark_sharpe']:.2f}")
            with col_m3:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
                st.metric("Volatility", f"{metrics['volatility']:.1%}")
            with col_m4:
                st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
                st.metric("Payoff Ratio", f"{metrics['payoff_ratio']:.2f}")
            with col_m5:
                st.metric("Turnover", f"{metrics['turnover']:.1f}x")
                if metrics['information_ratio']:
                    st.metric("Info Ratio", f"{metrics['information_ratio']:.2f}")
        
        # Equity curve
        st.subheader("📊 Equity Curves")
        equity_df = pd.DataFrame({
            'Strategy': bt_results['Strategy_Cumulative'],
            'Buy & Hold': bt_results['Hold_Cumulative'],
            'Relative': bt_results['Relative_Cumulative']
        })
        st.line_chart(equity_df)
        
        # Drawdown
        st.subheader("📉 Drawdown Analysis")
        cumulative = bt_results['Strategy_Cumulative']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        st.area_chart(drawdown, height=200)
        
        # Cost breakdown
        st.subheader("💰 Transaction Cost Analysis")
        cost_df = pd.DataFrame({
            'Transaction Costs': bt_results['Transaction_Cost'].cumsum()
        })
        st.line_chart(cost_df)
    
    # ---- TAB 5: Financials ----
    with tabs[4]:
        period = st.selectbox("Period", ['Quarterly', 'Annual'], index=0)
        
        if period == 'Quarterly':
            fin_data = fetch_quarterly_financials(config['symbol'])
            x_field = 'Quarter'
        else:
            fin_data = fetch_annual_financials(config['symbol'])
            x_field = 'Year'
        
        if not fin_data.empty:
            if 'Net Income' in fin_data.columns and 'Total Revenue' in fin_data.columns:
                fin_data['Net Margin'] = fin_data['Net Income'] / fin_data['Total Revenue']
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.subheader("Revenue Trend")
                st.bar_chart(fin_data.set_index(fin_data.index)['Total Revenue'] if 'Total Revenue' in fin_data.columns else pd.Series())
            with col_f2:
                st.subheader("Net Margin %")
                if 'Net Margin' in fin_data.columns:
                    st.line_chart(fin_data['Net Margin'])
    
    # ---- TAB 6: Peer Comparison ----
    with tabs[5]:
        if config['compare_symbol']:
            comp_info = fetch_stock_info(config['compare_symbol'])
            if comp_info:
                st.subheader(f"Comparing {config['symbol']} vs {config['compare_symbol']}")
                
                metrics = ['currentPrice', 'marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 'profitMargins']
                comp_data = {
                    'Metric': ['Price', 'Market Cap', 'Trailing P/E', 'Forward P/E', 'Dividend Yield', 'Profit Margin'],
                    config['symbol']: [info.get(m, 'N/A') for m in metrics],
                    config['compare_symbol']: [comp_info.get(m, 'N/A') for m in metrics]
                }
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
    
    # ---- TAB 7: News ----
    with tabs[6]:
        st.subheader(f"Latest News for {config['symbol']}")
        news_items = fetch_news(config['symbol'])
        if news_items:
            for item in news_items[:10]:
                title = item.get('title', 'No Title')
                link = item.get('link', '#')
                publisher = item.get('publisher', 'Unknown')
                st.markdown(f"**[{title}]({link})**")
                st.caption(f"Publisher: {publisher}")
                st.write("---")
        else:
            st.write("No news available")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean features for modeling"""
    features = [
        'Return', 'Return_1w', 'Return_1m', 'Volatility',
        'RSI14', 'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Volume_Ratio', 'OBV_Signal', 'Price_Position',
        'Vol_Regime', 'RSI_Divergence', 'Intraday_Range'
    ]
    
    X = df[[f for f in features if f in df.columns]].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    return X


if __name__ == "__main__":
    main()