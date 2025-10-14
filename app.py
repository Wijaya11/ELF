# app.py
"""S
Enhanced Load Forecasting - Pro UI (Strict & Auditable)

- Interactive Scenario Compare (no restart; uses session_state & fast post-process)
- All models visible: Regression (Ridge), RF, XGB, LGB, SARIMA(X), Prophet
- Naive kept only in Validation (baseline)
- Residualized base models with RJPP-aware guard controls
- Auto-calibrated p10-90 bands from raw residuals (RJPP bounding optional in UI)
- RJPP deviation chart uses % units
- Hard visual guards: bleed into history, negatives, MoM spikes, re-anchor gaps
- Model Matrix with include/exclude and live reweighting
"""

import os, io, uuid, zipfile, tempfile, warnings, logging, traceback, hashlib, sys, json, pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple, List, Optional
from contextlib import contextmanager

import numpy as np
import pandas as pd

import streamlit as st
import plotly
import plotly.graph_objects as go
import plotly.express as px

try:
    from scenario_engine import Scenario, ScenarioSet, sum_scenarios, apply_scenarios_to_bands, scenario_set_to_future_exogenous
except ImportError as e:
    st.error(f"Missing scenario_engine module: {e}")
    st.error("Please ensure scenario_engine.py is available in the same directory")
    st.stop()

try:
    import ai_insights as ai_mod
    from ai_insights import TASK_PROMPT as AI_TASK_PROMPT, build_analysis_payload, ollama_analyze
except ImportError as e:
    st.warning(f"AI insights module not available: {e}")
    # Create dummy functions to prevent errors
    AI_TASK_PROMPT = ""
    ai_mod = None
    def build_analysis_payload(*args, **kwargs): return {}
    def ollama_analyze(*args, **kwargs): return "AI analysis not available"

from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge, LinearRegression, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import statsmodels as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Optional models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor  # noqa: F401
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

"""Formatting helpers & glossary (standardized)"""
# --- Formatting helpers & glossary ---
def fmt_pct(x):
    try: return f"{float(x):.2f}%"
    except: return str(x)

def fmt_mw(x):
    try: return f"{float(x):,.0f} MW"
    except: return str(x)

def fmt_gwh(x):
    try: return f"{float(x):,.0f} GWh"
    except: return str(x)

def fmt_month(dt):
    try: return pd.to_datetime(dt).strftime("%b-%Y")
    except: return str(dt)

GLOSSARY = {
  "RMSE": "Root Mean Square Error; penalizes large errors. Lower is better.",
  "MAE": "Mean Absolute Error; average absolute difference.",
  "sMAPE": "Symmetric MAPE (%); scale-free; treats over/under symmetrically.",
  "WAPE": "Weighted APE; more stable when actuals near zero.",
  "Band calibration (p10–p90)": "Share of actuals inside p10–p90 band (expect ~80%). 100% may mean bands too wide.",
  "Clamped months": "Months where forecasts hit bounds/guards to avoid unrealistic jumps.",
  "CAGR": "Compound annual growth rate over the selected horizon.",
  "RJPP compliance": "Share of months within permissible RJPP deviation threshold.",
  "Validation RMSE": "RMSE measured on rolling/holdout windows; summary of fit quality.",
  "Ridge": "Linear with L2 regularization; stable, may underfit non-linearities.",
  "RF": "Random Forest; non-linear patterns, weaker long-range extrapolation.",
  "XGB": "Gradient boosting; captures interactions; watch leakage/overfit.",
  "Prophet": "Additive trend+seasonality; smooth extrapolation; tolerant to gaps.",
  "SARIMAX": "Seasonal ARIMA with exogenous variables; interpretable seasonality handling."
}

# Normalize any encoding artifacts in glossary keys and ensure stable labels
try:
    # Replace any malformed 'Band calibration' entry with a clean label
    for k in list(GLOSSARY.keys()):
        if "Band calibration" in k:
            GLOSSARY.pop(k, None)
    GLOSSARY["Band calibration (p10-p90)"] = (
        "Share of actuals inside p10-p90 band (expect ~80%). 100% may mean bands too wide."
    )
except Exception:
    pass

import re
def sanitize_ai(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove entire <think>...</think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # Keep from Summary onward if available (fallback to legacy anchors)
    anchors = ["# Summary", "## Summary", "# Executive Summary", "## Executive Summary"]
    for a in anchors:
        if a in text:
            text = text.split(a, 1)[1]
            text = "# Summary\n" + text
            break
    text = text.replace("## Executive Summary", "## Summary").replace("# Executive Summary", "# Summary")
    text = text.strip()
    return text[:10000]

# AI prompt template (tight, no chain-of-thought)
AI_TASK_PROMPT = (
"Analyze this forecast run end-to-end.\n"
"1) Summary (bullet points).\n"
"2) Key findings with numbers (forecast horizon: p50, p10-90, next-12m energy/avg, yearly peaks, CAGR, RJPP compliance %, worst gap, clamped months).\n"
"3) Risk & guardrail checks (history bleed, negatives, MoM spikes, re-anchor gaps, band calibration).\n"
"4) Scenario impacts (delta MW/GWh by year, peak changes, RJPP alignment).\n"
"5) Recommendations (model tuning priorities, data issues to investigate, operational actions).\n"
"Keep it concise, factual, and structured."
)

DEFAULT_AI_QUERY = AI_TASK_PROMPT

# If ai_insights is available, prefer its TASK_PROMPT to avoid divergence
if 'ai_mod' in globals() and ai_mod is not None:
    try:
        from ai_insights import TASK_PROMPT as _AI_PROMPT_SRC
        AI_TASK_PROMPT = _AI_PROMPT_SRC
    except Exception:
        pass

MODEL_DEFAULT = "Llama3.2:3b"

def ollama_analyze(model_name: str, prompt: str, payload: dict, timeout_s: int = 120) -> str:
    """Call local Ollama server; returns plain response text or empty string on failure."""
    try:
        import json, urllib.request
        body = json.dumps({
            "model": model_name,
            "prompt": f"{prompt}\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048,
                "num_predict": 600,
                "stop": ["</think>"]
            },
            "keep_alive": "30m"
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except Exception as e:
        return f"AI call failed: {e}"

# If ai_insights is available, rebind to its richer implementation (CLI + HTTP + fallback)
if 'ai_mod' in globals() and ai_mod is not None:
    try:
        from ai_insights import ollama_analyze as _ai_ollama
        ollama_analyze = _ai_ollama  # type: ignore
    except Exception:
        pass

def _find_glossary_key(label: str) -> Optional[str]:
    # Try exact match, then substring heuristics
    if label in GLOSSARY:
        return label
    lower = label.lower()
    for k in GLOSSARY:
        if k.lower() in lower or lower in k.lower():
            return k
    # fallback common mappings
    if "next-12" in lower or "next 12" in lower:
        return "CAGR"
    if "rmse" in lower:
        return "RMSE"
    if "wape" in lower:
        return "WAPE"
    if "smape" in lower:
        return "sMAPE"
    return None

def show_metric(container, label, value, delta=None, key=None, explain_on: bool = False):
    """Wrapper around Streamlit `metric` that adds glossary help and optional explanation.

    Some Streamlit versions don't accept a `key` argument on `metric`; avoid passing it
    to maintain compatibility. The `key` parameter is kept in the signature for callers
    that may supply it, but it won't be forwarded to `container.metric`.
    """
    help_key = _find_glossary_key(label)
    help_text = GLOSSARY.get(help_key) if help_key else None
    # `container.metric` may not accept `key` in older/newer Streamlit versions; do not forward it
    if delta is not None:
        container.metric(label, value, delta=delta, help=help_text)
    else:
        container.metric(label, value, help=help_text)
    if explain_on and help_text:
        container.caption(f"Why it matters: {help_text}")


def build_column_config(df: pd.DataFrame):
    """Return a Streamlit column_config mapping using GLOSSARY where available."""
    try:
        config = {}
        for col in df.columns:
            key = _find_glossary_key(col)
            desc = GLOSSARY.get(key) if key else None
            if desc:
                config[col] = st.column_config.Column(col, help=desc) if hasattr(st, 'column_config') else st.column_config
                # Note: st.column_config.Column may not exist on older Streamlit; fallback below handled by Streamlit
        return config
    except Exception:
        return {}

# ---------- Constants ----------
HISTORY_CUTOFF = pd.Timestamp("2025-08-31")
FORECAST_START = pd.Timestamp("2025-09-01")
FORECAST_END   = pd.Timestamp("2041-12-01")

NAIVE_TRAIN_CUTOFF = pd.Timestamp("2023-12-31")
NAIVE_HOLDOUT_START = pd.Timestamp("2024-01-01")
NAIVE_HOLDOUT_END   = pd.Timestamp("2025-08-01")

DEFAULT_VAL_MONTHS = 12
DEFAULT_ROLLING_K  = 2
DEFAULT_RANDOM_STATE = 42

DEFAULT_MAX_DEV = 0.04         # +/-4% clamp vs RJPP
BAND_MULT       = 1.5          # p10/p90 clamp +/- (1.5 * max_dev)
SOFT_MODEL_CLIP_MULT = 1.5     # soft-clip raw model vs RJPP
REANCHOR_BLEND  = 0.20         # blend last actual into first step

MODEL_SLOTS = ["Ridge","RF","XGB","LGB","SARIMAX","Prophet"]

# ---------- PRESETS (theory-sound defaults) ----------
PRESETS = {
    # RJPP governance (weaker magnet, annual-only anchor)
    "annual_anchor_tolerance": 0.005,  # tighten to <=0.5% at annual level
    "horizon_max_dev": {
        "h1_3": 0.04,   # 4% for h=1-3
        "h4_6": 0.04,   # 4% for h=4-6
        "h7_12": 0.04,  # 4% for h=7-12
        "h13p": 0.04,   # 4% for h>=13
    },

    # Smoothing & seasonality
    "smoothing_alpha": 0.0,           # disable unconditional EWM smoothing
    "seasonal_imprint_beta_default": 0.30,  # baseline seasonal imprint weight (30%) when seasonality is strong

    # Stability guards
    "non_negativity": True,
    "mom_cap": {
        "h1_12": 0.20,   # max 20% absolute month-over-month change for h<=12
        "h13p": 0.25,    # max 25% for h>12
    },

    # Prediction bands (empirical)
    "band_target_coverage": 0.80,     # target P10–P90 ≈ 80%
    "band_k_bounds": (0.6, 1.8, 0.05),  # (min, max, step) for band_k search
    "clamp_bands_to_rjpp": False,     # do NOT clamp bands to RJPP by default

    # Residual noise (realism)
    "add_noise": True,
    "noise_scale": 1.0,               # 1.0 by default; acceptable range 1.0–1.1

    # Ensemble
    "use_meta_learner": True,         # fit Ridge/ElasticNet on val preds
    "ensemble_selection": "auto",     # choose by lower validation RMSE ("auto", "weighted", "meta")
}

SCENARIO_FORM_DEFAULT = {
    "name": "",
    "mw": 0.0,
    "profile_type": "step",
    "start": "2027-01-01",
    "end": None,
    "peak": None,
    "plateau_end": None,
    "retire": None,
    "dependency": None,
    "priority": None,
    "probability": None,
    "active": True,
}

def _normalize_month_str(value):
    if value in (None, "", "None"):
        return None
    try:
        return pd.Timestamp(value).to_period("M").to_timestamp("MS").strftime("%Y-%m-%d")
    except Exception:
        return None


def _scenario_form_from_dict(data):
    merged = SCENARIO_FORM_DEFAULT.copy()
    if data:
        for key in merged:
            if key in data:
                merged[key] = data[key]
    return merged


def _scenario_clean_dict(raw):
    clean = SCENARIO_FORM_DEFAULT.copy()
    if raw:
        for key in clean:
            if key in raw and raw[key] is not None:
                clean[key] = raw[key]
    clean["name"] = str(clean.get("name") or "").strip()
    clean["profile_type"] = (clean.get("profile_type") or "step").lower()
    clean["mw"] = float(clean.get("mw") or 0.0)
    clean["start"] = _normalize_month_str(clean.get("start") or SCENARIO_FORM_DEFAULT["start"])
    for field in ("end", "peak", "plateau_end", "retire"):
        clean[field] = _normalize_month_str(clean.get(field))
    dependency = clean.get("dependency")
    clean["dependency"] = dependency.strip() if isinstance(dependency, str) and dependency.strip() else None
    priority = (clean.get("priority") or "").lower()
    clean["priority"] = priority if priority in ("low", "medium", "high") else None
    probability = clean.get("probability")
    if probability in ("", None):
        clean["probability"] = None
    else:
        try:
            clean["probability"] = float(probability)
        except Exception:
            clean["probability"] = None
    # Active flag (default True). If probability is explicitly 0, default to inactive unless user overrode.
    active_flag = raw.get("active") if isinstance(raw, dict) else True
    if active_flag is None:
        active_flag = True
    try:
        active_flag = bool(active_flag)
    except Exception:
        active_flag = True
    if clean.get("probability") == 0 and "active" not in raw:
        active_flag = False
    clean["active"] = active_flag
    return clean

def _calc_cagr_series(series):
    if series is None:
        return None
    series = pd.Series(series).dropna()
    if len(series) < 2:
        return None
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    if first <= 0 or last <= 0:
        return None
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return None
    return (last / first) ** (1.0 / years) - 1.0


def _trend_label_from_cagr(cagr):
    if cagr is None or not np.isfinite(cagr):
        return "Trend: n/a"
    pct = cagr * 100.0
    if abs(pct) < 0.1:
        return f"Trend: Flat ({pct:+.2f}%/yr)"
    if pct > 0:
        return f"Trend: Rising at {pct:.2f}%/yr"
    return f"Trend: Falling at {abs(pct):.2f}%/yr"

def _compute_rjpp_compliance(series: Optional[pd.Series], fc_df: pd.DataFrame, max_dev: Optional[float]) -> Optional[float]:
    if series is None or fc_df is None or "RJPP" not in fc_df.columns or max_dev is None:
        return None
    rjpp_vals = fc_df["RJPP"].reindex(series.index).dropna()
    if rjpp_vals.empty:
        return None
    aligned = pd.concat([series.reindex(rjpp_vals.index), rjpp_vals], axis=1).dropna()
    if aligned.empty:
        return None
    deviation = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs() / aligned.iloc[:, 1]
    return float((deviation <= max_dev).mean() * 100)

# ----- Helpers for session state -----
def _save_file(uploaded, prefix="file_"):
    if uploaded is None: return None
    path = os.path.join(tempfile.gettempdir(), f"{prefix}{uuid.uuid4().hex}.csv")
    with open(path,"wb") as f: f.write(uploaded.getvalue())
    return path

# ---------- Theme ----------
PALETTE = {
    "ensemble": "#F5BF2C",  # gold
    "rjpp":     "#8E8E93",  # gray
    "history":  "#4FB5C1",  # teal
    "good":     "#10B981",  # green
    "warn":     "#F59E0B",  # amber
    "bad":      "#EF4444",  # red
    "scenario": "#D97706",  # amber-orange for adjusted
    "planned":  "#6366F1",  # indigo for overlay
}

PLOTLY_BASE_LAYOUT = dict(
    font=dict(size=13),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis=dict(showspikes=True, spikethickness=1),
)

# ---------- Logging capture ----------
class UIBuffer(logging.Handler):
    def __init__(self):
        super().__init__()
        self.buffer: List[str] = []
    def emit(self, record):
        self.buffer.append(self.format(record))
ui_log_handler = UIBuffer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast")
logger.setLevel(logging.INFO)
# Guard against duplicate handlers on Streamlit reruns
if ui_log_handler not in logger.handlers:
    logger.addHandler(ui_log_handler)

# ---------- Utilities ----------
def _to_ms(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.freq is None:
        df = df.resample("MS").mean(numeric_only=True)
    else:
        df = df.asfreq("MS")
    return df.sort_index()

def _rmse(y, yhat): return float(np.sqrt(mean_squared_error(y, yhat)))
def _wape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    # Add small epsilon to avoid division by zero/near-zero explosions
    denom = float(np.sum(np.abs(y)))
    denom = max(denom, 1e-6)
    return float(np.sum(np.abs(y - yhat)) / denom * 100)
def _smape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = np.abs(y) + np.abs(yhat)
    denom[denom == 0] = np.nan
    return float(np.nanmean(2.0*np.abs(y - yhat)/denom) * 100)

def _limit_mom(arr: np.ndarray, pct: float = 0.18) -> np.ndarray:
    if arr is None or len(arr) == 0:
        return arr
    out = np.array(arr, dtype=float)
    for i in range(1, len(out)):
        prev = out[i-1]
        if not np.isfinite(prev):
            continue
        upper = prev * (1.0 + pct)
        lower = prev * (1.0 - pct)
        out[i] = min(max(out[i], lower), upper)
    return out

def _smooth_center(arr: np.ndarray, window: int = 3) -> np.ndarray:
    if arr is None or len(arr) == 0:
        return arr
    series = pd.Series(arr)
    return series.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


class SeasonalNaiveEstimator:
    """Simple seasonal naive predictor used as a last-resort fallback."""

    def __init__(self, history: pd.Series, season: int = 12):
        hist = pd.Series(history).dropna()
        self.season = max(1, season)
        if len(hist) >= self.season:
            self.pattern = hist.iloc[-self.season :].to_numpy()
        elif len(hist) > 0:
            self.pattern = np.resize(hist.iloc[-1], self.season)
        else:
            self.pattern = np.zeros(self.season)

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        if n <= 0:
            return np.array([])
        return np.resize(self.pattern, n)


class SeasonalRidgeModel:
    """Ridge regression on seasonal/RJPP features for prophet fallback."""

    def __init__(self, feature_cols: List[str], selected_cols: List[str], scaler: RobustScaler, model: Ridge):
        self.feature_cols = feature_cols
        self.selected_cols = selected_cols
        self.indices = [feature_cols.index(c) for c in selected_cols]
        self.scaler = scaler
        self.model = model

    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if hasattr(X, "columns"):
            missing = [c for c in self.selected_cols if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required features for seasonal ridge: {missing}")
            subset = X[self.selected_cols].values
        elif arr.shape[1] == len(self.feature_cols):
            subset = arr[:, self.indices]
        elif arr.shape[1] == len(self.selected_cols):
            subset = arr
        else:
            raise ValueError("Predict called with wrong feature shape")
        subset = self.scaler.transform(subset)
        return self.model.predict(subset)

# ---------- Helper functions for self-choosing ensemble ----------
def _horizon_bucket(h: int) -> str:
    """Map forecast horizon h to a bucket label."""
    if h <= 3:
        return "h1_3"
    if h <= 6:
        return "h4_6"
    if h <= 12:
        return "h7_12"
    return "h13p"

def _max_dev_for_horizon(h: int) -> float:
    """Return max deviation % for a given horizon from PRESETS."""
    bucket = _horizon_bucket(h)
    return PRESETS["horizon_max_dev"].get(bucket, 0.12)

def _mom_cap_for_horizon(h: int) -> float:
    """Return MoM cap for a given horizon from PRESETS."""
    bucket = "h1_12" if h <= 12 else "h13p"
    return PRESETS["mom_cap"].get(bucket, 0.20)

def _is_fitted_estimator(model) -> bool:
    # Extended to cover linear models & generic sklearn fitted flags
    return any([
        hasattr(model, "booster_"),
        hasattr(model, "_Booster"),
        hasattr(model, "n_estimators_"),
        hasattr(model, "feature_importances_"),
        hasattr(model, "coef_"),
        hasattr(model, "intercept_"),
        hasattr(model, "n_features_in_"),
    ])

def _month_start(d: datetime) -> datetime:
    return datetime(d.year, d.month, 1)

def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def _file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _stable_json_dumps(obj: Any) -> str:
    """JSON stringify with stable ordering and numpy/pandas fallback."""
    def _default(o: Any):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, set):
            return sorted(list(o))
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="ignore")
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default)


def _training_signature(load_path: str, rjpp_path: Optional[str], init_params: Dict[str, Any], runtime_params: Dict[str, Any]) -> str:
    parts = [
        _file_md5(load_path),
        _file_md5(rjpp_path) if rjpp_path else "",
        _stable_json_dumps(init_params),
        _stable_json_dumps(runtime_params),
    ]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False)
def _train_forecaster_cached(load_path: str, rjpp_path: Optional[str], init_params: Dict[str, Any], runtime_params: Dict[str, Any], signature: str) -> bytes:
    """Cache trained forecaster keyed by data signature to avoid repeat training."""
    logger.info(f"Cache miss – training forecaster for signature {signature}")
    forecaster = EnhancedLoadForecaster(**init_params)
    forecaster.load(load_path, rjpp_path)
    for attr, value in runtime_params.items():
        setattr(forecaster, attr, value)
    forecaster.train()
    return pickle.dumps(forecaster)

@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, sig: str) -> pd.DataFrame:
    # sig is content hash to invalidate cache on file changes
    return pd.read_csv(path)


def _basic_data_checks(df: pd.DataFrame) -> Dict[str, int]:
    """Compute core data-quality counts for Average_Load series."""
    if df is None or df.empty or "Average_Load" not in df.columns:
        return {"negatives": 0, "missing": 0, "dupe_dates": 0, "near_zero": 0}
    series = df["Average_Load"].astype(float)
    negatives = int((series < 0).sum())
    missing = int(series.isna().sum())
    dupe_dates = int(df.index.duplicated().sum())
    near_zero = int((series.abs() < 1e-6).sum())
    return {
        "negatives": negatives,
        "missing": missing,
        "dupe_dates": dupe_dates,
        "near_zero": near_zero,
    }

@contextmanager
def temp_params(obj, **kwargs):
    old = {k: getattr(obj, k) for k in kwargs}
    try:
        for k, v in kwargs.items(): setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items(): setattr(obj, k, v)

# ---------- Forecaster ----------
@dataclass
class EnhancedLoadForecaster:
    max_dev: float = DEFAULT_MAX_DEV
    val_months: int = DEFAULT_VAL_MONTHS
    random_state: int = DEFAULT_RANDOM_STATE
    rolling_k: int = DEFAULT_ROLLING_K
    use_prophet: bool = PROPHET_AVAILABLE
    use_xgb: bool = XGB_AVAILABLE
    use_lgb: bool = LGB_AVAILABLE

    residualize: bool = True
    apply_guards_to: str = "ensemble"
    stabilize_output: bool = False
    guard_mom_cap: Optional[float] = None
    guard_smoothing_window: int = 1
    mom_limit_quantile: float = 0.90
    # Seasonal/anchoring controls (now from PRESETS)
    seasonal_imprint_beta: float = PRESETS["seasonal_imprint_beta_default"]
    annual_anchor_target_pct: float = PRESETS["annual_anchor_tolerance"]

    add_noise: bool = PRESETS["add_noise"]
    use_rjpp: bool = True
    ensemble_override: str | None = None  # "weighted" | "meta" | None
    noise_scale: float = PRESETS["noise_scale"]

    def __post_init__(self):
        self.scaler = RobustScaler()
        self.models: Dict[str, object] = {}
        self.model_status: Dict[str, str] = {}
        self.model_errors: Dict[str, str] = {}
        self.weights: Dict[str, float] = {}
        self.residuals = None
        self.residuals_by_moy: Dict[int, np.ndarray] = {}
        self.feature_cols: List[str] = []
        self.feature_cols_master: List[str] = []
        self.model_feature_cols: Dict[str, List[str]] = {}
        self.model_scalers: Dict[str, Optional[RobustScaler]] = {}
        self.model_imputers: Dict[str, Optional[pd.Series]] = {}
        self.model_output_mode: Dict[str, str] = {}
        self.latest_raw_forecast: Optional[pd.DataFrame] = None
        self.validation_results: Dict[str, Dict[str, float]] = {}
        self.rjpp_alignment_factors: Dict[int, float] = {}
        self.val_preds: Dict[str, pd.Series | np.ndarray] = {}
        self.val_index = None
        self.in_sample_pred: pd.DataFrame | None = None
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.meta_model = None
        self.meta_models_order: List[str] = []
        self.ensemble_choice = "weighted"
        self.val_df = None
        self.band_k = 1.0  # auto-calibrated band scale on validation
        self.validation_windows: Dict[str, pd.DataFrame] = {}
        self.slot_info: Dict[str, Dict[str, str]] = {}
        self.model_impl: Dict[str, str] = {}
        self.linear_model_keys: set[str] = set()
        self.raw_model_keys: set[str] = set()
        self.non_iterative_keys: set[str] = set()
        self.residual_model_keys: set[str] = set()
        self.model_calibrations: Dict[str, Dict[str, float]] = {}
        self.mom_limit_pct: float = 0.18
        self.smoothing_window: int = 1
        self._residualize_effective: bool = False
        # New: per-bucket weights for self-choosing ensemble
        self.model_weights_by_bucket: Dict[str, Dict[str, float]] = {}
        self.meta_cols: List[str] = []
        self.seasonal_strength: float = 0.0  # computed from STL if available
        self.val_rmse_weighted: float = np.inf
        self.val_rmse_meta: float = np.inf
        self.validation_bucket_metrics: Dict[str, pd.DataFrame] = {}
        self.data_quality_summary: Dict[str, int] = {}

    def _register_model(self, slot: str, estimator, impl_label: str, status: str, model_type: str):
        # clear previous type assignment
        self.linear_model_keys.discard(slot)
        self.raw_model_keys.discard(slot)
        self.non_iterative_keys.discard(slot)

        if estimator is not None:
            self.models[slot] = estimator
            self.model_status[slot] = status
            self.model_errors.pop(slot, None)
            self.slot_info[slot] = {"status": status, "impl": impl_label}
            self.model_impl[slot] = impl_label
        else:
            self.models.pop(slot, None)
            self.model_status[slot] = "unavailable"
            self.slot_info[slot] = {"status": "unavailable", "impl": impl_label}
            self.model_impl[slot] = impl_label

        if model_type == "linear":
            self.linear_model_keys.add(slot)
        elif model_type == "raw":
            self.raw_model_keys.add(slot)
        elif model_type == "non_iterative":
            self.non_iterative_keys.add(slot)
    def _prepare_xy(self, frame: pd.DataFrame, features: List[str], target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if not features:
            return pd.DataFrame(columns=[]), pd.Series(dtype=float)
        df = pd.concat([frame[features], target], axis=1, join='inner').dropna()
        if df.empty:
            return pd.DataFrame(columns=features), pd.Series(dtype=float)
        X = df[features]
        y = df[target.name]
        return X, y

    def _make_tscv(self, n_samples: int) -> Optional[TimeSeriesSplit]:
        if n_samples < 12:
            return None
        splits = max(2, min(5, n_samples // 12))
        if splits >= n_samples:
            splits = max(2, min(5, n_samples // 6))
        if splits >= n_samples:
            return None
        return TimeSeriesSplit(n_splits=splits)

    def _make_decay_weights(self, index: pd.Index) -> pd.Series:
        if index is None or len(index) == 0:
            return pd.Series(dtype=float)
        idx = pd.Index(index)
        try:
            months_back = (idx.max().to_period("M") - idx.to_period("M")).astype(int)
        except Exception:
            months_back = np.arange(len(idx))[::-1]
        weights = np.power(0.5, months_back / 36.0)
        return pd.Series(weights, index=idx, dtype=float)

    def _apply_series_guards(self, values: np.ndarray, rvals: Optional[np.ndarray]) -> np.ndarray:
        arr = np.array(values, dtype=float) if values is not None else np.array([], dtype=float)
        if arr.size == 0:
            return arr
        if self.stabilize_output:
            pct = self.guard_mom_cap if self.guard_mom_cap is not None else self.mom_limit_pct
            if pct is not None:
                arr = _limit_mom(arr, pct=float(pct))
            window = max(1, int(self.guard_smoothing_window))
            if window > 1:
                arr = _smooth_center(arr, window=window)
        arr = np.maximum(arr, 0.0)
        if rvals is not None and self.use_rjpp:
            arr = np.minimum(arr, rvals * (1 + self.max_dev))
            arr = np.maximum(arr, rvals * (1 - self.max_dev))
        return arr

    # ---------- Load ----------
    def load(self, load_path, rjpp_path=None):
        logger.info("Loading data...")
        df = _to_ms(_read_csv_cached(load_path, _file_md5(load_path))).loc[:HISTORY_CUTOFF]
        if "Average_Load" not in df:
            raise ValueError("Historical CSV must contain 'Average_Load'.")
        # Coerce to numeric and clean
        df["Average_Load"] = pd.to_numeric(df["Average_Load"], errors="coerce")
        df["Average_Load"] = df["Average_Load"].interpolate().clip(lower=0)
        self.load_df = df[["Average_Load"]]
        self.data_quality_summary = _basic_data_checks(self.load_df)

        if rjpp_path:
            rj = _to_ms(_read_csv_cached(rjpp_path, _file_md5(rjpp_path)))
            if "RJPP" not in rj:
                num = rj.select_dtypes("number").columns[0]
                rj = rj.rename(columns={num: "RJPP"})
            rj["RJPP"] = pd.to_numeric(rj["RJPP"], errors="coerce")
            rj["RJPP"] = rj["RJPP"].interpolate().clip(lower=0)
            self.rjpp = rj[["RJPP"]]
        else:
            self.rjpp = None
        # Respect toggle to disable RJPP in modeling
        if not self.use_rjpp:
            self.rjpp = None
        self._update_mom_limit_pct()

    def _update_mom_limit_pct(self):
        series = self.load_df["Average_Load"].dropna() if hasattr(self, "load_df") else pd.Series(dtype=float)
        if series.empty:
            self.mom_limit_pct = 0.18
            return
        mom = series.pct_change().abs().dropna()
        if len(mom):
            q = getattr(self, "mom_limit_quantile", 0.90)
            pct = float(np.nanquantile(mom, q))
            pct = float(np.clip(pct, 0.05, 0.25))
            if self.guard_mom_cap is not None:
                self.mom_limit_pct = float(self.guard_mom_cap)
            else:
                self.mom_limit_pct = pct
        else:
            self.mom_limit_pct = 0.18

    # ---------- Features ----------
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        f["year"] = f.index.year
        f["month"] = f.index.month
        f["month_sin"] = np.sin(2*np.pi*f["month"]/12)
        f["month_cos"] = np.cos(2*np.pi*f["month"]/12)
        f["quarter"] = f.index.quarter
        f["quarter_sin"] = np.sin(2*np.pi*f["quarter"]/4)
        f["quarter_cos"] = np.cos(2*np.pi*f["quarter"]/4)
        for lag in [1,12]:
            f[f"lag{lag}"] = f["Average_Load"].shift(lag)
        f["roll3_mean"] = f["Average_Load"].rolling(window=3, min_periods=1).mean().shift(1)
        f["roll12_mean"] = f["Average_Load"].rolling(window=12, min_periods=1).mean().shift(1)
        f["mom_pct"] = f["Average_Load"].pct_change().shift(1)
        f["yoy_pct"] = f["Average_Load"].pct_change(12).shift(1)
        base = f["Average_Load"].copy()
        base.loc[base.index > NAIVE_TRAIN_CUTOFF] = np.nan
        f["naive"] = base.shift(12)
        if self.rjpp is not None:
            f = f.join(self.rjpp, how="left")
            f["RJPP"] = f["RJPP"].ffill().bfill()
            f["RJPP_lag1"] = f["RJPP"].shift(1)
            f["rjpp_gap"] = f["RJPP_lag1"] - f["Average_Load"].shift(1)
            f["rjpp_yoy_pct"] = f["RJPP"].pct_change(12).shift(1)
            f["rjpp_mom_pct"] = f["RJPP"].pct_change().shift(1)
        return f

    # ---------- Naive baseline ----------
    def _naive_forecast_series(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        idx = pd.date_range(start, end, freq="MS")
        hist = self.load_df["Average_Load"]
        # Use the requested forecast start to determine last-12 pattern
        anchor = pd.to_datetime(start)
        last12 = hist.loc[anchor - pd.offsets.MonthBegin(12):anchor - pd.offsets.MonthBegin(1)]
        if len(last12) < 12: last12 = hist.iloc[-12:]
        pattern = last12.values
        vals = [pattern[i % 12] for i in range(len(idx))]
        return pd.Series(vals, index=idx, name="Naive")

    def naive_holdout(self):
        idx = pd.date_range(NAIVE_HOLDOUT_START, NAIVE_HOLDOUT_END, freq="MS")
        y = self.load_df["Average_Load"].reindex(idx)
        naive = self._naive_forecast_series(NAIVE_HOLDOUT_START, NAIVE_HOLDOUT_END)
        mask = y.notna() & naive.notna()
        if mask.any():
            return naive, {
                "MAE": mean_absolute_error(y[mask], naive[mask]),
                "RMSE": _rmse(y[mask], naive[mask]),
                "MAPE": mean_absolute_percentage_error(y[mask], naive[mask])*100,
                "WAPE": _wape(y[mask], naive[mask]),
                "sMAPE": _smape(y[mask], naive[mask]),
            }
        return naive, {}

    # ---------- Train ----------
    def train(self):
        logger.info("Training models...")
        feats = self.features(self.load_df)
        self.feature_cols = [c for c in feats.columns if c != "Average_Load"]
        self.feature_cols_master = self.feature_cols.copy()
        self._residualize_effective = bool(self.residualize and "RJPP" in self.feature_cols)
        if self._residualize_effective:
            drop_cols = {"RJPP", "year", "quarter"}
            self.feature_cols = [c for c in self.feature_cols if c not in drop_cols]
        else:
            self.feature_cols = [c for c in self.feature_cols if c != "RJPP"]
        self.residual_model_keys.clear()

        self.slot_info = {}
        self.model_impl = {}
        self.model_errors = {}
        self.linear_model_keys.clear()
        self.raw_model_keys.clear()
        self.non_iterative_keys.clear()
        self.model_feature_cols.clear()
        self.model_scalers.clear()
        self.model_output_mode.clear()
        self.latest_raw_forecast = None
        self.model_calibrations = {}

        # split into train / val_last
        train, val_last = (feats.iloc[:-self.val_months], feats.iloc[-self.val_months:]) if self.val_months < len(feats) else (feats, pd.DataFrame())
        self.train_df, self.val_df = train, val_last

        X = train[self.feature_cols]
        y_level = train["Average_Load"]
        rjpp_series = train["RJPP"] if "RJPP" in train.columns else pd.Series(index=train.index, dtype=float)
        if self._residualize_effective:
            y = (y_level - rjpp_series).rename("target")
        else:
            y = y_level.rename("target")
        mask = X.notna().all(axis=1) & y.notna()
        if self._residualize_effective:
            mask &= rjpp_series.notna()
        X, y = X.loc[mask], y.loc[mask]
        if self._residualize_effective:
            rjpp_series = rjpp_series.loc[mask]
        X_unscaled = X.copy()
        w_series = self._make_decay_weights(X_unscaled.index)

        self.scaler.fit(X_unscaled)

        val_features = pd.DataFrame()
        val_target = None
        if not val_last.empty:
            val_features = val_last[self.feature_cols].copy()
            vt = val_last["Average_Load"].copy()
            if self._residualize_effective and "RJPP" in val_last.columns:
                vt = vt - val_last["RJPP"]
            val_target = vt.rename("target")

        # Optimized feature selection for residual models with additional engineered features
        residual_feature_whitelist = [
            # Lag features (core temporal patterns)
            "lag1", "lag12", "lag2", "lag3",  # Added lag2, lag3 for short-term patterns
            # Rolling statistics (trend and volatility)
            "roll3_mean", "roll6_mean", "roll12_mean",  # Added roll6_mean for medium-term trends
            "roll3_std", "roll12_std",  # Rolling standard deviations for volatility
            # Growth rates (momentum indicators)
            "mom_pct", "yoy_pct",
            # Seasonal features (cyclical patterns)
            "month_sin", "month_cos", "quarter",
            # RJPP-related features (reference alignment)
            "rjpp_mom_pct", "rjpp_yoy_pct", "rjpp_gap"
        ]
        train_weights = w_series.loc[X_unscaled.index].astype(float).values if len(X_unscaled) else None

        # Ridge/ElasticNet with optimized feature selection and hyperparameters
        ridge_features = [c for c in ["lag1", "lag12", "roll3_mean", "roll12_mean", "month_sin", "month_cos", "rjpp_mom_pct", "rjpp_yoy_pct", "rjpp_gap"] if c in X_unscaled.columns]
        ridge_model = None
        ridge_status = "unavailable"
        ridge_impl = "Unavailable"
        ridge_scaler = None
        if ridge_features and not X_unscaled.empty:
            try:
                ridge_scaler = RobustScaler().fit(X_unscaled[ridge_features])
                X_ridge_scaled = ridge_scaler.transform(X_unscaled[ridge_features])
                cv = self._make_tscv(len(X_unscaled))
                if cv is not None:
                    # Optimized ElasticNetCV with expanded search space
                    enet = ElasticNetCV(
                        l1_ratio=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7],  # Expanded l1_ratio search
                        alphas=np.logspace(-3, 1.5, 15),  # Expanded alpha search range
                        cv=cv,
                        max_iter=30000,  # Increased max iterations for convergence
                        tol=1e-4,  # Added tolerance for better convergence
                        selection='random',  # Use random selection for faster convergence
                        random_state=DEFAULT_RANDOM_STATE,
                        n_jobs=-1  # Parallelize CV
                    )
                    enet.fit(X_ridge_scaled, y.values, sample_weight=train_weights)
                    ridge_model = enet
                    ridge_impl = "ElasticNetCV (optimized)"
                else:
                    ridge_model = Ridge(alpha=1.0, solver='auto').fit(X_ridge_scaled, y.values, sample_weight=train_weights)
                    ridge_impl = "Ridge(alpha=1.0)"
                ridge_status = "ok"
            except Exception as e:
                self.model_errors["Ridge"] = f"Linear model fit failed: {e}"
                try:
                    if ridge_scaler is None:
                        ridge_scaler = RobustScaler().fit(X_unscaled[ridge_features])
                        X_ridge_scaled = ridge_scaler.transform(X_unscaled[ridge_features])
                    ridge_model = LinearRegression().fit(X_ridge_scaled, y.values, sample_weight=train_weights)
                    ridge_impl = "LinearRegression (fallback)"
                    ridge_status = "fallback"
                except Exception as e2:
                    self.model_errors["Ridge"] += f" | Linear fallback failed: {e2}"
                    ridge_model = None
                    ridge_impl = "Unavailable"
                    ridge_status = "unavailable"
                    ridge_scaler = None
        self.model_feature_cols["Ridge"] = ridge_features
        self.model_scalers["Ridge"] = ridge_scaler
        self.model_imputers["Ridge"] = X_unscaled[ridge_features].median(numeric_only=True) if ridge_features else pd.Series(dtype=float)
        self.model_output_mode["Ridge"] = "residual" if self._residualize_effective else "level"
        if ridge_model is not None:
            self._register_model("Ridge", ridge_model, ridge_impl, ridge_status, "linear")
            if self._residualize_effective:
                self.residual_model_keys.add("Ridge")
        else:
            self._register_model("Ridge", None, ridge_impl, "unavailable", "linear")

        # Random Forest (residual mode smoothing) - Optimized hyperparameters
        rf_features = ([c for c in residual_feature_whitelist if c in X_unscaled.columns]
                       if self._residualize_effective else list(X_unscaled.columns))
        rf_model = None
        rf_status = "unavailable"
        rf_impl = "Unavailable"
        if rf_features and not X_unscaled.empty:
            try:
                # Optimized RF hyperparameters for better performance
                rf_model = RandomForestRegressor(
                    n_estimators=1000,  # Increased for better stability
                    max_depth=7,  # Slightly deeper for better pattern capture
                    min_samples_leaf=10,  # Reduced for better fit
                    min_samples_split=20,  # Reduced for better fit
                    max_features=0.65,  # Slightly increased feature subset
                    bootstrap=True,
                    ccp_alpha=2e-4,  # Reduced pruning for better fit
                    random_state=DEFAULT_RANDOM_STATE,
                    n_jobs=-1,
                    warm_start=False,
                    oob_score=False
                ).fit(X_unscaled[rf_features], y.values, sample_weight=train_weights)
                rf_status = "ok"
                rf_impl = "RandomForestRegressor (optimized)"
            except Exception as e:
                self.model_errors["RF"] = f"RF failed: {e}"
                try:
                    # First fallback: Try ExtraTreesRegressor with optimized params
                    rf_model = ExtraTreesRegressor(
                        n_estimators=1000,
                        max_depth=7,
                        min_samples_leaf=10,
                        min_samples_split=20,
                        max_features=0.65,
                        random_state=DEFAULT_RANDOM_STATE,
                        n_jobs=-1,
                        bootstrap=False
                    ).fit(X_unscaled[rf_features], y.values, sample_weight=train_weights)
                    rf_status = "fallback"
                    rf_impl = "ExtraTreesRegressor (fallback)"
                except Exception as e2:
                    self.model_errors["RF"] = self.model_errors.get("RF", "") + f" | ExtraTrees fallback failed: {e2}"
                    # Second fallback: Try GradientBoostingRegressor as last resort
                    try:
                        rf_model = GradientBoostingRegressor(
                            n_estimators=600,
                            learning_rate=0.04,
                            max_depth=4,
                            min_samples_leaf=10,
                            subsample=0.8,
                            random_state=DEFAULT_RANDOM_STATE
                        ).fit(X_unscaled[rf_features], y.values, sample_weight=train_weights)
                        rf_status = "fallback"
                        rf_impl = "GradientBoostingRegressor (2nd fallback for RF)"
                    except Exception as e3:
                        self.model_errors["RF"] = self.model_errors.get("RF", "") + f" | GradientBoosting fallback failed: {e3}"
                        rf_model = None
                        rf_status = "unavailable"
                        rf_impl = "Unavailable"
        self.model_feature_cols["RF"] = rf_features
        self.model_scalers["RF"] = None
        self.model_imputers["RF"] = X_unscaled[rf_features].median(numeric_only=True) if rf_features else pd.Series(dtype=float)
        self.model_output_mode["RF"] = "residual" if self._residualize_effective else "level"
        if rf_model is not None:
            self._register_model("RF", rf_model, rf_impl, rf_status, "raw")
            if self._residualize_effective:
                self.residual_model_keys.add("RF")
        else:
            self._register_model("RF", None, rf_impl, "unavailable", "raw")

        xgb_fallback_error = None
        if self.use_xgb:
            xgb_features = ([c for c in residual_feature_whitelist if c in X_unscaled.columns]
                            if self._residualize_effective else list(X_unscaled.columns))
            xgb_model = None
            xgb_impl = "Unavailable"
            xgb_status = "unavailable"
            fit_kwargs = {"verbose": False}
            if val_target is not None and not val_features.empty and xgb_features:
                Xv = val_features[xgb_features].copy()
                m = Xv.notna().all(axis=1) & val_target.notna()
                Xv = Xv.loc[m]
                yv = val_target.loc[m]
                if len(Xv):
                    fit_kwargs["eval_set"] = [(Xv.values, yv.values)]
                    fit_kwargs["eval_metric"] = "rmse"
                    fit_kwargs["early_stopping_rounds"] = 50
            try:
                if xgb_features and not X_unscaled.empty:
                    # Optimized XGBoost hyperparameters for better performance
                    xgb_model = XGBRegressor(
                        n_estimators=1500,  # Increased for better learning
                        learning_rate=0.025,  # Slightly reduced for stability
                        max_depth=5,  # Increased depth for better pattern capture
                        min_child_weight=6,  # Reduced for better fit
                        subsample=0.75,  # Increased for better generalization
                        colsample_bytree=0.75,  # Increased feature sampling
                        reg_lambda=10.0,  # Slightly reduced L2 regularization
                        reg_alpha=1.5,  # Slightly reduced L1 regularization
                        gamma=0.3,  # Reduced for better split decisions
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=DEFAULT_RANDOM_STATE,
                        verbosity=0,
                        importance_type='gain'
                    ).fit(X_unscaled[xgb_features].values, y.values, sample_weight=train_weights, **fit_kwargs)
                    xgb_impl = "XGBRegressor (optimized)"
                    xgb_status = "ok"
            except Exception as e:
                xgb_fallback_error = e
                self.model_errors["XGB"] = f"XGB failed: {e}"
            if xgb_model is None and xgb_features and not X_unscaled.empty:
                try:
                    # Improved fallback with better hyperparameters
                    gbr_model = GradientBoostingRegressor(
                        n_estimators=1000,  # Increased iterations
                        learning_rate=0.025,  # Slightly reduced for stability
                        max_depth=4,  # Increased depth
                        min_samples_leaf=8,  # Added for better generalization
                        min_samples_split=16,  # Added for better generalization
                        subsample=0.8,
                        max_features=0.7,  # Added feature subsampling
                        random_state=DEFAULT_RANDOM_STATE
                    ).fit(X_unscaled[xgb_features], y, sample_weight=train_weights)
                    xgb_model = gbr_model
                    xgb_impl = "GradientBoostingRegressor (optimized fallback)" if xgb_fallback_error else "GradientBoostingRegressor (optimized)"
                    xgb_status = "fallback" if xgb_fallback_error else "ok"
                except Exception as e2:
                    self.model_errors["XGB"] = self.model_errors.get("XGB", "") + (" | " if xgb_fallback_error else "") + f"GradientBoosting fallback failed: {e2}"
                    xgb_model = None
                    xgb_impl = "Unavailable"
                    xgb_status = "unavailable"
            self.model_feature_cols["XGB"] = xgb_features
            self.model_scalers["XGB"] = None
            self.model_imputers["XGB"] = X_unscaled[xgb_features].median(numeric_only=True) if xgb_features else pd.Series(dtype=float)
            self.model_output_mode["XGB"] = "residual" if self._residualize_effective else "level"
            if xgb_model is not None:
                self._register_model("XGB", xgb_model, xgb_impl, xgb_status, "raw")
                if self._residualize_effective:
                    self.residual_model_keys.add("XGB")
            else:
                self._register_model("XGB", None, xgb_impl, "unavailable", "raw")

        lgb_model = None
        lgb_impl = "Unavailable"
        lgb_status = "unavailable"
        lgb_features = ([c for c in residual_feature_whitelist if c in X_unscaled.columns]
                        if self._residualize_effective else list(X_unscaled.columns))
        self.model_feature_cols["LGB"] = lgb_features
        self.model_scalers["LGB"] = None
        self.model_imputers["LGB"] = X_unscaled[lgb_features].median(numeric_only=True) if lgb_features else pd.Series(dtype=float)
        self.model_output_mode["LGB"] = "residual" if self._residualize_effective else "level"
        if self.use_lgb and lgb_features and not X_unscaled.empty:
            try:
                from lightgbm import LGBMRegressor, early_stopping, log_evaluation
                callbacks = [early_stopping(stopping_rounds=60), log_evaluation(period=0)]
                fit_kwargs = {"callbacks": callbacks}
                if val_target is not None and not val_features.empty:
                    Xv = val_features[lgb_features].copy()
                    m = Xv.notna().all(axis=1) & val_target.notna()
                    Xv = Xv.loc[m]
                    yv = val_target.loc[m]
                    if len(Xv):
                        fit_kwargs["eval_set"] = [(Xv.values, yv.values)]
                # Optimized LightGBM hyperparameters for better performance
                lgb_model = LGBMRegressor(
                    n_estimators=5000,  # Increased for better learning with early stopping
                    learning_rate=0.018,  # Slightly reduced for better generalization
                    num_leaves=40,  # Increased for better expressiveness
                    max_depth=7,  # Increased depth for better pattern capture
                    min_data_in_leaf=35,  # Slightly reduced for better fit
                    feature_fraction=0.8,  # Increased for better feature utilization
                    bagging_fraction=0.8,  # Increased for better generalization
                    bagging_freq=1,
                    lambda_l1=0.08,  # Slightly reduced L1 regularization
                    lambda_l2=8.0,  # Slightly reduced L2 regularization
                    min_gain_to_split=0.0,
                    min_child_weight=0.001,  # Added for better control
                    path_smooth=0.1,  # Added for better generalization
                    random_state=DEFAULT_RANDOM_STATE,
                    verbose=-1,
                    importance_type='gain'
                )
                lgb_model.fit(X_unscaled[lgb_features].values, y.values, sample_weight=train_weights, **fit_kwargs)
                lgb_impl = "LGBMRegressor (optimized)"
                lgb_status = "ok"
            except Exception as e:
                self.model_errors["LGB"] = f"LGB failed: {e}"
                lgb_model = None
        if lgb_model is None and lgb_features and not X_unscaled.empty:
            try:
                # First fallback: HistGradientBoostingRegressor with optimized params
                hgb_model = HistGradientBoostingRegressor(
                    max_depth=7,  # Increased depth
                    learning_rate=0.025,  # Slightly reduced for stability
                    max_iter=1500,  # Increased iterations
                    min_samples_leaf=30,  # Added for better generalization
                    l2_regularization=5.0,  # Added regularization
                    max_leaf_nodes=45,  # Added for better control
                    random_state=DEFAULT_RANDOM_STATE
                )
                hgb_model.fit(X_unscaled[lgb_features], y, sample_weight=train_weights)
                lgb_model = hgb_model
                prev_err = self.model_errors.get("LGB")
                lgb_impl = "HistGradientBoostingRegressor (optimized fallback)" if prev_err else "HistGradientBoostingRegressor (optimized)"
                lgb_status = "fallback" if prev_err else "ok"
            except Exception as e2:
                prev_err = self.model_errors.get("LGB")
                msg = f"HistGradientBoosting fallback failed: {e2}"
                self.model_errors["LGB"] = f"{prev_err + ' | ' if prev_err else ''}{msg}"
        if lgb_model is None and lgb_features and not X_unscaled.empty:
            try:
                # Second fallback: GradientBoostingRegressor with optimized params
                gb_model = GradientBoostingRegressor(
                    n_estimators=1200,  # Increased iterations
                    learning_rate=0.025,  # Slightly reduced for stability
                    max_depth=4,  # Increased depth
                    min_samples_leaf=25,  # Added for better generalization
                    min_samples_split=50,  # Added for better generalization
                    subsample=0.8,
                    max_features=0.75,  # Added feature subsampling
                    random_state=DEFAULT_RANDOM_STATE
                )
                gb_model.fit(X_unscaled[lgb_features], y.values if hasattr(y, 'values') else y, sample_weight=train_weights)
                lgb_model = gb_model
                lgb_impl = "GradientBoostingRegressor (optimized 2nd fallback)"
                lgb_status = "fallback"
            except Exception as e3:
                prev_err = self.model_errors.get("LGB")
                msg = f"GradientBoosting fallback failed: {e3}"
                self.model_errors["LGB"] = f"{prev_err + ' | ' if prev_err else ''}{msg}"
        if lgb_model is not None:
            self._register_model("LGB", lgb_model, lgb_impl, lgb_status, "raw")
            if self._residualize_effective:
                self.residual_model_keys.add("LGB")
        else:
            seasonal_history = self.load_df["Average_Load"].copy()
            fallback_model = SeasonalNaiveEstimator(seasonal_history, season=12)
            self._register_model("LGB", fallback_model, "Seasonal Naive (fallback for LGB)", "unavailable_fallback_naive", "raw")
            existing = self.model_errors.get("LGB", "").strip()
            prefix = f"{existing} | " if existing else ""
        # SARIMAX (residual-first) with optimized configuration
        self.model_feature_cols["SARIMAX"] = []
        self.model_scalers["SARIMAX"] = None
        self.model_imputers["SARIMAX"] = None
        sarimax_error_msgs: List[str] = []
        sarimax_model = None
        if self._residualize_effective and "RJPP" in train.columns:
            resid_series = (train["Average_Load"] - train["RJPP"]).dropna()
            if len(resid_series):
                try:
                    # Optimized SARIMAX for residual mode with better specification
                    sarimax_model = SARIMAX(
                        resid_series,
                        order=(1, 0, 1),  # Added MA term for better residual capture
                        seasonal_order=(1, 1, 1, 12),  # Added seasonal AR for better patterns
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        initialization='approximate_diffuse',  # Better initialization
                        concentrate_scale=True  # Improve estimation efficiency
                    ).fit(disp=False, maxiter=200, method='lbfgs')  # Better optimization
                    impl = "SARIMAX-resid(1,0,1)x(1,1,1,12) optimized"
                    self._register_model("SARIMAX", sarimax_model, impl, "ok", "non_iterative")
                    self.model_output_mode["SARIMAX"] = "residual"
                    self.residual_model_keys.add("SARIMAX")
                    logger.info("SARIMAX trained successfully in residual mode")
                except Exception as err:
                    sarimax_error_msgs.append(f"Residual SARIMAX fit failed: {err}")
        if sarimax_model is None:
            try:
                level_series = train["Average_Load"].astype(float).dropna()
                exog = train["RJPP"].astype(float).reindex(level_series.index) if "RJPP" in train.columns else None
                # Optimized SARIMAX for level mode with better specification
                sarimax_model = SARIMAX(
                    level_series,
                    order=(1, 1, 1),  # Added MA term for better fit
                    seasonal_order=(1, 1, 1, 12),
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization='approximate_diffuse',  # Better initialization
                    concentrate_scale=True  # Improve estimation efficiency
                ).fit(disp=False, maxiter=200, method='lbfgs')  # Better optimization
                impl = "SARIMAX(1,1,1)x(1,1,1,12) optimized" + (" with RJPP" if exog is not None else "")
                self._register_model("SARIMAX", sarimax_model, impl, "ok", "non_iterative")
                self.model_output_mode["SARIMAX"] = "level"
                self.residual_model_keys.discard("SARIMAX")
                logger.info(f"SARIMAX trained successfully in level mode{' with RJPP exogenous' if exog is not None else ''}")
            except Exception as err_level:
                sarimax_error_msgs.append(f"Level SARIMAX fit failed: {err_level}")
                try:
                    ets_series = train["Average_Load"].astype(float).ffill().bfill()
                    ets_model = ExponentialSmoothing(ets_series, trend="add", seasonal="mul", seasonal_periods=12).fit()
                    self._register_model("SARIMAX", ets_model, "ExponentialSmoothing (fallback)", "fallback", "non_iterative")
                    self.model_output_mode["SARIMAX"] = "level"
                    self.residual_model_keys.discard("SARIMAX")
                except Exception as err_ets:
                    sarimax_error_msgs.append(f"ETS fallback failed: {err_ets}")
                    self._register_model("SARIMAX", None, "Unavailable", "unavailable", "non_iterative")
                    self.model_output_mode["SARIMAX"] = "level"
                    self.residual_model_keys.discard("SARIMAX")
        if sarimax_error_msgs:
            self.model_errors["SARIMAX"] = " | ".join(sarimax_error_msgs)
        else:
            self.model_errors.pop("SARIMAX", None)

        # Prophet (residual seasonal)
        self.model_feature_cols["Prophet"] = []
        self.model_scalers["Prophet"] = None
        self.model_imputers["Prophet"] = None
        prophet_errors: List[str] = []
        prophet_trained = False
        if self.use_prophet and PROPHET_AVAILABLE:
            use_residual_prophet = self._residualize_effective and "RJPP" in train.columns
            try:
                try:
                    # Optimized Prophet configuration for better forecasting
                    p = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode="additive",
                        n_changepoints=5,  # Increased from 0 for better trend capture
                        changepoint_prior_scale=0.05,  # Increased for more flexibility
                        seasonality_prior_scale=12.0,  # Increased for stronger seasonality
                        growth="linear" if not use_residual_prophet else "flat",  # Linear growth for level, flat for residual
                        changepoint_range=0.8,  # Focus changepoints on first 80% of data
                    )
                except Exception:
                    p = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode="additive",
                        n_changepoints=5,
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=12.0
                    )
                # Optimized monthly seasonality with higher Fourier order for better fit
                p.add_seasonality(name="monthly", period=12.0, fourier_order=8)
                pdf = train.reset_index().rename(columns={"Date": "ds", "Average_Load": "y"})
                if use_residual_prophet:
                    pdf["y"] = pdf["y"] - train["RJPP"].values
                elif "RJPP" in train.columns:
                    p.add_regressor("RJPP")
                    pdf["RJPP"] = train["RJPP"].values
                pdf = pdf.dropna()
                if not pdf.empty:
                    p.fit(pdf)
                    self._register_model("Prophet", p, "Prophet (optimized)", "ok", "non_iterative")
                    if use_residual_prophet:
                        self.model_output_mode["Prophet"] = "residual"
                        self.residual_model_keys.add("Prophet")
                    else:
                        self.model_output_mode["Prophet"] = "level"
                        self.residual_model_keys.discard("Prophet")
                    prophet_trained = True
                    logger.info(f"Prophet trained successfully in {'residual' if use_residual_prophet else 'level'} mode")
                else:
                    prophet_errors.append("Prophet training data empty")
            except Exception as err_prophet:
                prophet_errors.append(f"Prophet fit failed: {err_prophet}")
        if not prophet_trained:
            self._register_model("Prophet", None, "Unavailable", "unavailable", "non_iterative")
            self.model_output_mode["Prophet"] = "level"
            self.residual_model_keys.discard("Prophet")
            if prophet_errors:
                self.model_errors["Prophet"] = " | ".join(prophet_errors)
            else:
                self.model_errors.pop("Prophet", None)

# Residual diagnostics feeding band calibration
        resid = pd.Series(dtype=float)
        if "Ridge" in self.models and self.models["Ridge"] is not None:
            ridge_cols = self.model_feature_cols.get("Ridge", list(X_unscaled.columns))
            X_ridge = X_unscaled[ridge_cols] if ridge_cols else X_unscaled
            try:
                scaler = self.model_scalers.get("Ridge")
                preds = self.models["Ridge"].predict(scaler.transform(X_ridge) if scaler is not None else X_ridge.values)
                preds_series = pd.Series(preds, index=X_ridge.index)
                if self._residualize_effective and "RJPP" in train.columns:
                    preds_series = preds_series + train.loc[preds_series.index, "RJPP"]
                resid = (y_level.loc[preds_series.index] - preds_series).dropna()
            except Exception as e:
                logger.warning(f"Residual computation fallback (Ridge) due to: {e}")
        if resid.empty:
            resid = y_level.diff().dropna()
        self.residuals = resid
        self.residuals_by_moy = {m: resid.loc[resid.index.month == m].values if (resid.index.month == m).any() else resid.values for m in range(1, 13)}

        # Seasonal strength informs seasonal imprint guard
        self.seasonal_strength = self._compute_seasonal_strength()
        logger.info(f"Seasonal strength: {self.seasonal_strength:.3f}")

        # Rolling validation with time-decay weighting
        self.validation_results, self.val_preds, self.val_index, self.validation_windows = self._validate_models_rolling(feats)
        self._calibrate_residual_models()
        self.weights = self._weights_from_validation(self.validation_results)
        self.model_weights_by_bucket = self._compute_per_bucket_weights(self.validation_bucket_metrics)
        logger.info(f"Per-bucket weights computed: {list(self.model_weights_by_bucket.keys())}")

        self.in_sample_pred = self._in_sample_fit(feats)
        
        # Log model weights for transparency
        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        for model_name in MODEL_SLOTS:
            if model_name in self.weights:
                weight = self.weights[model_name]
                status = self.model_status.get(model_name, "unknown")
                impl = self.model_impl.get(model_name, "unknown")
                metrics = self.validation_results.get(model_name, {})
                rmse = metrics.get("RMSE", np.nan)
                mae = metrics.get("MAE", np.nan)
                logger.info(f"{model_name:10s} | Weight: {weight:6.3f} | Status: {status:12s} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | Impl: {impl}")
        logger.info("=" * 60)

        # Feature importances for interpretability with logging
        self.feature_importance = {}
        logger.info("Computing feature importances...")
        if "RF" in self.models and hasattr(self.models.get("RF"), "feature_importances_"):
            cols = self.model_feature_cols.get("RF", self.feature_cols)
            self.feature_importance["RF"] = dict(zip(cols, self.models["RF"].feature_importances_.tolist()))
            # Log top 5 features
            top_features = sorted(self.feature_importance["RF"].items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"RF top features: {', '.join([f'{k}({v:.3f})' for k, v in top_features])}")
        if "XGB" in self.models:
            imp = getattr(self.models.get("XGB"), "feature_importances_", None)
            if imp is not None:
                cols = self.model_feature_cols.get("XGB", self.feature_cols)
                self.feature_importance["XGB"] = dict(zip(cols, imp.tolist()))
                top_features = sorted(self.feature_importance["XGB"].items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"XGB top features: {', '.join([f'{k}({v:.3f})' for k, v in top_features])}")
        if "LGB" in self.models:
            imp = getattr(self.models.get("LGB"), "feature_importances_", None)
            if imp is not None:
                cols = self.model_feature_cols.get("LGB", self.feature_cols)
                self.feature_importance["LGB"] = dict(zip(cols, imp.tolist()))
                top_features = sorted(self.feature_importance["LGB"].items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"LGB top features: {', '.join([f'{k}({v:.3f})' for k, v in top_features])}")

        # Meta-ensemble on validation residuals with optimized configuration
        self.meta_model = None
        self.meta_models_order = []
        if self.val_index is not None and self.val_preds:
            candidate_order = ["Ridge", "RF", "XGB", "LGB", "SARIMAX", "Prophet"]
            common = [m for m in candidate_order if m in self.val_preds]
            if common:
                df_meta = pd.DataFrame(index=self.val_index)
                for m in common:
                    arr = self.val_preds[m]
                    if isinstance(arr, np.ndarray):
                        arr = pd.Series(arr, index=self.val_index[:len(arr)])
                    df_meta[m] = pd.Series(arr).reindex(self.val_index)
                yv = self.val_df["Average_Load"].reindex(self.val_index) if isinstance(self.val_df, pd.DataFrame) else pd.Series(dtype=float)
                mask = df_meta.notna().all(axis=1) & yv.notna()
                if mask.sum() >= 4:
                    # Optimized meta-model: ElasticNetCV for better model combination
                    try:
                        # Try ElasticNetCV for optimal blending
                        cv_meta = self._make_tscv(mask.sum())
                        if cv_meta is not None:
                            self.meta_model = ElasticNetCV(
                                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                alphas=np.logspace(-4, 0, 12),
                                cv=cv_meta,
                                max_iter=10000,
                                positive=True,  # Ensure positive weights for interpretability
                                random_state=DEFAULT_RANDOM_STATE,
                                n_jobs=-1
                            )
                            self.meta_model.fit(df_meta.loc[mask, common].values, yv.loc[mask].values)
                        else:
                            # Fallback to Ridge with very light regularization
                            self.meta_model = Ridge(alpha=1e-3, positive=True, random_state=DEFAULT_RANDOM_STATE)
                            self.meta_model.fit(df_meta.loc[mask, common].values, yv.loc[mask].values)
                        self.meta_models_order = common
                    except Exception as meta_err:
                        logger.warning(f"Meta-model training with ElasticNetCV failed: {meta_err}, using Ridge fallback")
                        self.meta_model = Ridge(alpha=1e-3, random_state=DEFAULT_RANDOM_STATE)
                        self.meta_model.fit(df_meta.loc[mask, common].values, yv.loc[mask].values)
                        self.meta_models_order = common



    def _validate_models_rolling(self, feats: pd.DataFrame):
        """Rolling-origin backtest with per-window refits (no leakage)."""
        windows: List[pd.DatetimeIndex] = []
        if feats is None or feats.empty:
            return {}, {}, None, {}
        end_anchor = feats.index.max()
        for k in range(max(1, self.rolling_k)):
            end = end_anchor - pd.DateOffset(months=k)
            start = end - pd.DateOffset(months=self.val_months - 1)
            windows.append(pd.date_range(start, end, freq="MS"))

        actual_series = self.load_df["Average_Load"].astype(float)
        last_preds: Dict[str, pd.Series] = {}
        last_idx = windows[0] if windows else None
        metrics_accum: Dict[str, List[Dict[str, float]]] = {}
        per_window_metrics: Dict[str, List[Dict[str, float]]] = {}
        per_bucket_metrics: Dict[str, List[Dict[str, float]]] = {}
        model_eval_order = MODEL_SLOTS + ["Naive"]
        band_components: List[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = []
        residual_feature_whitelist = [
            "lag1", "lag12", "roll3_mean", "roll12_mean",
            "mom_pct", "yoy_pct", "month_sin", "month_cos",
            "rjpp_mom_pct", "rjpp_yoy_pct", "rjpp_gap",
        ]

        for w_i, win_idx in enumerate(windows):
            actual_win = actual_series.reindex(win_idx)
            if actual_win.isna().all():
                continue
            win_start = win_idx[0]
            hist_end = win_start - pd.offsets.MonthBegin(1)
            hist = feats.loc[:hist_end].copy()
            if hist.empty or "Average_Load" not in hist.columns:
                continue
            X_all = hist[self.feature_cols] if self.feature_cols else pd.DataFrame(index=hist.index)
            y_level = hist["Average_Load"].astype(float)
            if self._residualize_effective and "RJPP" in hist.columns:
                rjpp_hist = hist["RJPP"].astype(float)
                y_target = (y_level - rjpp_hist).rename("target")
            else:
                rjpp_hist = hist.get("RJPP")
                y_target = y_level.rename("target")

            mask = (X_all.notna().all(axis=1) if not X_all.empty else pd.Series(True, index=hist.index))
            mask &= y_target.notna()
            if self._residualize_effective and rjpp_hist is not None:
                mask &= rjpp_hist.notna()
            X_all = X_all.loc[mask]
            y_target = y_target.loc[mask]
            if self._residualize_effective and rjpp_hist is not None:
                rjpp_hist = rjpp_hist.loc[mask]
            if len(y_target) < max(12, len(self.feature_cols)):
                continue

            train_weights_series = self._make_decay_weights(X_all.index)
            train_weights = train_weights_series.loc[X_all.index].astype(float).values if len(X_all) else None

            tmp_models = dict(self.models)
            tmp_feat_cols = {k: (v.copy() if isinstance(v, list) else v) for k, v in self.model_feature_cols.items()}
            tmp_scalers = dict(self.model_scalers)
            tmp_out_modes = dict(self.model_output_mode)
            tmp_residual_keys = set(self.residual_model_keys)

            ridge_features = [c for c in [
                "lag1", "lag12", "roll3_mean", "roll12_mean",
                "month_sin", "month_cos", "rjpp_mom_pct", "rjpp_yoy_pct", "rjpp_gap"
            ] if c in X_all.columns]
            if ridge_features:
                try:
                    r_scaler = RobustScaler().fit(X_all[ridge_features])
                    Xr = r_scaler.transform(X_all[ridge_features])
                    cv = self._make_tscv(len(X_all))
                    if cv is not None:
                        enet = ElasticNetCV(
                            l1_ratio=[0.1, 0.25, 0.5],
                            alphas=np.logspace(-3, 1, 12),
                            cv=cv,
                            max_iter=20000,
                            random_state=DEFAULT_RANDOM_STATE
                        )
                        enet.fit(Xr, y_target.values, sample_weight=train_weights)
                        tmp_models["Ridge"] = enet
                    else:
                        ridge = Ridge(alpha=1.0)
                        ridge.fit(Xr, y_target.values, sample_weight=train_weights)
                        tmp_models["Ridge"] = ridge
                    tmp_feat_cols["Ridge"] = ridge_features
                    tmp_scalers["Ridge"] = r_scaler
                    tmp_out_modes["Ridge"] = "residual" if self._residualize_effective else "level"
                    if self._residualize_effective:
                        tmp_residual_keys.add("Ridge")
                    else:
                        tmp_residual_keys.discard("Ridge")
                except Exception:
                    pass

            rf_features = ([c for c in residual_feature_whitelist if c in X_all.columns]
                           if self._residualize_effective else list(X_all.columns))
            if rf_features:
                try:
                    rf = RandomForestRegressor(
                        n_estimators=900,
                        max_depth=6,
                        min_samples_leaf=12,
                        min_samples_split=24,
                        max_features=0.6,
                        bootstrap=True,
                        ccp_alpha=3e-4,
                        random_state=DEFAULT_RANDOM_STATE,
                        n_jobs=-1
                    )
                    rf.fit(X_all[rf_features], y_target.values, sample_weight=train_weights)
                    tmp_models["RF"] = rf
                    tmp_feat_cols["RF"] = rf_features
                    tmp_scalers["RF"] = None
                    tmp_out_modes["RF"] = "residual" if self._residualize_effective else "level"
                    if self._residualize_effective:
                        tmp_residual_keys.add("RF")
                    else:
                        tmp_residual_keys.discard("RF")
                except Exception:
                    pass

            if self.use_xgb:
                xgb_features = ([c for c in residual_feature_whitelist if c in X_all.columns]
                                if self._residualize_effective else list(X_all.columns))
                if xgb_features:
                    try:
                        xgb = XGBRegressor(
                            n_estimators=1200,
                            learning_rate=0.03,
                            max_depth=4,
                            min_child_weight=8,
                            subsample=0.7,
                            colsample_bytree=0.7,
                            reg_lambda=12.0,
                            reg_alpha=2.0,
                            gamma=0.5,
                            objective="reg:squarederror",
                            tree_method="hist",
                            random_state=DEFAULT_RANDOM_STATE,
                            verbosity=0
                        )
                        xgb.fit(X_all[xgb_features].values, y_target.values, sample_weight=train_weights, verbose=False)
                        tmp_models["XGB"] = xgb
                        tmp_feat_cols["XGB"] = xgb_features
                        tmp_scalers["XGB"] = None
                        tmp_out_modes["XGB"] = "residual" if self._residualize_effective else "level"
                        if self._residualize_effective:
                            tmp_residual_keys.add("XGB")
                        else:
                            tmp_residual_keys.discard("XGB")
                    except Exception:
                        tmp_models.pop("XGB", None)

            if self.use_lgb:
                lgb_features = ([c for c in residual_feature_whitelist if c in X_all.columns]
                                if self._residualize_effective else list(X_all.columns))
                if lgb_features:
                    try:
                        from lightgbm import LGBMRegressor
                        lgb = LGBMRegressor(
                            n_estimators=4000,
                            learning_rate=0.02,
                            num_leaves=31,
                            max_depth=6,
                            min_data_in_leaf=40,
                            feature_fraction=0.75,
                            bagging_fraction=0.75,
                            bagging_freq=1,
                            lambda_l1=0.1,
                            lambda_l2=10.0,
                            min_gain_to_split=0.0,
                            random_state=DEFAULT_RANDOM_STATE,
                            verbose=-1
                        )
                        lgb.fit(X_all[lgb_features].values, y_target.values, sample_weight=train_weights)
                        tmp_models["LGB"] = lgb
                        tmp_feat_cols["LGB"] = lgb_features
                        tmp_scalers["LGB"] = None
                        tmp_out_modes["LGB"] = "residual" if self._residualize_effective else "level"
                        if self._residualize_effective:
                            tmp_residual_keys.add("LGB")
                        else:
                            tmp_residual_keys.discard("LGB")
                    except Exception:
                        tmp_models.pop("LGB", None)

            try:
                base_sar = self.models.get("SARIMAX")
                order = seasonal_order = None
                mode_sar = self.model_output_mode.get("SARIMAX", "level")
                if base_sar is not None:
                    try:
                        order = tuple(getattr(base_sar.model, "order", None))  # type: ignore[attr-defined]
                        seasonal_order = tuple(getattr(base_sar.model, "seasonal_order", None))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if order is None:
                    order = (1, 0, 0) if mode_sar == "residual" else (1, 1, 0)
                if seasonal_order is None:
                    seasonal_order = (0, 1, 1, 12) if mode_sar == "residual" else (1, 1, 1, 12)
                if mode_sar == "residual" and rjpp_hist is not None:
                    series = (y_level - rjpp_hist).dropna()
                    series = series - float(series.mean())
                    exog = None
                else:
                    series = y_level.dropna()
                    exog = hist.loc[series.index, "RJPP"] if "RJPP" in hist.columns else None
                if not series.empty:
                    sar_tmp = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        exog=exog,
                        trend=("n" if mode_sar == "residual" else "c"),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                    tmp_models["SARIMAX"] = sar_tmp
                    tmp_out_modes["SARIMAX"] = mode_sar
                    if mode_sar == "residual":
                        tmp_residual_keys.add("SARIMAX")
                    else:
                        tmp_residual_keys.discard("SARIMAX")
            except Exception:
                pass

            with temp_params(
                self,
                models=tmp_models,
                model_feature_cols=tmp_feat_cols,
                model_scalers=tmp_scalers,
                model_output_mode=tmp_out_modes,
                residual_model_keys=tmp_residual_keys,
                add_noise=False,
            ):
                fc_win = self.forecast(win_idx[0], win_idx[-1])
            raw_win = getattr(self, "latest_raw_forecast", pd.DataFrame()).reindex(fc_win.index) if hasattr(self, "latest_raw_forecast") else fc_win.copy()
            if self.rjpp is not None:
                fc_win["RJPP"] = self.rjpp.reindex(fc_win.index).ffill().values
                raw_win["RJPP"] = raw_win.get("RJPP", self.rjpp.reindex(fc_win.index).ffill().values)

            preds_for_window: Dict[str, pd.Series] = {}
            window_end = win_idx[-1]
            for name in model_eval_order:
                source_df = raw_win if name in raw_win.columns else fc_win
                if name not in source_df.columns:
                    continue
                preds = pd.Series(source_df[name]).reindex(win_idx)
                df_eval = pd.DataFrame({"actual": actual_win, "pred": preds}).dropna()
                if df_eval.empty:
                    continue
                mae = mean_absolute_error(df_eval["actual"], df_eval["pred"])
                rmse = _rmse(df_eval["actual"], df_eval["pred"])
                mape = mean_absolute_percentage_error(df_eval["actual"], df_eval["pred"])*100
                wape = _wape(df_eval["actual"], df_eval["pred"])
                smape = _smape(df_eval["actual"], df_eval["pred"])
                dev = np.nan
                bias = np.nan
                if self.rjpp is not None and "RJPP" in fc_win.columns:
                    r = pd.Series(fc_win["RJPP"]).reindex(win_idx)
                    df_dev = pd.DataFrame({"pred": preds, "rjpp": r}).dropna()
                    if not df_dev.empty:
                        denom = df_dev["rjpp"].values
                        mask_eval = denom != 0
                        if mask_eval.any():
                            ratios = (df_dev["pred"].values[mask_eval] - denom[mask_eval]) / denom[mask_eval]
                            dev = float(np.mean(np.abs(ratios)) * 100)
                            bias = float(np.mean(ratios) * 100)
                metrics_accum.setdefault(name, []).append({
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "WAPE": wape,
                    "sMAPE": smape,
                    "RJPP_dev%": dev,
                    "Bias%": bias,
                })
                per_window_metrics.setdefault(name, []).append({
                    "window_end": window_end,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "WAPE": wape,
                    "sMAPE": smape,
                    "RJPP_dev%": dev,
                    "Bias%": bias,
                })
                try:
                    df_bucket_eval = df_eval.sort_index()
                    if len(df_bucket_eval):
                        horizons = np.arange(1, len(df_bucket_eval) + 1, dtype=int)
                        df_bucket_eval = df_bucket_eval.assign(
                            _horizon=horizons,
                            _bucket=[_horizon_bucket(int(h)) for h in horizons],
                            _pred=df_bucket_eval["pred"].astype(float),
                            _actual=df_bucket_eval["actual"].astype(float),
                        )
                        if self.rjpp is not None and "RJPP" in fc_win.columns:
                            r_vals = pd.Series(fc_win["RJPP"]).reindex(df_bucket_eval.index).astype(float)
                            df_bucket_eval["_rjpp"] = r_vals
                        else:
                            df_bucket_eval["_rjpp"] = np.nan
                        for bucket, grp in df_bucket_eval.groupby("_bucket"):
                            if grp.empty:
                                continue
                            y_true = grp["_actual"].to_numpy(dtype=float)
                            y_pred = grp["_pred"].to_numpy(dtype=float)
                            mae_b = mean_absolute_error(y_true, y_pred)
                            rmse_b = _rmse(y_true, y_pred)
                            mape_b = mean_absolute_percentage_error(y_true, y_pred) * 100.0
                            wape_b = _wape(y_true, y_pred)
                            smape_b = _smape(y_true, y_pred)
                            dev_b = np.nan
                            bias_b = np.nan
                            if self.rjpp is not None and "RJPP" in fc_win.columns:
                                rjpp_vals = grp["_rjpp"].to_numpy(dtype=float)
                                mask_dev = np.isfinite(rjpp_vals) & (rjpp_vals != 0.0)
                                if mask_dev.any():
                                    ratios = (y_pred[mask_dev] - rjpp_vals[mask_dev]) / rjpp_vals[mask_dev]
                                    dev_b = float(np.mean(np.abs(ratios)) * 100.0)
                                    bias_b = float(np.mean(ratios) * 100.0)
                            per_bucket_metrics.setdefault(name, []).append({
                                "window_end": window_end,
                                "bucket": bucket,
                                "MAE": float(mae_b),
                                "RMSE": float(rmse_b),
                                "MAPE": float(mape_b),
                                "WAPE": float(wape_b),
                                "sMAPE": float(smape_b),
                                "RJPP_dev%": dev_b,
                                "Bias%": bias_b,
                            })
                except Exception as bucket_exc:
                    logger.debug(f"Bucket metric computation failed for {name}: {bucket_exc}")
                if w_i == 0:
                    last_preds[name] = preds.copy()
                preds_for_window[name] = preds

            model_cols_used = [m for m in ["Ridge", "RF", "XGB", "LGB", "SARIMAX", "Prophet"] if m in preds_for_window]
            if model_cols_used:
                preds_df = pd.concat([preds_for_window[m] for m in model_cols_used], axis=1, keys=model_cols_used)
                ensemble_equal = preds_df.mean(axis=1)
                spread_std = preds_df.std(axis=1, ddof=0)
                res_std = pd.Series([
                    np.nanstd(self.residuals_by_moy.get(ts.month, self.residuals.values if self.residuals is not None else np.array([0.0])))
                    for ts in preds_df.index
                ], index=preds_df.index)
                band_components.append((ensemble_equal, spread_std, res_std, actual_series.reindex(preds_df.index)))

        agg = {name: pd.DataFrame(lst).mean(numeric_only=True).to_dict() for name, lst in metrics_accum.items() if lst}

        try:
            if band_components:
                ens_concat = pd.concat([comp[0] for comp in band_components])
                spread_concat = pd.concat([comp[1] for comp in band_components])
                res_concat = pd.concat([comp[2] for comp in band_components])
                actual_concat = pd.concat([comp[3] for comp in band_components])
                sigma_base = np.sqrt(res_concat.values**2 + spread_concat.values**2)
                sigma_base = np.nan_to_num(sigma_base, nan=0.0)
                z = 1.2816
                best_k, best_err = 1.0, float("inf")
                target_coverage = PRESETS["band_target_coverage"]
                k_min, k_max, k_step = PRESETS["band_k_bounds"]
                for k in np.arange(k_min, k_max + k_step, k_step):
                    p10 = ens_concat.values - z * (k * sigma_base)
                    p90 = ens_concat.values + z * (k * sigma_base)
                    mask_vals = actual_concat.notna().values
                    if mask_vals.any():
                        hit = ((actual_concat.values[mask_vals] >= p10[mask_vals]) & (actual_concat.values[mask_vals] <= p90[mask_vals])).mean()
                        err = abs(hit - target_coverage)
                        if err < best_err:
                            best_err, best_k = err, k
                self.band_k = float(best_k)
                logger.info(f"Calibrated band_k={best_k:.2f} for target coverage={target_coverage:.0%}")
        except Exception as exc:
            logger.warning(f"Band calibration failed: {exc}")
            self.band_k = 1.0

        per_window_metrics_dfs = {name: pd.DataFrame(lst).sort_values("window_end")
                                  for name, lst in per_window_metrics.items() if lst}
        per_bucket_metrics_dfs = {
            name: pd.DataFrame(lst).sort_values(["bucket", "window_end"])
            for name, lst in per_bucket_metrics.items() if lst
        }
        self.validation_bucket_metrics = per_bucket_metrics_dfs

        return agg, last_preds, last_idx, per_window_metrics_dfs

    def _calibrate_residual_models(self) -> None:
        """Apply improved amplitude and bias calibration to residual-mode models."""
        self.model_calibrations = {}
        if not self._residualize_effective or not self.val_preds or self.val_index is None:
            return
        if not isinstance(self.val_df, pd.DataFrame) or "RJPP" not in self.val_df.columns:
            return
        rjpp_val = self.val_df["RJPP"].reindex(self.val_index).astype(float)
        actual_level = self.val_df["Average_Load"].reindex(self.val_index).astype(float)
        actual_resid = (actual_level - rjpp_val)

        logger.info("Calibrating residual models...")
        for model_name, preds in list(self.val_preds.items()):
            if model_name not in self.residual_model_keys:
                continue
            if preds is None:
                continue
            if isinstance(preds, np.ndarray):
                pred_series = pd.Series(preds, index=self.val_index[:len(preds)])
            else:
                pred_series = pd.Series(preds).reindex(self.val_index)
            pred_resid = pred_series - rjpp_val
            mask = pred_resid.notna() & actual_resid.notna()
            if mask.sum() < 3:
                logger.debug(f"{model_name}: Insufficient data for calibration (n={mask.sum()})")
                continue
            
            # Compute calibration parameters with improved bounds
            sigma_model = float(np.nanstd(pred_resid.loc[mask], ddof=0))
            sigma_true = float(np.nanstd(actual_resid.loc[mask], ddof=0))
            if not np.isfinite(sigma_model) or sigma_model <= 1e-9 or not np.isfinite(sigma_true):
                alpha = 1.0
            else:
                # Improved alpha bounds for better calibration: allow wider range
                alpha = float(np.clip(sigma_true / sigma_model, 0.75, 1.25))
            
            adjusted_resid = pred_resid.loc[mask] * alpha
            bias = float(adjusted_resid.mean()) if len(adjusted_resid) else 0.0
            
            # Apply bounds to bias to prevent extreme corrections
            bias_std = float(np.nanstd(adjusted_resid)) if len(adjusted_resid) else 0.0
            if abs(bias) > 2.0 * bias_std and bias_std > 0:
                bias = np.sign(bias) * 2.0 * bias_std  # Cap bias at 2 standard deviations
            
            self.model_calibrations[model_name] = {"scale": alpha, "bias": bias}
            calibrated = (pred_resid * alpha) - bias
            self.val_preds[model_name] = calibrated.add(rjpp_val, fill_value=0.0).reindex(self.val_index)
            
            # Log calibration details
            pre_rmse = _rmse(actual_resid.loc[mask], pred_resid.loc[mask])
            post_rmse = _rmse(actual_resid.loc[mask], calibrated.loc[mask])
            logger.info(f"{model_name:10s} | Scale: {alpha:6.3f} | Bias: {bias:8.2f} | RMSE: {pre_rmse:7.2f} → {post_rmse:7.2f} ({((post_rmse-pre_rmse)/pre_rmse*100):+.1f}%)")

    def _weights_from_validation(self, valres: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute global inverse-error weights with improved multi-metric optimization."""
        keys = [k for k in valres if k!="Naive"]
        if not keys: return {}
        adjusted = {}
        for k in keys:
            v = valres[k]
            rmse = v.get("RMSE")
            mae = v.get("MAE", rmse)  # Use MAE if available, else RMSE
            dev = v.get("RJPP_dev%") or 0
            bias = abs(v.get("Bias%", 0))  # Consider bias magnitude
            
            if rmse and np.isfinite(rmse) and mae and np.isfinite(mae):
                # Combined metric: weighted average of RMSE and MAE, penalized by deviation and bias
                # RMSE gets 60% weight, MAE gets 40% for better balance
                combined_error = 0.6 * rmse + 0.4 * mae
                # Penalize deviation from RJPP (if applicable) and bias
                penalty = 1.0 + (dev / 100.0) + (bias / 200.0)  # Bias has half the weight of deviation
                adjusted[k] = combined_error * penalty
        
        if not adjusted: return {k:1/len(keys) for k in keys}
        
        # Optimized temperature for better weight distribution
        temperature = 0.65  # Slightly reduced for more decisive weighting
        scores = {k: (1.0/adj)**temperature for k, adj in adjusted.items() if adj>0}
        if not scores:
            return {k:1/len(keys) for k in keys}
        
        total = sum(scores.values())
        weights = {k: scores[k]/total for k in scores}
        
        # Improved floor and ceiling for better weight distribution
        min_weight = 0.03  # Minimum weight per model (3%)
        max_weight = 0.45  # Maximum weight per model (45%) to prevent dominance
        
        # Apply floor
        floor = min_weight
        weights = {k: max(w, floor) for k, w in weights.items()}
        
        # Apply ceiling and renormalize
        weights = {k: min(w, max_weight) for k, w in weights.items()}
        total = sum(weights.values())
        if total > 0:
            weights = {k: w/total for k, w in weights.items()}
        
        return weights

    def _compute_per_bucket_weights(self, bucket_metrics: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Compute per-bucket inverse-error weights from rolling validation with improved multi-metric scoring.
        bucket_metrics: {model_name: DataFrame with columns [bucket, RMSE, RJPP_dev%, Bias%, ...]}
        Returns: {bucket: {model: weight}}
        """
        # Optimized penalty weights for better model selection
        LAMBDA_DEV = 0.8   # penalty on absolute deviation vs RJPP (reduced for better balance)
        LAMBDA_BIAS = 0.4  # penalty on signed bias vs RJPP (reduced for better balance)
        LAMBDA_MAE = 0.3   # weight for MAE in addition to RMSE
        MAX_W = 0.50       # prevent any single model from dominating a bucket (reduced from 0.55)
        MIN_W = 0.02       # minimum weight per model per bucket
        bucket_rmse: Dict[str, Dict[str, List[float]]] = {}

        for model_name, df in bucket_metrics.items():
            if model_name == "Naive":
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "bucket" not in df.columns or "RMSE" not in df.columns:
                continue

            for bucket, df_bucket in df.groupby("bucket"):
                df_bucket = df_bucket.dropna(subset=["RMSE"])
                if df_bucket.empty:
                    continue
                rmse_values = df_bucket["RMSE"].astype(float)
                mae_values = df_bucket.get("MAE", rmse_values).astype(float)  # Use MAE if available
                dev_series = df_bucket.get("RJPP_dev%", pd.Series(index=df_bucket.index, dtype=float)).fillna(0.0)
                bias_series = df_bucket.get("Bias%", pd.Series(index=df_bucket.index, dtype=float)).fillna(0.0).abs()
                
                # Improved composite error metric combining RMSE, MAE, deviation, and bias
                # RMSE (60%) + MAE (30%) weighted combination with penalty terms
                base_error = 0.6 * rmse_values + 0.3 * mae_values
                adj = base_error * (
                    1.0
                    + LAMBDA_DEV * np.clip(dev_series.to_numpy(dtype=float), 0, None) / 100.0
                    + LAMBDA_BIAS * bias_series.to_numpy(dtype=float) / 100.0
                )
                bucket_rmse.setdefault(bucket, {}).setdefault(model_name, []).extend(adj.tolist())

        # Compute weights per bucket with improved normalization
        weights_by_bucket: Dict[str, Dict[str, float]] = {}
        for bucket, model_rmses in bucket_rmse.items():
            if not model_rmses:
                continue
            # Average adjusted error per model in this bucket
            avg_rmse = {m: np.mean(rmse_list) for m, rmse_list in model_rmses.items() if len(rmse_list) > 0}
            if not avg_rmse:
                continue
            # Inverse-error weights with epsilon
            eps = 1e-9
            inv_errors = {m: 1.0 / (r + eps) for m, r in avg_rmse.items()}
            total = sum(inv_errors.values())
            if total > 0:
                raw_w = {m: w / total for m, w in inv_errors.items()}
                # Apply floor first to ensure minimum representation
                floored = {m: max(w, MIN_W) for m, w in raw_w.items()}
                # Then apply ceiling and renormalize
                capped = {m: min(w, MAX_W) for m, w in floored.items()}
                s = sum(capped.values())
                if s > 0:
                    weights_by_bucket[bucket] = {m: w / s for m, w in capped.items()}
                else:
                    # Equal weights fallback
                    n = len(capped)
                    weights_by_bucket[bucket] = {m: 1.0 / n for m in capped}
            else:
                # Equal weights fallback
                n = len(avg_rmse)
                weights_by_bucket[bucket] = {m: 1.0 / n for m in avg_rmse}

        return weights_by_bucket

    def _combine_with_weights(self, idx: pd.Index, base_cols: List[str], value_matrix: np.ndarray) -> np.ndarray:
        """
        Combine model predictions using horizon-aware weights with per-row NaN renormalization.
        """
        if idx is None or len(idx) == 0 or not base_cols:
            return np.full(len(idx) if idx is not None else 0, np.nan, dtype=float)
        if value_matrix.size == 0:
            return np.full(len(idx), np.nan, dtype=float)

        value_matrix = np.asarray(value_matrix, dtype=float)
        rows = min(len(idx), value_matrix.shape[0])
        combined = np.full(len(idx), np.nan, dtype=float)
        use_per_bucket = bool(self.model_weights_by_bucket)
        fallback_weights = self.weights if isinstance(self.weights, dict) else {}

        for i in range(rows):
            row = value_matrix[i, :]
            mask = np.isfinite(row)
            if not mask.any():
                continue
            bucket_weights: Dict[str, float] = {}
            if use_per_bucket:
                bucket = _horizon_bucket(i + 1)
                bucket_weights = self.model_weights_by_bucket.get(bucket, {})
            if not bucket_weights and fallback_weights:
                bucket_weights = fallback_weights

            weights_vec = np.array([bucket_weights.get(c, 0.0) for c in base_cols], dtype=float)
            weights_vec = weights_vec * mask.astype(float)
            total = float(weights_vec.sum())
            if total <= 0.0:
                weights_vec = mask.astype(float)
                total = float(weights_vec.sum())
            if total <= 0.0:
                combined[i] = float(np.mean(row[mask]))
                continue
            weights_vec = weights_vec / total
            combined[i] = float(np.dot(row[mask], weights_vec[mask]))

        return combined


    def _in_sample_fit(self, feats: pd.DataFrame) -> pd.DataFrame:
        idx = feats.index
        out = pd.DataFrame(index=idx)
        rjpp_series = feats["RJPP"] if "RJPP" in feats.columns else None

        # Non-iterative models
        sarimax_model = self.models.get("SARIMAX")
        if "SARIMAX" in self.non_iterative_keys and sarimax_model is not None:
            try:
                mode = self.model_output_mode.get("SARIMAX", "level")
                ex = (rjpp_series.values if (rjpp_series is not None and mode != "residual") else None)
                # Use predict with start/end aligned to the model's data
                pred = sarimax_model.predict(start=0, end=len(idx)-1, exog=ex)
                vals = pred.values if hasattr(pred, 'values') else np.asarray(pred)
                if mode == "residual" and rjpp_series is not None:
                    vals = np.asarray(vals, dtype=float) + rjpp_series.reindex(idx).astype(float).values
                out["SARIMAX"] = vals
            except Exception as e:
                logger.warning(f"In-sample SARIMAX prediction failed: {e}")

        prophet_model = self.models.get("Prophet")
        if "Prophet" in self.non_iterative_keys and prophet_model is not None and hasattr(prophet_model, "predict"):
            try:
                mode = self.model_output_mode.get("Prophet", "level")
                fdf = feats.reset_index().rename(columns={"Date": "ds"})
                if mode == "level" and rjpp_series is not None:
                    fdf = fdf.assign(RJPP=rjpp_series.values)
                    cols = ["ds", "RJPP"]
                else:
                    cols = ["ds"]
                yhat = prophet_model.predict(fdf[cols])["yhat"].values
                if mode == "residual" and rjpp_series is not None:
                    yhat = yhat + rjpp_series.reindex(idx).astype(float).values
                out["Prophet"] = yhat
            except Exception as e:
                logger.warning(f"In-sample Prophet prediction failed: {e}")

        for slot in MODEL_SLOTS:
            if slot in self.non_iterative_keys:
                continue
            model = self.models.get(slot)
            if model is None:
                continue
            features = self.model_feature_cols.get(slot, self.feature_cols)
            if not features:
                continue
            x_slot = feats[features].copy()
            # Fill missing values with model-specific imputed values
            imputer_values = self.model_imputers.get(slot, pd.Series(dtype=float))
            if isinstance(imputer_values, pd.Series) and not imputer_values.empty:
                for col in features:
                    if col in imputer_values and col in x_slot.columns:
                        x_slot[col] = x_slot[col].fillna(imputer_values[col])
            # Forward/backward fill any remaining NaNs
            x_slot = x_slot.ffill().bfill()
            if x_slot.empty:
                continue
            values = np.nan_to_num(x_slot.values.astype(float), nan=0.0)
            try:
                if slot in self.linear_model_keys:
                    scaler = self.model_scalers.get(slot)
                    if scaler is not None and hasattr(scaler, "n_features_in_") and values.shape[1] == scaler.n_features_in_:
                        preds = model.predict(scaler.transform(values))
                    else:
                        preds = model.predict(values)
                else:
                    preds = model.predict(values)
            except Exception as e:
                logger.warning(f"In-sample fit failed for {slot}: {e}")
                continue
            series = pd.Series(preds, index=idx)
            if slot in self.residual_model_keys:
                cal = self.model_calibrations.get(slot)
                if cal:
                    series = (series * cal.get("scale", 1.0)) - cal.get("bias", 0.0)
                if rjpp_series is not None:
                    series = series.add(rjpp_series, fill_value=0.0)
            out[slot] = np.maximum(series, 0.0)

        out["Naive"] = self.load_df["Average_Load"].shift(12).reindex(idx)
        return out

    # ---------- Forecast ----------

    def forecast(self, start, end):
        logger.info("Forecasting...")
        idx = pd.date_range(start, end, freq="MS")
        if len(idx) == 0:
            return pd.DataFrame(index=idx)

        rjpp_future = None
        if self.rjpp is not None:
            rjpp_future = self.rjpp.reindex(pd.date_range(self.rjpp.index.min(), end, freq="MS")).ffill().reindex(idx)["RJPP"]

        preds_raw: Dict[str, np.ndarray] = {}
        naive_series = self._naive_forecast_series(start, end)
        preds_raw["Naive"] = naive_series.values

        feature_cols = self.feature_cols if hasattr(self, "feature_cols") else []
        col_idx = {c: i for i, c in enumerate(feature_cols)}

        def build_vec(ts: pd.Timestamp, lag_series: pd.Series) -> np.ndarray:
            v = np.full((1, len(feature_cols)), np.nan, dtype=float)
            if not feature_cols:
                return v
            if 'year' in col_idx:
                v[0, col_idx['year']] = ts.year
            if 'month' in col_idx:
                v[0, col_idx['month']] = ts.month
            if 'month_sin' in col_idx:
                v[0, col_idx['month_sin']] = np.sin(2*np.pi*ts.month/12)
            if 'month_cos' in col_idx:
                v[0, col_idx['month_cos']] = np.cos(2*np.pi*ts.month/12)
            if 'quarter' in col_idx:
                v[0, col_idx['quarter']] = ts.quarter
            if 'quarter_sin' in col_idx:
                v[0, col_idx['quarter_sin']] = np.sin(2*np.pi*ts.quarter/4)
            if 'quarter_cos' in col_idx:
                v[0, col_idx['quarter_cos']] = np.cos(2*np.pi*ts.quarter/4)
            if 'lag1' in col_idx:
                v[0, col_idx['lag1']] = lag_series.get(ts - pd.DateOffset(months=1), np.nan)
            if 'lag12' in col_idx:
                v[0, col_idx['lag12']] = lag_series.get(ts - pd.DateOffset(months=12), np.nan)
            if 'naive' in col_idx:
                v[0, col_idx['naive']] = lag_series.get(ts - pd.DateOffset(months=12), np.nan)
            if 'roll3_mean' in col_idx:
                v[0, col_idx['roll3_mean']] = lag_series.rolling(window=3, min_periods=1).mean().iloc[-1] if len(lag_series) else np.nan
            if 'roll12_mean' in col_idx:
                v[0, col_idx['roll12_mean']] = lag_series.rolling(window=12, min_periods=1).mean().iloc[-1] if len(lag_series) else np.nan
            if 'mom_pct' in col_idx:
                v[0, col_idx['mom_pct']] = lag_series.pct_change().iloc[-1] if len(lag_series) > 1 else np.nan
            if 'yoy_pct' in col_idx:
                v[0, col_idx['yoy_pct']] = lag_series.pct_change(12).iloc[-1] if len(lag_series) > 12 else np.nan
            if rjpp_future is not None and 'RJPP' in col_idx:
                v[0, col_idx['RJPP']] = float(rjpp_future.loc[ts])
                prev_rjpp = rjpp_future.get(ts - pd.DateOffset(months=1), np.nan)
                if 'RJPP_lag1' in col_idx:
                    v[0, col_idx['RJPP_lag1']] = prev_rjpp
                if 'rjpp_gap' in col_idx:
                    prev_load = lag_series.get(ts - pd.DateOffset(months=1), np.nan)
                    v[0, col_idx['rjpp_gap']] = np.nan if np.isnan(prev_rjpp) or np.isnan(prev_load) else prev_rjpp - prev_load
                if 'rjpp_yoy_pct' in col_idx:
                    prev = rjpp_future.get(ts - pd.DateOffset(months=12), np.nan)
                    curr = float(rjpp_future.loc[ts])
                    v[0, col_idx['rjpp_yoy_pct']] = np.nan if np.isnan(prev) or prev == 0 else (curr/prev - 1)
                if 'rjpp_mom_pct' in col_idx:
                    prev = prev_rjpp
                    curr = float(rjpp_future.loc[ts])
                    v[0, col_idx['rjpp_mom_pct']] = np.nan if np.isnan(prev) or prev == 0 else (curr/prev - 1)
            return v

        history_series = self.load_df["Average_Load"] if hasattr(self, "load_df") else pd.Series(dtype=float)
        anchor_end = pd.Timestamp(start) - pd.offsets.MonthBegin(1)
        anchor_vals = history_series.loc[:anchor_end].dropna() if not history_series.empty else pd.Series(dtype=float)
        if len(anchor_vals):
            last_actual = float(anchor_vals.iloc[-1])
        else:
            non_nan_hist = history_series.dropna()
            last_actual = float(non_nan_hist.iloc[-1]) if len(non_nan_hist) else 0.0

        # Non-iterative models
        sarimax_model = self.models.get("SARIMAX")
        if "SARIMAX" in self.non_iterative_keys and sarimax_model is not None and hasattr(sarimax_model, "get_forecast"):
            try:
                mode = self.model_output_mode.get("SARIMAX", "level")
                ex = rjpp_future.values if (rjpp_future is not None and self.rjpp is not None and mode != "residual") else None
                sarimax_forecast = sarimax_model.get_forecast(len(idx), exog=ex).predicted_mean.values
                arr = np.array(sarimax_forecast, dtype=float)
                if mode == "residual":
                    cal = self.model_calibrations.get("SARIMAX")
                    if cal:
                        arr = (arr * cal.get("scale", 1.0)) - cal.get("bias", 0.0)
                    if rjpp_future is not None:
                        arr = arr + rjpp_future.values
            except Exception as e:
                logger.warning(f"SARIMAX forecast fallback to naive: {e}")
                arr = naive_series.values.copy()
        elif "SARIMAX" in self.non_iterative_keys and sarimax_model is not None and hasattr(sarimax_model, "forecast"):
            try:
                arr = np.array(sarimax_model.forecast(len(idx)), dtype=float)
                if mode == "residual":
                    cal = self.model_calibrations.get("SARIMAX")
                    if cal:
                        arr = (arr * cal.get("scale", 1.0)) - cal.get("bias", 0.0)
                    if rjpp_future is not None:
                        arr = arr + rjpp_future.values
            except Exception as e:
                logger.warning(f"ETS forecast fallback to naive: {e}")
                arr = naive_series.values.copy()
        else:
            arr = naive_series.values.copy()
        preds_raw["SARIMAX"] = np.maximum(arr, 0.0) if self.apply_guards_to == "all" else arr.astype(float)

        prophet_model = self.models.get("Prophet")
        if "Prophet" in self.non_iterative_keys and prophet_model is not None and hasattr(prophet_model, "predict"):
            try:
                mode = self.model_output_mode.get("Prophet", "level")
                fdf = pd.DataFrame({"ds": idx})
                if mode == "level" and rjpp_future is not None:
                    fdf["RJPP"] = rjpp_future.values
                arr = prophet_model.predict(fdf)["yhat"].values
                if mode == "residual":
                    cal = self.model_calibrations.get("Prophet")
                    if cal:
                        arr = (arr * cal.get("scale", 1.0)) - cal.get("bias", 0.0)
                    if rjpp_future is not None:
                        arr = arr + rjpp_future.values
            except Exception as e:
                logger.warning(f"Prophet forecast fallback to naive: {e}")
                arr = naive_series.values.copy()
            preds_raw["Prophet"] = (np.maximum(arr.astype(float), 0.0) if self.apply_guards_to == "all" else arr.astype(float))
        elif "Prophet" in self.non_iterative_keys and prophet_model is not None:
            preds_raw["Prophet"] = (np.maximum(naive_series.values.copy(), 0.0) if self.apply_guards_to == "all" else naive_series.values.copy())

        driver_base = history_series.copy()
        base_defaults = naive_series.values
        for slot in MODEL_SLOTS:
            if slot in preds_raw:
                continue
            model = self.models.get(slot)
            feature_list = self.model_feature_cols.get(slot, feature_cols)
            scaler = self.model_scalers.get(slot)
            series = driver_base.copy()
            outputs: List[float] = []
            for step, ts in enumerate(idx):
                if feature_cols:
                    vec_raw = build_vec(ts, series)
                    vec_series = pd.Series(vec_raw.flatten(), index=feature_cols)
                else:
                    vec_series = pd.Series(dtype=float)
                features = feature_list if feature_list else feature_cols
                if model is None or (not features and slot in self.linear_model_keys):
                    yhat = float(base_defaults[step])
                else:
                    row = vec_series.reindex(features).to_frame().T if features else pd.DataFrame([[0.0]])
                    # Per-model median imputation (avoids zero-bias and wild extrapolation)
                    imp = self.model_imputers.get(slot)
                    if imp is not None and not getattr(imp, "empty", False):
                        row = row.fillna(imp)
                    else:
                        row = row.fillna(0.0)
                    row_values = np.nan_to_num(row.values.astype(float), nan=0.0)
                    try:
                        if slot in self.linear_model_keys and scaler is not None and hasattr(scaler, "n_features_in_") and row_values.shape[1] == scaler.n_features_in_:
                            yhat = float(model.predict(scaler.transform(row_values)))
                        else:
                            yhat = float(model.predict(row_values))
                    except Exception:
                        yhat = float(base_defaults[step])
                if slot in self.residual_model_keys:
                    cal = self.model_calibrations.get(slot)
                    if cal:
                        yhat = (yhat * cal.get("scale", 1.0)) - cal.get("bias", 0.0)
                if slot in self.residual_model_keys and rjpp_future is not None:
                    r_base = float(rjpp_future.loc[ts]) if ts in rjpp_future.index and pd.notna(rjpp_future.loc[ts]) else 0.0
                    yhat += r_base
                if step == 0 and np.isfinite(last_actual):
                    yhat = REANCHOR_BLEND * last_actual + (1 - REANCHOR_BLEND) * yhat
                # Apply per-model guard clamps ONLY when scope == 'all'; otherwise leave raw
                if self.apply_guards_to == "all":
                    if rjpp_future is not None and self.use_rjpp:
                        cap = self.max_dev * SOFT_MODEL_CLIP_MULT
                        r = float(rjpp_future.loc[ts]) if ts in rjpp_future.index and pd.notna(rjpp_future.loc[ts]) else np.nan
                        if np.isfinite(r):
                            yhat = float(np.clip(yhat, r * (1 - cap), r * (1 + cap)))
                    else:
                        if len(outputs):
                            prev = outputs[-1]
                            yhat = float(np.clip(yhat, prev * 0.75, prev * 1.25))
                    yhat = max(0.0, float(yhat))
                outputs.append(yhat)
                series.loc[ts] = yhat
            preds_raw[slot] = np.array(outputs, dtype=float)

        raw_df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in preds_raw.items()}, index=idx)
        if rjpp_future is not None:
            raw_df["RJPP"] = rjpp_future.values

        guard_all = self.apply_guards_to == "all"
        rvals = rjpp_future.values if rjpp_future is not None else None
        out_df = pd.DataFrame(index=idx)
        base_model_cols: List[str] = []
        for slot in MODEL_SLOTS:
            if slot not in raw_df:
                continue
            base_model_cols.append(slot)
            series_vals = raw_df[slot].values
            guarded_vals = self._apply_series_guards(series_vals, rvals) if guard_all else series_vals
            out_df[slot] = guarded_vals
            out_df[f"{slot}_raw"] = series_vals
        out_df["Naive"] = raw_df.get("Naive", naive_series).values if not guard_all else np.maximum(raw_df.get("Naive", naive_series).values, 0.0)
        out_df["Naive_raw"] = raw_df.get("Naive", naive_series).values
        if rjpp_future is not None:
            out_df["RJPP"] = rjpp_future.values

        base_cols_for_weights = [c for c in base_model_cols if c in raw_df.columns]
        raw_matrix = raw_df[base_cols_for_weights].to_numpy() if base_cols_for_weights else np.empty((len(idx), 0))

        if raw_matrix.size:
            ens_weighted_raw = self._combine_with_weights(idx, base_cols_for_weights, raw_matrix)
            with np.errstate(all='ignore'):
                equal_mean = np.nanmean(raw_matrix, axis=1)
            fallback_series = np.where(np.isfinite(equal_mean), equal_mean, naive_series.values)
            nan_mask_raw = ~np.isfinite(ens_weighted_raw)
            if nan_mask_raw.any():
                ens_weighted_raw[nan_mask_raw] = fallback_series[nan_mask_raw]
        else:
            ens_weighted_raw = naive_series.values.copy()

        if guard_all and base_cols_for_weights:
            guarded_matrix = out_df[base_cols_for_weights].to_numpy(dtype=float)
            ens_weighted_pre = self._combine_with_weights(idx, base_cols_for_weights, guarded_matrix)
            with np.errstate(all='ignore'):
                equal_mean_guard = np.nanmean(guarded_matrix, axis=1)
            fallback_guard = np.where(np.isfinite(equal_mean_guard), equal_mean_guard, naive_series.values)
            nan_mask_pre = ~np.isfinite(ens_weighted_pre)
            if nan_mask_pre.any():
                ens_weighted_pre[nan_mask_pre] = fallback_guard[nan_mask_pre]
        else:
            ens_weighted_pre = ens_weighted_raw.copy()

        out_df["Ensemble_weighted_raw"] = ens_weighted_raw.astype(float)
        out_df["Ensemble_weighted"] = ens_weighted_pre.astype(float)

        ens_meta_raw = None
        ens_meta_pre = None
        if self.meta_model is not None and self.meta_models_order:
            meta_cols = [m for m in self.meta_models_order if m in raw_df.columns]
            if meta_cols:
                meta_matrix = raw_df[meta_cols].to_numpy()
                mask = np.isfinite(meta_matrix).all(axis=1)
                ens_meta_raw = np.full(len(idx), np.nan)
                if mask.any():
                    ens_meta_raw[mask] = self.meta_model.predict(meta_matrix[mask])
                if guard_all:
                    guarded_meta = out_df[meta_cols].to_numpy()
                    mask_guard = np.isfinite(guarded_meta).all(axis=1)
                    ens_meta_pre = np.full(len(idx), np.nan)
                    if mask_guard.any():
                        ens_meta_pre[mask_guard] = self.meta_model.predict(guarded_meta[mask_guard])
                else:
                    ens_meta_pre = ens_meta_raw.copy()
                out_df["Ensemble_meta_raw"] = ens_meta_raw
                out_df["Ensemble_meta"] = ens_meta_pre

        self.latest_raw_forecast = raw_df.copy()
        self.latest_raw_forecast["Ensemble_weighted_raw"] = ens_weighted_raw
        if ens_meta_raw is not None:
            self.latest_raw_forecast["Ensemble_meta_raw"] = ens_meta_raw

        fc = self._align_noise_clip(out_df, raw_df, rjpp_future)
        fc = fc.join(self._quantile_bands(fc, raw_df))
        return fc

    def _compute_seasonal_strength(self) -> float:
        """Compute seasonal strength from STL decomposition (0-1 scale)."""
        try:
            if not hasattr(self, 'load_df') or self.load_df is None:
                return 0.0
            from statsmodels.tsa.seasonal import STL
            series = self.load_df["Average_Load"].dropna()
            if len(series) < 24:
                return 0.0
            stl = STL(series, seasonal=13, robust=True)
            result = stl.fit()
            var_seasonal = np.var(result.seasonal)
            var_resid = np.var(result.resid)
            total_var = var_seasonal + var_resid
            if total_var == 0:
                return 0.0
            strength = max(0.0, min(1.0, 1.0 - (var_resid / total_var)))
            return float(strength)
        except Exception as e:
            logger.warning(f"Seasonal strength computation failed: {e}")
            return 0.0

    def _apply_annual_scaler(self, ensemble: pd.Series, rjpp: pd.Series) -> pd.Series:
        """Apply annual-only RJPP scaler with tolerance from PRESETS."""
        if rjpp is None or ensemble is None:
            return ensemble
        tolerance = PRESETS["annual_anchor_tolerance"]
        result = ensemble.copy()
        for year in sorted(set(ensemble.index.year)):
            mask = ensemble.index.year == year
            if not mask.any():
                continue
            e_mean = float(ensemble.loc[mask].mean())
            r_mean = float(rjpp.loc[mask].mean())
            if not (np.isfinite(e_mean) and np.isfinite(r_mean) and r_mean > 0 and e_mean > 0):
                continue
            k = r_mean / e_mean
            k = float(np.clip(k, 1 - tolerance, 1 + tolerance))
            result.loc[mask] = result.loc[mask] * k
            logger.info(f"Annual scaler for {year}: k={k:.4f}")
        return result

    def _apply_rjpp_guards_vec(self, ensemble: pd.Series, rjpp: pd.Series) -> pd.Series:
        """Vectorized RJPP guard applying horizon-aware deviation caps."""
        if ensemble is None or ensemble.empty or rjpp is None:
            return ensemble
        idx = ensemble.index
        dev = pd.Series([
            _max_dev_for_horizon(i + 1) for i in range(len(idx))
        ], index=idx, dtype=float)
        aligned = rjpp.reindex(idx)
        if aligned is None or aligned.empty:
            return ensemble
        lower = aligned * (1 - dev.values)
        upper = aligned * (1 + dev.values)
        return ensemble.clip(lower=lower, upper=upper)

    def _apply_horizon_caps(self, ensemble: pd.Series, rjpp: pd.Series, start_date: pd.Timestamp) -> pd.Series:
        """Apply horizon-aware deviation caps vs RJPP (soft guard, not magnet)."""
        return self._apply_rjpp_guards_vec(ensemble, rjpp)

    def _apply_mom_caps(self, ensemble: pd.Series, start_date: pd.Timestamp) -> pd.Series:
        """Apply horizon-aware MoM caps."""
        result = ensemble.copy()
        for i in range(1, len(result)):
            h = i + 1  # horizon
            cap = _mom_cap_for_horizon(h)
            prev = result.iloc[i-1]
            curr = result.iloc[i]
            if prev > 0 and np.isfinite(prev) and np.isfinite(curr):
                change = abs((curr - prev) / prev)
                if change > cap:
                    # Clamp to cap
                    if curr > prev:
                        result.iloc[i] = prev * (1 + cap)
                    else:
                        result.iloc[i] = prev * (1 - cap)
        return result

    def _add_residual_noise(self, ensemble: pd.Series, rjpp: pd.Series) -> pd.Series:
        """Add residual-based noise for realism."""
        if not self.add_noise or not self.residuals_by_moy:
            return ensemble
        rng = np.random.default_rng(self.random_state)
        result = ensemble.copy()
        for ts in ensemble.index:
            pool = self.residuals_by_moy.get(ts.month, self.residuals.values if self.residuals is not None else np.array([0.0]))
            if pool is not None and len(pool):
                noise_val = rng.choice(pool) * self.noise_scale
                result.loc[ts] = result.loc[ts] + noise_val
        # Re-apply non-negativity after noise
        result = result.clip(lower=0.0)
        return result

    def _align_noise_clip(self, df_fc: pd.DataFrame, df_raw: pd.DataFrame, rjpp_future: Optional[pd.Series]) -> pd.DataFrame:
        """
        NEW THEORY-CORRECT APPROACH:
        1. Choose ensemble (weighted vs meta) based on self.ensemble_choice
        2. Apply annual-only RJPP scaler (±2% tolerance, no monthly blending)
        3. Apply horizon-aware deviation caps vs RJPP
        4. Apply horizon-aware MoM caps
        5. Add residual noise for realism
        6. Enforce non-negativity
        NO unconditional smoothing, NO monthly blending toward RJPP
        """
        out = df_fc.copy()
        
        # 1. Choose ensemble based on validation RMSE
        chosen = "Ensemble_meta" if ("Ensemble_meta" in out.columns and (self.ensemble_override=="meta" or (self.ensemble_override is None and self.ensemble_choice=="meta"))) else "Ensemble_weighted"
        
        base_series = out[chosen] if chosen in out.columns else None
        if base_series is None:
            raw_key = f"{chosen}_raw"
            if raw_key in df_raw.columns:
                base_series = pd.Series(df_raw[raw_key], index=df_raw.index).reindex(out.index)
            elif chosen == "Ensemble_meta" and "Ensemble_weighted_raw" in df_raw.columns:
                base_series = pd.Series(df_raw["Ensemble_weighted_raw"], index=df_raw.index).reindex(out.index)
            elif "Ensemble_weighted_raw" in df_raw.columns:
                base_series = pd.Series(df_raw["Ensemble_weighted_raw"], index=df_raw.index).reindex(out.index)
            else:
                base_series = pd.Series(np.nan, index=out.index)
            out[chosen] = base_series
        if chosen == "Ensemble_meta" and isinstance(base_series, pd.Series) and "Ensemble_weighted" in out.columns:
            base_series = base_series.astype(float).fillna(out["Ensemble_weighted"].astype(float))
            out[chosen] = base_series

        ensemble_series = base_series.astype(float).copy()
        
        # Prepare RJPP series
        rjpp_series = None
        if "RJPP" in out.columns:
            rjpp_series = out["RJPP"].astype(float)
        elif rjpp_future is not None:
            rjpp_series = rjpp_future.reindex(out.index).ffill().astype(float)
        
        # 2. Apply annual-only RJPP scaler (no monthly blending)
        if rjpp_series is not None and self.use_rjpp:
            ensemble_series = self._apply_annual_scaler(ensemble_series, rjpp_series)
            # Store annual factors for reporting
            self.rjpp_alignment_factors = {
                int(y): float(ensemble_series.loc[ensemble_series.index.year == y].mean() / 
                             rjpp_series.loc[rjpp_series.index.year == y].mean())
                for y in sorted(set(out.index.year))
                if (ensemble_series.index.year == y).any() and 
                   rjpp_series.loc[rjpp_series.index.year == y].mean() > 0
            }
        else:
            self.rjpp_alignment_factors = {}
        
        # 3. Apply horizon-aware deviation caps (soft guard, not magnet)
        if rjpp_series is not None and self.use_rjpp:
            ensemble_series = self._apply_horizon_caps(ensemble_series, rjpp_series, out.index[0])
        
        # 4. Apply horizon-aware MoM caps
        ensemble_series = self._apply_mom_caps(ensemble_series, out.index[0])
        
        # 5. Enforce non-negativity
        ensemble_series = ensemble_series.clip(lower=0.0)
        
        # 6. Seasonal imprint anchored to historical seasonal index
        beta_base = float(self.seasonal_imprint_beta)
        if beta_base > 0.0 and self.seasonal_strength > 0.1:
            try:
                hist = self.load_df["Average_Load"].dropna()
                if len(hist) >= 24:
                    monthly_means = hist.groupby(hist.index.month).mean()
                    base = float(monthly_means.mean()) if len(monthly_means) else 0.0
                    if base > 0:
                        seasonal_idx = (monthly_means / base) - 1.0
                        beta = beta_base * float(min(1.0, max(0.0, self.seasonal_strength)))
                        shaped = []
                        for ts in ensemble_series.index:
                            val = ensemble_series.loc[ts]
                            s = float(seasonal_idx.get(ts.month, 0.0))
                            shaped.append(val * (1.0 + beta * s))
                        ensemble_series = pd.Series(shaped, index=ensemble_series.index, dtype=float)
                        ensemble_series = ensemble_series.clip(lower=0.0)
            except Exception as e:
                logger.warning(f"Seasonal imprint failed: {e}")


# 7. Add residual noise for realism
        if self.add_noise:
            ensemble_series = self._add_residual_noise(ensemble_series, rjpp_series)
        
        # Store final ensemble
        out["Ensemble"] = ensemble_series.values
        
        # For backward compatibility, also create Ensemble_sampled with noise
        if self.add_noise and self.residuals_by_moy:
            out["Ensemble_sampled"] = ensemble_series.values  # Already has noise
        
        if chosen not in out.columns:
            out[chosen] = base_series
        
        return out
    # ---------- Bands ----------
    def _quantile_bands(self, fc_final: pd.DataFrame, fc_raw: pd.DataFrame) -> pd.DataFrame:
        idx = fc_final.index; ens = fc_final["Ensemble"].values
        res_std = np.zeros(len(idx))
        for i,ts in enumerate(idx):
            pool = self.residuals_by_moy.get(ts.month, (self.residuals.values if self.residuals is not None else np.array([0.0])))
            res_std[i] = np.nanstd(pool) if pool is not None and len(pool) else 0.0
        model_cols = [c for c in ["Ridge","RF","XGB","LGB","SARIMAX","Prophet"] if c in fc_raw.columns]
        if model_cols:
            mtx = np.column_stack([np.nan_to_num(fc_raw[c].values) for c in model_cols])
            def _robust_std(a, axis=0):
                med = np.nanmedian(a, axis=axis)
                mad = np.nanmedian(np.abs(a - np.expand_dims(med, axis=axis)), axis=axis)
                return 1.4826 * mad
            spread_std = _robust_std(mtx, axis=1)
        else:
            spread_std = np.zeros(len(idx))
        sigma_base = np.sqrt(res_std**2 + spread_std**2)
        z = 1.2816
        k = getattr(self, "band_k", 1.0)  # auto-calibrated on validation
        sigma = k * sigma_base

        p50 = ens
        p10 = np.clip(ens - z*sigma, 0, None)
        p90 = np.clip(ens + z*sigma, 0, None)
        return pd.DataFrame({"p10":p10,"p50":p50,"p90":p90}, index=idx)

    def _validation_bands_and_coverage(self, ens_val_series: pd.Series) -> Tuple[pd.DataFrame, float]:
        if self.val_index is None or not len(self.val_index): return pd.DataFrame(), np.nan
        idx = self.val_index
        res_std = np.array([
            np.nanstd(self.residuals_by_moy.get(ts.month, self.residuals.values if self.residuals is not None else np.array([0.0])))
            for ts in idx
        ])
        model_cols = [m for m in ["Ridge","RF","XGB","LGB","SARIMAX","Prophet"] if m in self.val_preds]
        if model_cols:
            mtx = np.column_stack([pd.Series(self.val_preds[m]).reindex(idx).astype(float).values for m in model_cols])
            spread_std = np.nanstd(mtx, axis=1)
        else:
            spread_std = np.zeros(len(idx))
        sigma_base = np.sqrt(res_std**2 + spread_std**2)
        z=1.2816
        k = getattr(self, "band_k", 1.0)
        p50 = ens_val_series.reindex(idx).astype(float).values
        p10 = np.clip(p50 - z*(k*sigma_base), 0, None)
        p90 = np.clip(p50 + z*(k*sigma_base), 0, None)
        bands = pd.DataFrame({"p10":p10,"p50":p50,"p90":p90}, index=idx)
        if self.val_df is not None and "Average_Load" in self.val_df:
            actual = self.val_df["Average_Load"].reindex(idx)
            # Mask out NaNs in actuals before computing hit-rate
            mask = actual.notna()
            if mask.any():
                cover = ((actual[mask] >= bands.loc[mask, "p10"]) & (actual[mask] <= bands.loc[mask, "p90"]))
                cover = float(cover.mean() * 100)
            else:
                cover = np.nan
        else:
            cover = np.nan
        return bands, cover

    # ---------- Run ----------
    def run_pipeline(self, load_path, rjpp_path, start_date, end_date):
        self.load(load_path, rjpp_path)
        self.train()
        return self._assemble_results(start_date, end_date)

    def run_post_training(self, start_date, end_date):
        """Reuse a pre-trained forecaster (e.g., from cache) to generate outputs."""
        if not hasattr(self, "load_df") or self.load_df is None:
            raise RuntimeError("Forecaster must be loaded before running post-training pipeline")
        return self._assemble_results(start_date, end_date)

    def _assemble_results(self, start_date, end_date):
        # choose ensemble mode using validation
        ens_weight_last = None; ens_meta_last = None
        if self.val_index is not None and self.val_preds:
            base_models_val = [m for m in MODEL_SLOTS if m in self.val_preds]
            ens_weight_last = None
            if base_models_val:
                val_matrix = []
                for mname in base_models_val:
                    arr = self.val_preds[mname]
                    if isinstance(arr, np.ndarray):
                        arr = pd.Series(arr, index=self.val_index[:len(arr)])
                    series = pd.Series(arr).reindex(self.val_index).astype(float)
                    val_matrix.append(series.to_numpy())
                val_matrix_np = np.column_stack(val_matrix) if val_matrix else np.empty((len(self.val_index), 0))
                if val_matrix_np.size:
                    combined_vals = self._combine_with_weights(self.val_index, base_models_val, val_matrix_np)
                    with np.errstate(all='ignore'):
                        equal_mean_val = np.nanmean(val_matrix_np, axis=1)
                    fallback_val = np.where(np.isfinite(equal_mean_val), equal_mean_val, np.nan)
                    nan_mask_val = ~np.isfinite(combined_vals)
                    if nan_mask_val.any():
                        combined_vals[nan_mask_val] = fallback_val[nan_mask_val]
                    ens_weight_last = pd.Series(combined_vals, index=self.val_index, name="Ensemble_weighted_val")
            if ens_weight_last is None:
                ens_weight_last = pd.Series(np.nan, index=self.val_index, name="Ensemble_weighted_val")
            if self.meta_model is not None and self.meta_models_order:
                df_meta = pd.DataFrame(index=self.val_index)
                for m in self.meta_models_order:
                    arr = self.val_preds.get(m)
                    if isinstance(arr, np.ndarray): arr = pd.Series(arr, index=self.val_index[:len(arr)])
                    df_meta[m] = pd.Series(arr).reindex(self.val_index)
                mask_m = df_meta.notna().all(axis=1)
                if mask_m.any():
                    ens_meta_last = pd.Series(self.meta_model.predict(df_meta.loc[mask_m, self.meta_models_order].values),
                                              index=self.val_index[mask_m], name="Ensemble_meta_val")

            # auto pick if no override (use PRESETS setting)
            yv = self.val_df["Average_Load"].reindex(self.val_index)
            rmse_w = np.inf; rmse_m = np.inf
            if ens_weight_last is not None:
                mask_w = yv.notna() & ens_weight_last.notna()
                if mask_w.any(): rmse_w = _rmse(yv.loc[mask_w], ens_weight_last.loc[mask_w])
            if ens_meta_last is not None:
                mask_m = yv.loc[ens_meta_last.index].notna()
                if mask_m.any(): rmse_m = _rmse(yv.loc[ens_meta_last.index][mask_m], ens_meta_last.loc[mask_m])
            
            # Store validation RMSEs for reporting
            self.val_rmse_weighted = rmse_w
            self.val_rmse_meta = rmse_m
            
            # Choose based on PRESETS or validation RMSE
            if PRESETS["ensemble_selection"] == "auto":
                self.ensemble_choice = "meta" if rmse_m < rmse_w else "weighted"
                logger.info(f"Auto-selected ensemble: {self.ensemble_choice} (weighted RMSE={rmse_w:.2f}, meta RMSE={rmse_m:.2f})")
            elif PRESETS["ensemble_selection"] == "weighted":
                self.ensemble_choice = "weighted"
            elif PRESETS["ensemble_selection"] == "meta":
                self.ensemble_choice = "meta"
            else:
                self.ensemble_choice = "weighted"  # default fallback

        if self.ensemble_override in ("weighted","meta"):
            self.ensemble_choice = self.ensemble_override
            logger.info(f"Ensemble override to: {self.ensemble_override}")

        fc = self.forecast(start_date, end_date)
        fc_raw = self.latest_raw_forecast.reindex(fc.index) if getattr(self, "latest_raw_forecast", None) is not None else fc.copy()
        rjpp_proximity = {}
        if "RJPP" in fc_raw.columns:
            rjpp_vals = fc_raw["RJPP"].replace(0, np.nan)
            for model_name in MODEL_SLOTS:
                if model_name in fc_raw.columns:
                    diff = (fc_raw[model_name] - rjpp_vals).astype(float)
                    rel = diff / rjpp_vals
                    rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
                    if not rel.empty:
                        mean_signed_pct = float(rel.mean()*100)
                        pct_within = float((rel.abs() <= 0.05).mean()*100)
                    else:
                        mean_signed_pct = np.nan
                        pct_within = np.nan
                    rjpp_proximity[model_name] = {"mean_signed_pct": mean_signed_pct, "pct_within_5": pct_within}

        naive_curve, naive_metrics = self.naive_holdout()

        full_idx = pd.date_range(self.load_df.index.min(), fc.index.max(), freq="MS")
        combined = pd.Series(np.nan, index=full_idx, name="Combined")
        combined.loc[self.load_df.index] = self.load_df["Average_Load"]
        combined.loc[fc.index] = fc["Ensemble"]

        rjpp_full = None
        if self.rjpp is not None:
            rjpp_full = self.rjpp.reindex(full_idx).ffill()

        best_model=None
        if self.validation_results:
            rmse_vals={k:v["RMSE"] for k,v in self.validation_results.items() if "RMSE" in v}
            if rmse_vals: best_model=min(rmse_vals, key=rmse_vals.get)

        # ensemble metrics on validation
        ensemble_val_pred=None; ensemble_val_metrics=None; calib_pct=np.nan; val_bands=pd.DataFrame()
        if self.val_index is not None and ens_weight_last is not None:
            yv = self.val_df["Average_Load"].reindex(self.val_index).values
            yhat_last = ens_weight_last.reindex(self.val_index).fillna(0.0).values \
                        if self.ensemble_choice=="weighted" else \
                        (ens_meta_last.reindex(self.val_index).ffill().bfill().values if ens_meta_last is not None else None)
            if yhat_last is not None:
                ensemble_val_pred = pd.Series(yhat_last, index=self.val_index, name="Ensemble_val")
                ensemble_val_metrics = {
                    "MAE": mean_absolute_error(yv, yhat_last),
                    "RMSE": _rmse(yv, yhat_last),
                    "MAPE": mean_absolute_percentage_error(yv, yhat_last)*100,
                    "WAPE": _wape(yv, yhat_last),
                    "sMAPE": _smape(yv, yhat_last)
                }
        val_bands, calib_pct = self._validation_bands_and_coverage(ensemble_val_pred)

        return {
            "forecast": fc,
            "forecast_raw": fc_raw,
            "combined_forecast": combined.to_frame(),
            "rjpp_full": rjpp_full,
            "validation_results": self.validation_results,
            "model_weights": self.weights,
            "naive_holdout_forecast": naive_curve,
            "naive_holdout_metrics": naive_metrics,
            "rjpp_alignment_factors": self.rjpp_alignment_factors,
            "best_model": best_model,
            "in_sample_fitted": self.in_sample_pred,
            "val_index": self.val_index,
            "ensemble_val_pred": ensemble_val_pred,
            "ensemble_val_metrics": ensemble_val_metrics,
            "feature_importance": self.feature_importance,
            "ensemble_choice": self.ensemble_choice,
            "val_df": self.val_df.copy() if self.val_df is not None else None,
            "model_status": self.model_status,
            "model_errors": self.model_errors,
            "val_bands": val_bands,
            "calibration_pct": calib_pct,
            "validation_windows": self.validation_windows,
            "rjpp_proximity": rjpp_proximity,
        }

# ---------- Streamlit App ----------
st.set_page_config(page_title="Load Forecasting", page_icon="LF", layout="wide")

# ----- Initialize Session State First (Before Any Usage) -----
if "results" not in st.session_state: 
    st.session_state.results = None
if "forecaster" not in st.session_state: 
    st.session_state.forecaster = None
if "fc_raw" not in st.session_state: 
    st.session_state.fc_raw = None
if "combined" not in st.session_state: 
    st.session_state.combined = None
if "rjpp_full" not in st.session_state: 
    st.session_state.rjpp_full = None
if "filesig" not in st.session_state: 
    st.session_state.filesig = (None, None)
if "training_params" not in st.session_state:
    st.session_state.training_params = {
        "val_months": DEFAULT_VAL_MONTHS,
        "rolling_k": DEFAULT_ROLLING_K,
        "mom_limit_quantile": 0.90,
        "max_dev": DEFAULT_MAX_DEV,
        "add_noise": PRESETS["add_noise"],
        "noise_scale": PRESETS["noise_scale"],
    }
if "model_flags" not in st.session_state:
    st.session_state.model_flags = {
        "use_prophet": PROPHET_AVAILABLE,
        "use_xgb": XGB_AVAILABLE,
        "use_lgb": LGB_AVAILABLE,
    }
if "training_params_dirty" not in st.session_state:
    st.session_state.training_params_dirty = False
if "ai_focus_banner" not in st.session_state:
    st.session_state.ai_focus_banner = False
if "last_training_sig" not in st.session_state:
    st.session_state.last_training_sig = None

# When this file is imported (not run as main), avoid executing the Streamlit UI code
# which expects a ScriptRunContext. Exiting early prevents import-time NameErrors.
if __name__ != "__main__":
    raise SystemExit(0)

# Scenario-related state
if "scenarios" not in st.session_state: 
    st.session_state.scenarios = []
if "scenario_files" not in st.session_state: 
    st.session_state.scenario_files = {}
if "apply_scenarios" not in st.session_state: 
    st.session_state.apply_scenarios = False
if "fc_adjusted" not in st.session_state: 
    st.session_state.fc_adjusted = None
if "combined_adjusted" not in st.session_state: 
    st.session_state.combined_adjusted = None
if "planned_overlay" not in st.session_state: 
    st.session_state.planned_overlay = None

# AI-related state
if "ai_markdown" not in st.session_state: 
    st.session_state.ai_markdown = None
if "ai_error" not in st.session_state: 
    st.session_state.ai_error = None
if "ai_running" not in st.session_state: 
    st.session_state.ai_running = False
if "ai_cancel" not in st.session_state:
    st.session_state.ai_cancel = False

# Form and editor state
if "scenario_editor_selected" not in st.session_state: 
    st.session_state.scenario_editor_selected = "<New>"
if "scenario_form_state" not in st.session_state: 
    st.session_state.scenario_form_state = SCENARIO_FORM_DEFAULT.copy()
if "pending_filesig" not in st.session_state: 
    st.session_state.pending_filesig = (None, None)

st.title("Load Forecasting")
st.caption("RJPP-anchored ensemble with uncertainty bands and model diagnostics.")

# Explain mode toggle (inline help everywhere)
explain_on = st.toggle("🧠 Explain mode", value=False, key="explain_mode")

# CSS polish
st.markdown("""
<style>
[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 10px 12px; }
.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ----- Sidebar inputs & quick actions -----
with st.sidebar:
    st.header("Inputs")
    load_file = st.file_uploader(
        "Historical Load Data (CSV)",
        type=["csv"],
        help="Must contain 'Date' and 'Average_Load' (monthly).",
    )
    rjpp_file = st.file_uploader(
        "RJPP Data (CSV)",
        type=["csv"],
        help="Optional 'Date' + 'RJPP'.",
    )

    forecast_start = st.date_input(
        "Forecast start",
        value=FORECAST_START,
        min_value=FORECAST_START,
        max_value=FORECAST_END,
    )
    forecast_end = st.date_input(
        "Forecast end",
        value=FORECAST_END,
        min_value=FORECAST_START,
        max_value=FORECAST_END,
    )

    st.markdown("---")
    st.header("Quick actions")
    run_btn = st.button(
        "🔮 Run forecast",
        type="primary",
        use_container_width=True,
        help="Compute forecasts using uploaded data and current settings.",
    )
    run_ai_quick = st.button(
        "🧠 Run AI insight",
        use_container_width=True,
        help="Jump to the AI Insight tab and prepare a summary.",
    )
    prev_noise = bool(st.session_state.training_params.get("add_noise", PRESETS["add_noise"]))
    add_noise_toggle = st.toggle(
        "Add residual noise",
        value=prev_noise,
    )
    if bool(add_noise_toggle) != prev_noise:
        st.session_state.training_params_dirty = True
    st.session_state.training_params["add_noise"] = bool(add_noise_toggle)
    st.caption("Noise adds realism to bands & scenarios")

if 'run_ai_quick' in locals() and run_ai_quick:
    st.session_state.ai_focus_banner = True

# Derive flags from PRESETS (no UI controls)
use_rjpp = True  # Always use RJPP if available
residualize_flag = True  # Always residualize models
apply_guards_to = "ensemble"  # Only apply guards to ensemble, not individual models
guard_mom_cap = None  # Will be computed dynamically per-horizon from PRESETS
guard_smoothing_window = 1  # No unconditional smoothing (smoothing_alpha=0.0)
display_mode = "Both"  # Default to showing both history and forecast

# Scenario apply toggle now lives exclusively in the Scenarios tab header (single source of truth per redesign spec).

# ----- Helper functions for file handling and plotting -----

def _file_sig(uploaded):
    return hashlib.md5(uploaded.getvalue()).hexdigest() if uploaded is not None else None

def shaded(fig, start_x, end_x): fig.add_vrect(x0=start_x, x1=end_x, fillcolor="rgba(200,200,200,0.15)", line_width=0)

def plot_hero(combined, fc, rjpp_full, key_suffix="", show_bands: bool = True, bounded: bool = False, max_dev: Optional[float] = None, fc_adj: Optional[pd.DataFrame] = None, combined_adj: Optional[pd.Series] = None, planned_overlay: Optional[pd.Series] = None, label_base: str = "Base", label_adj: str = "Scenario-Adjusted", forecast_start: Optional[pd.Timestamp] = None):
    fig = go.Figure(layout=PLOTLY_BASE_LAYOUT)
    shade_start = fc.index.min() if len(fc.index) else None
    if shade_start is not None:
        shaded(fig, shade_start, fc.index.max())

    expected_start = None
    if forecast_start is not None:
        expected_start = pd.to_datetime(forecast_start).to_period("M").to_timestamp()
    if expected_start is not None and shade_start is not None:
        if shade_start != expected_start:
            st.warning(f"Forecast start misalignment detected at {shade_start:%Y-%m}; expected {expected_start:%Y-%m}")
        assert shade_start == expected_start, (
            f"Forecast begins at {shade_start:%Y-%m}; expected {expected_start:%Y-%m}."
        )

    base_col = "p50" if "p50" in fc.columns else ("Ensemble" if "Ensemble" in fc.columns else None)
    base_series = fc[base_col].astype(float) if base_col is not None else pd.Series(dtype=float)

    adjusted_series = None
    if fc_adj is not None:
        if "p50_adj" in fc_adj.columns:
            adjusted_series = fc_adj["p50_adj"].astype(float)
        elif "Ensemble_adj" in fc_adj.columns:
            adjusted_series = fc_adj["Ensemble_adj"].astype(float)

    if isinstance(combined, pd.DataFrame):
        combined_series = combined.iloc[:, 0].astype(float)
        combined_series.name = combined.columns[0]
    else:
        combined_series = pd.Series(combined, dtype=float)

    forecast_index = fc.index if len(fc.index) else combined_series.index
    stitched_base = pd.Series(
        np.nan,
        index=combined_series.index.union(forecast_index),
        dtype=float,
        name=base_series.name if hasattr(base_series, "name") else None,
    )
    if base_col and len(base_series):
        stitched_base.loc[forecast_index] = base_series.reindex(forecast_index).values

    stitched_adj = None
    if adjusted_series is not None:
        adj_index = adjusted_series.index
        if combined_adj is not None:
            if isinstance(combined_adj, pd.DataFrame):
                combined_adj_series = combined_adj.iloc[:, 0].astype(float)
            else:
                combined_adj_series = pd.Series(combined_adj, dtype=float)
            stitched_adj_index = combined_adj_series.index.union(adj_index)
        else:
            stitched_adj_index = combined_series.index.union(adj_index)
        stitched_adj = pd.Series(np.nan, index=stitched_adj_index, dtype=float)
        stitched_adj.loc[adj_index] = adjusted_series.reindex(adj_index).values

    if fc_adj is not None and {"p10_adj", "p90_adj"}.issubset(fc_adj.columns):
        col_map = {}
        if "p10_adj" in fc_adj.columns:
            col_map["p10"] = fc_adj["p10_adj"]
        if "p90_adj" in fc_adj.columns:
            col_map["p90"] = fc_adj["p90_adj"]
        if "p50_adj" in fc_adj.columns:
            col_map["p50"] = fc_adj["p50_adj"]
        bands_df = pd.DataFrame(col_map, index=fc_adj.index)
        band_label = label_adj
        band_color = PALETTE.get("scenario", PALETTE["ensemble"])
    else:
        bands_df = fc.copy()
        if "p50" not in bands_df.columns and "Ensemble" in bands_df.columns:
            bands_df["p50"] = bands_df["Ensemble"]
        band_label = label_base
        band_color = PALETTE["ensemble"]

    if bounded and show_bands and max_dev is not None and {"p10", "p90"}.issubset(bands_df.columns):
        rjpp_series = None
        if rjpp_full is not None and isinstance(rjpp_full, pd.DataFrame) and "RJPP" in rjpp_full.columns:
            rjpp_series = rjpp_full["RJPP"].reindex(fc.index)
        elif "RJPP" in fc.columns:
            rjpp_series = fc["RJPP"]
        if rjpp_series is not None:
            clamp_lo = rjpp_series * (1 - BAND_MULT * max_dev)
            clamp_hi = rjpp_series * (1 + BAND_MULT * max_dev)
            bands_df = bands_df.copy()
            bands_df["p10"] = np.maximum(bands_df["p10"], clamp_lo)
            bands_df["p90"] = np.minimum(bands_df["p90"], clamp_hi)

    if show_bands and {"p10", "p90"}.issubset(bands_df.columns):
        fig.add_trace(go.Scatter(x=bands_df.index, y=bands_df["p90"], name=f"{band_label} p90", mode="lines", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=bands_df.index, y=bands_df["p10"], name=f"{band_label} p10", mode="lines", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=bands_df.index, y=bands_df["p90"], name=f"{band_label} band upper", mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bands_df.index, y=bands_df["p10"], name=f"{band_label} band lower", mode="lines", fill="tonexty", opacity=0.12 if fc_adj is not None else 0.18, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        if "p50" in bands_df.columns:
            fig.add_trace(go.Scatter(x=bands_df.index, y=bands_df["p50"], name=f"{band_label} p50", mode="lines", line=dict(width=2, dash="dot", color=band_color), showlegend=False))

    history_series = combined_series.copy()
    if shade_start is not None:
        history_series.loc[shade_start:] = np.nan
    fig.add_trace(go.Scatter(x=history_series.index, y=history_series.values, name="Actual (history)", mode="lines", line=dict(width=1.6, color=PALETTE["history"])))

    if len(stitched_base):
        base_name = f"{label_base} ({base_col})" if base_col else label_base
        fig.add_trace(go.Scatter(x=stitched_base.index, y=stitched_base.values, name=base_name, mode="lines", line=dict(width=3, color=PALETTE["ensemble"])))

    if stitched_adj is not None:
        fig.add_trace(go.Scatter(x=stitched_adj.index, y=stitched_adj.values, name=label_adj, mode="lines", line=dict(width=3, color=PALETTE.get("scenario", PALETTE["ensemble"]), dash="dash")))

    if planned_overlay is not None:
        overlay_series = planned_overlay.reindex(fc.index).fillna(0)
        if overlay_series.any():
            fig.add_trace(go.Scatter(x=overlay_series.index, y=overlay_series.values, name="Planned MW", mode="lines", fill="tozeroy", opacity=0.2, line=dict(width=1.4, color=PALETTE.get("planned", "#6366F1"))))

    if rjpp_full is not None:
        fig.add_trace(go.Scatter(x=rjpp_full.index, y=rjpp_full["RJPP"], name="RJPP", mode="lines", line=dict(dash="dot", color=PALETTE["rjpp"])))

    if shade_start is not None:
        fig.add_vline(x=shade_start, line_width=1, line_dash="dot", line_color="orange")

    fig.update_yaxes(range=[0, None], ticksuffix=" MW")
    fig.update_layout(height=560, yaxis_title="Load (MW)")
    st.plotly_chart(fig, use_container_width=True, key=f"hero_{key_suffix}")

def plot_per_model(combined, fc, fc_raw, rjpp_full, slot_info):
    figm = go.Figure(layout=PLOTLY_BASE_LAYOUT)
    shade_start = fc.index.min() if len(fc.index) else None
    if shade_start is not None:
        shaded(figm, shade_start, fc.index.max())
    full_idx = pd.date_range(combined.index.min(), combined.index.max(), freq="MS")
    actual_only = pd.Series(np.nan, index=full_idx, name="Actual (history)")
    actual_only.loc[:HISTORY_CUTOFF] = combined.loc[:HISTORY_CUTOFF]
    figm.add_trace(go.Scatter(x=actual_only.index, y=actual_only.values, name="Actual (history)", mode="lines",
                              line=dict(width=1.2, color="rgba(120,120,120,0.8)")))
    for m in MODEL_SLOTS:
        if m in fc_raw.columns:
            s = pd.Series(np.nan, index=full_idx, dtype=float)
            s.loc[fc_raw.index.min():] = fc_raw[m].values
            impl = slot_info.get(m, {}).get("impl")
            label = impl if impl else ("SARIMA(X)" if m=="SARIMAX" else ("Regression (Ridge)" if m=="Ridge" else m))
            figm.add_trace(go.Scatter(x=s.index, y=s.values, name=label, mode="lines", line=dict(width=1.6)))
    stitched = pd.concat([actual_only.loc[:HISTORY_CUTOFF], fc["Ensemble"]]).reindex(full_idx)
    figm.add_trace(go.Scatter(x=stitched.index, y=stitched.values, name="Ensemble", mode="lines", line=dict(width=3, color=PALETTE["ensemble"])))
    if rjpp_full is not None:
        figm.add_trace(go.Scatter(x=rjpp_full.index, y=rjpp_full["RJPP"], name="RJPP", line=dict(dash="dot", color=PALETTE["rjpp"])))
    if shade_start is not None:
        figm.add_vline(x=shade_start, line_width=1, line_dash="dot", line_color="orange")
    figm.update_yaxes(range=[0, None], ticksuffix=" MW", title_text="Load (MW)")
    figm.update_layout(height=560)
    st.plotly_chart(figm, use_container_width=True, key="per_model_stitched")


def plot_reliability(val_actual: pd.Series, p10: pd.Series, p90: pd.Series):
    aligned_idx = val_actual.index.intersection(p10.index).intersection(p90.index)
    if aligned_idx.empty:
        st.info("No overlapping validation data available for reliability diagram.")
        return
    actual = val_actual.reindex(aligned_idx).astype(float)
    lower = p10.reindex(aligned_idx).astype(float)
    upper = p90.reindex(aligned_idx).astype(float)
    mask = actual.notna() & lower.notna() & upper.notna()
    if not mask.any():
        st.info("Validation data lacks complete bands for reliability diagram.")
        return
    inside = ((actual >= lower) & (actual <= upper))[mask].astype(int)
    quarter_map = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}
    quarter_series = ((inside.index.month - 1) // 3).map(quarter_map)
    grouped = inside.groupby(quarter_series).mean().reindex(["Q1", "Q2", "Q3", "Q4"]).fillna(0.0)
    fig = go.Figure(
        go.Bar(x=grouped.index.astype(str), y=(grouped * 100.0), marker_color=PALETTE.get("ensemble", "#F5BF2C"))
    )
    fig.update_layout(
        PLOTLY_BASE_LAYOUT | {
            "yaxis_title": "Inside band (%)",
            "xaxis_title": "Validation quarter",
            "yaxis": {"range": [0, 100]},
        }
    )
    st.plotly_chart(fig, use_container_width=True, key="band_reliability")

def corr_heatmap(df: pd.DataFrame, cols: List[str]):
    x = df[cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2: return None
    c = x.corr()
    fig = px.imshow(c, text_auto=".2f", aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(PLOTLY_BASE_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

def scenario_apply(forecaster, fc_raw, max_dev_pct: float, ensemble_mode: str, add_noise_flag: bool):
    mode = None if ensemble_mode == "Auto" else ("weighted" if ensemble_mode=="Weighted" else "meta")
    with temp_params(forecaster, max_dev=max_dev_pct, ensemble_override=mode, add_noise=add_noise_flag):
        base = fc_raw.drop(columns=["RJPP"]) if (not getattr(forecaster, "use_rjpp", True) and "RJPP" in fc_raw.columns) else fc_raw
        rjpp_future = None
        if getattr(forecaster, "rjpp", None) is not None: 
            rjpp_future = forecaster.rjpp.reindex(base.index).ffill()["RJPP"]
        fc_alt = forecaster._align_noise_clip(base, fc_raw, rjpp_future)
        fc_alt = fc_alt.join(forecaster._quantile_bands(fc_alt, fc_raw))
    return fc_alt

# ----- Session state bootstrap (already initialized above) -----

# ----- Run/Refresh pipeline when needed -----
if load_file is not None:
    load_sig = _file_sig(load_file)
    rjpp_sig = _file_sig(rjpp_file) if rjpp_file is not None else None
    st.session_state.pending_filesig = (load_sig, rjpp_sig)
else:
    st.session_state.pending_filesig = (None, None)

if run_btn and load_file is None:
    st.warning("Upload CSVs before running the forecast.")

if load_file is not None and run_btn:
    try:
        start_dt, end_dt = _month_start(forecast_start), _month_start(forecast_end)
        load_path = _save_file(load_file, "load_")
        rjpp_path = _save_file(rjpp_file, "rjpp_") if rjpp_file else None
        
        if load_path is None:
            st.error("Failed to save load file. Please try uploading again.")
            st.stop()
        
        training_params = st.session_state.training_params.copy()
        model_flags = st.session_state.model_flags.copy()

        # ensure plain python types for hashing
        training_params["val_months"] = int(training_params.get("val_months", DEFAULT_VAL_MONTHS))
        training_params["rolling_k"] = int(training_params.get("rolling_k", DEFAULT_ROLLING_K))
        training_params["mom_limit_quantile"] = float(training_params.get("mom_limit_quantile", 0.90))
        training_params["max_dev"] = float(training_params.get("max_dev", DEFAULT_MAX_DEV))
        training_params["add_noise"] = bool(training_params.get("add_noise", PRESETS["add_noise"]))
        training_params["noise_scale"] = float(training_params.get("noise_scale", PRESETS["noise_scale"]))

        init_params = {
            "max_dev": training_params["max_dev"],
            "add_noise": training_params["add_noise"],
            "ensemble_override": None,
            "random_state": DEFAULT_RANDOM_STATE,
            "use_rjpp": use_rjpp,
            "residualize": residualize_flag,
            "apply_guards_to": apply_guards_to,
            "stabilize_output": False,
            "guard_mom_cap": guard_mom_cap,
            "guard_smoothing_window": guard_smoothing_window,
            "noise_scale": training_params["noise_scale"],
            "use_prophet": bool(model_flags.get("use_prophet", PROPHET_AVAILABLE)),
            "use_xgb": bool(model_flags.get("use_xgb", XGB_AVAILABLE)),
            "use_lgb": bool(model_flags.get("use_lgb", LGB_AVAILABLE)),
        }

        runtime_params = {
            "val_months": training_params["val_months"],
            "rolling_k": training_params["rolling_k"],
            "mom_limit_quantile": training_params["mom_limit_quantile"],
            "max_dev": training_params["max_dev"],
            "add_noise": training_params["add_noise"],
            "noise_scale": training_params["noise_scale"],
            "ensemble_override": None,
            "guard_mom_cap": guard_mom_cap,
            "guard_smoothing_window": guard_smoothing_window,
            "use_prophet": init_params["use_prophet"],
            "use_xgb": init_params["use_xgb"],
            "use_lgb": init_params["use_lgb"],
            "use_rjpp": use_rjpp,
            "residualize": residualize_flag,
            "apply_guards_to": apply_guards_to,
            "stabilize_output": False,
        }

        cache_sig = _training_signature(load_path, rjpp_path, init_params, runtime_params)
        files_changed = st.session_state.pending_filesig != st.session_state.filesig
        if files_changed:
            logger.info("Detected new input data; recomputing forecast.")
        
        cache_hit = (st.session_state.last_training_sig == cache_sig) and not files_changed
        status_label = "Reusing cached models…" if cache_hit else "Training models…"
        with st.status(status_label, expanded=True) as status:
            status.write("Loading & validating data")
            forecaster_blob = _train_forecaster_cached(load_path, rjpp_path, init_params, runtime_params, cache_sig)
            status.write("Assembling ensembles & calibration")
            forecaster = pickle.loads(forecaster_blob)
            for attr, value in runtime_params.items():
                setattr(forecaster, attr, value)
            status.write("Generating forecast, bands, and diagnostics")
            results = forecaster.run_post_training(start_dt, end_dt)
            final_label = "Forecast ready (cached)" if cache_hit else "Training complete"
            status.update(label=final_label, state="complete")
        
        results["max_dev"] = forecaster.max_dev
        st.session_state.results = results
        st.session_state.forecaster = forecaster
        st.session_state.fc_raw = results["forecast_raw"]
        st.session_state.combined = results["combined_forecast"]["Combined"]
        st.session_state.rjpp_full = results["rjpp_full"]
        st.session_state.filesig = st.session_state.pending_filesig
        st.session_state.last_training_sig = cache_sig
        st.session_state.training_params_dirty = False
        st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.fc_adjusted = None
        st.session_state.combined_adjusted = None
        st.session_state.planned_overlay = None
        st.session_state.ai_markdown = None
        st.session_state.ai_error = None
        st.success("Forecast completed successfully!")
        
    except Exception as e:
        st.error(f"Error during forecast computation: {str(e)}")
        st.error("Please check your input data format and try again.")
        if st.checkbox("Show detailed error information", value=False):
            st.code(traceback.format_exc())

# ----- Require a run before showing tabs -----
if st.session_state.results is None:
    st.info("Upload CSVs and click **Run / Refresh** to compute forecasts.")
    st.stop()

# Unpack session artifacts
results = st.session_state.results
forecaster = st.session_state.forecaster
fc = results["forecast"]
fc_raw = st.session_state.fc_raw
combined = st.session_state.combined
rjpp_full = st.session_state.rjpp_full
planned_overlay = None
scenario_records = st.session_state.scenarios
results["scenarios"] = scenario_records
fc_adj = None

forecast_start_ts = pd.to_datetime(_month_start(forecast_start)).to_period("M").to_timestamp()

if len(fc.index):
    assert (fc.index < forecast_start_ts).sum() == 0, (
        f"Forecast index contains values before {forecast_start_ts:%Y-%m}."
    )
if st.session_state.apply_scenarios and scenario_records and len(fc.index):
    scenario_objects = []
    for entry in scenario_records:
        clean_entry = _scenario_clean_dict(entry)
        try:
            scenario_objects.append(Scenario(**clean_entry))
        except Exception as exc:
            logger.warning(f"Skipping scenario {entry.get('name')}: {exc}")
    if scenario_objects:
        idx = fc.index
        custom_map = st.session_state.scenario_files
        planned_overlay = sum_scenarios(scenario_objects, idx, custom_map=custom_map)
        planned_overlay.name = "planned_overlay"
        rjpp_series = fc["RJPP"] if "RJPP" in fc.columns else None
        # Do not clamp scenario-adjusted forecast to RJPP. Scenarios represent
        # additional firm load; physically they should add to demand even if
        # they exceed RJPP envelopes. Compliance can be evaluated separately.
        fc_adj = apply_scenarios_to_bands(
            fc,
            planned_overlay,
            rjpp_series,
            forecaster.max_dev,
            clamp_to_rjpp=False,
        )
        base_adj_col = "p50_adj" if "p50_adj" in fc_adj.columns else ("Ensemble_adj" if "Ensemble_adj" in fc_adj.columns else None)
        combined_adj = st.session_state.combined.copy() if st.session_state.combined is not None else None
        if combined_adj is not None and base_adj_col:
            combined_adj = combined_adj.reindex(combined_adj.index.union(fc_adj.index))
            combined_adj.loc[fc_adj.index] = fc_adj[base_adj_col].values
            st.session_state.combined_adjusted = combined_adj
        else:
            st.session_state.combined_adjusted = None
        st.session_state.fc_adjusted = fc_adj
        st.session_state.planned_overlay = planned_overlay
        results["forecast_adjusted"] = fc_adj
        results["scenario_planned_overlay"] = planned_overlay
        scenario_set = ScenarioSet(scenario_objects)
        results["scenario_set"] = scenario_set
        results["scenario_exogenous_stub"] = scenario_set_to_future_exogenous(scenario_set, idx)
    else:
        st.session_state.fc_adjusted = None
        st.session_state.planned_overlay = None
        st.session_state.combined_adjusted = None
        results.pop("forecast_adjusted", None)
        results.pop("scenario_planned_overlay", None)
        results.pop("scenario_set", None)
        results.pop("scenario_exogenous_stub", None)
else:
    st.session_state.fc_adjusted = None
    st.session_state.planned_overlay = None
    st.session_state.combined_adjusted = None
    results.pop("forecast_adjusted", None)
    results.pop("scenario_planned_overlay", None)
    results.pop("scenario_set", None)
    results.pop("scenario_exogenous_stub", None)

fc_adj = st.session_state.fc_adjusted
planned_overlay = st.session_state.planned_overlay
val_results = results.get("validation_results",{})
weights = results.get("model_weights",{})
rjpp_proximity = results.get("rjpp_proximity", {})
naive_metrics = results.get("naive_holdout_metrics",{})
naive_curve = results.get("naive_holdout_forecast")
best_model = results.get("best_model")
val_index = results.get("val_index")
ens_val_pred = results.get("ensemble_val_pred")
ens_val_metrics = results.get("ensemble_val_metrics")
feat_imp = results.get("feature_importance",{})
ensemble_choice = results.get("ensemble_choice","weighted")
val_df_for_bundle = results.get("val_df")
model_status = results.get("model_status",{})
model_errors = results.get("model_errors",{})
val_bands = results.get("val_bands")
calibration_pct = results.get("calibration_pct")
validation_windows = results.get("validation_windows", {})

has_rjpp = forecaster.use_rjpp and ("RJPP" in fc.columns)

# Display options (from PRESETS, no UI controls)
show_bands = True  # Always show uncertainty bands
bounded_bands = PRESETS["clamp_bands_to_rjpp"]  # Default False

# ---------- Tabs ----------
tab_overview, tab_forecast, tab_models, tab_validation, tab_quality, tab_exports, tab_scenarios, tab_ai = st.tabs(
    ["Overview", "Forecast", "Models", "Validation", "Data Quality", "Exports & Logs", "Scenarios", "AI Insight"]
)

# ---------- Overview ----------
with tab_overview:
    base_col = "p50" if "p50" in fc.columns else "Ensemble"
    horizon = fc[base_col].astype(float) if len(fc) else pd.Series(dtype=float)
    hours = pd.Series(fc.index.days_in_month, index=fc.index, dtype=float) * 24 if len(fc) else pd.Series(dtype=float)
    energy_mwh = horizon * hours if len(horizon) else pd.Series(dtype=float)
    next_12_energy_gwh = float(energy_mwh.iloc[:12].sum() / 1000) if len(energy_mwh) else np.nan
    next_12_avg = float(horizon.iloc[:12].mean()) if len(horizon) else np.nan

    annual_mean_mw = horizon.groupby(fc.index.year).mean() if len(horizon) else pd.Series(dtype=float)
    annual_energy_gwh = energy_mwh.groupby(fc.index.year).sum() / 1000 if len(energy_mwh) else pd.Series(dtype=float)
    annual_peak_mw = horizon.groupby(fc.index.year).max() if len(horizon) else pd.Series(dtype=float)
    annual_summary = pd.DataFrame({"Average_MW": annual_mean_mw, "Energy_GWh": annual_energy_gwh, "Peak_MW": annual_peak_mw})
    if not annual_summary.empty:
        annual_summary["Load_Factor"] = annual_summary["Average_MW"] / annual_summary["Peak_MW"].replace(0, np.nan)
    annual_summary.index.name = "Year"

    adj_col = None
    adj_horizon = None
    adj_energy_mwh = None
    adj_next_12_avg = np.nan
    adj_next_12_energy_gwh = np.nan
    annual_mean_adj = pd.Series(dtype=float)
    if fc_adj is not None:
        if "p50_adj" in fc_adj.columns:
            adj_col = "p50_adj"
        elif "Ensemble_adj" in fc_adj.columns:
            adj_col = "Ensemble_adj"
        if adj_col:
            adj_horizon = fc_adj[adj_col].astype(float)
            if len(adj_horizon):
                adj_energy_mwh = adj_horizon * hours
                adj_next_12_avg = float(adj_horizon.iloc[:12].mean()) if len(adj_horizon) else np.nan
                adj_next_12_energy_gwh = float(adj_energy_mwh.iloc[:12].sum() / 1000) if adj_energy_mwh is not None else np.nan
                annual_mean_adj = adj_horizon.groupby(fc.index.year).mean()
            else:
                adj_energy_mwh = pd.Series(dtype=float)

    cagr = _calc_cagr_series(horizon)
    cagr_adj = _calc_cagr_series(adj_horizon) if adj_horizon is not None else None

    # Apply display mode filter for overview plot
    if display_mode == "History only":
        comb_for_plot = combined.loc[:HISTORY_CUTOFF] if combined is not None else pd.Series(dtype=float)
    elif display_mode == "Forecast only":
        comb_for_plot = pd.concat([combined.loc[:HISTORY_CUTOFF].tail(1), horizon]) if combined is not None and len(horizon) else horizon
    else:  # "Both"
        comb_for_plot = combined if combined is not None else pd.Series(dtype=float)

    def _fmt_val(value, fmt_str="{:,.0f}"):
        try:
            if value is None or not np.isfinite(value):
                return "n/a"
        except TypeError:
            return "n/a"
        return fmt_str.format(value)

    def _fmt_delta_value(base_value, adj_value, fmt=".1f", suffix=""):
        try:
            if base_value is None or adj_value is None:
                return None
            if not np.isfinite(base_value) or not np.isfinite(adj_value):
                return None
        except TypeError:
            return None
        diff = adj_value - base_value
        if abs(diff) < 1e-9:
            return None
        return f"{diff:+{fmt}}{suffix}"

    comp_rate = None
    breaches = 0
    worst_gap = np.nan
    worst_month = None
    worst_dev = None
    breach_first = None
    breach_last = None
    clamp_pct = np.nan
    clamped_months = 0
    first_clamp = None
    last_clamp = None
    anchor_rows = []
    if has_rjpp:
        dev = (fc["Ensemble"] - fc["RJPP"]) / fc["RJPP"] * 100
        comp_rate = (dev.abs() <= (forecaster.max_dev * 100)).mean() * 100
        breach_mask = dev.abs() > (forecaster.max_dev * 100)
        breaches = int(breach_mask.sum())
        if breaches:
            worst_month = dev.abs().idxmax()
            worst_dev = float(dev.loc[worst_month])
            breach_dates = breach_mask[breach_mask].index
            breach_first = breach_dates.min()
            breach_last = breach_dates.max()
        for y in sorted(fc.index.year.unique()):
            mask = fc.index.year == y
            yr_mean = fc.loc[mask, "Ensemble"].mean()
            dec_val = fc.loc[fc.index == pd.Timestamp(y, 12, 1), "RJPP"]
            if len(dec_val):
                rjpp_dec = float(dec_val.values[0])
                gap_pct = abs((yr_mean - rjpp_dec) / rjpp_dec) * 100 if rjpp_dec else np.nan
                if np.isfinite(gap_pct):
                    if gap_pct <= 1.0:
                        status = "Pass"
                    elif gap_pct <= 1.5:
                        status = "Watch"
                    else:
                        status = "Fail"
                else:
                    status = "n/a"
                anchor_rows.append({"Year": int(y), "Gap_%": gap_pct, "Status": status})
        if anchor_rows:
            finite_gaps = [row["Gap_%"] for row in anchor_rows if row["Gap_%"] is not None and np.isfinite(row["Gap_%"]) ]
            if finite_gaps:
                worst_gap = float(np.nanmax(finite_gaps))
        lo = fc["RJPP"] * (1 - forecaster.max_dev)
        hi = fc["RJPP"] * (1 + forecaster.max_dev)
        tol = np.maximum(fc["RJPP"].abs() * 0.002, 0.5)
        clamp_mask = ((fc["Ensemble"] - lo).abs() <= tol) | ((fc["Ensemble"] - hi).abs() <= tol)
        clamped_months = int(clamp_mask.sum())
        clamp_pct = float(clamp_mask.mean() * 100) if len(clamp_mask) else np.nan
        if clamped_months:
            clamp_indices = clamp_mask[clamp_mask].index
            first_clamp = clamp_indices.min()
            last_clamp = clamp_indices.max()

    comp_display = _fmt_val(comp_rate, "{:.1f}%") if comp_rate is not None else "n/a"
    comp_rate_adj = None
    if has_rjpp and adj_horizon is not None:
        comp_rate_adj = _compute_rjpp_compliance(adj_horizon, fc, forecaster.max_dev)
    comp_display_adj = _fmt_val(comp_rate_adj, "{:.1f}%") if comp_rate_adj is not None else "n/a"

    ensemble_rmse = None
    if ens_val_metrics and "RMSE" in ens_val_metrics:
        ensemble_rmse = ens_val_metrics["RMSE"]
    naive_rmse = naive_metrics.get("RMSE") if naive_metrics else None
    rmse_value = "n/a"
    rmse_delta = None
    if ensemble_rmse is not None and np.isfinite(ensemble_rmse):
        rmse_value = f"{ensemble_rmse:.2f}"
        if naive_rmse is not None and np.isfinite(naive_rmse) and naive_rmse > 0:
            rmse_delta = f"{((naive_rmse - ensemble_rmse)/naive_rmse)*100:+.1f}%"

    band_display = _fmt_val(calibration_pct, "{:.1f}%")

    first_metrics = []
    first_metrics.append(("Next-12m Energy (GWh)", _fmt_val(next_12_energy_gwh, "{:,.1f}"), _fmt_delta_value(next_12_energy_gwh, adj_next_12_energy_gwh, ".1f", " GWh") if adj_horizon is not None else None))
    first_metrics.append(("Next-12m Avg (p50)", _fmt_val(next_12_avg, "{:,.0f}"), _fmt_delta_value(next_12_avg, adj_next_12_avg, ".0f", " MW") if adj_horizon is not None else None))
    if has_rjpp:
        comp_delta = _fmt_delta_value(comp_rate, comp_rate_adj, ".1f", "%") if comp_rate_adj is not None else None
        first_metrics.append(("RJPP Compliance", comp_display, comp_delta))
        clamp_display = "n/a"
        if clamped_months and np.isfinite(clamp_pct):
            clamp_display = f"{clamped_months} ({clamp_pct:.1f}%)"
        elif np.isfinite(clamp_pct):
            clamp_display = "0"
        first_metrics.append(("Clamped Months", clamp_display, None))
    else:
        cagr_display = _fmt_val(cagr * 100, "{:.2f}%") if cagr is not None else "n/a"
        cagr_delta = _fmt_delta_value(cagr * 100 if cagr is not None else np.nan, cagr_adj * 100 if cagr_adj is not None else np.nan, ".2f", "%") if cagr_adj is not None else None
        first_metrics.append(("Horizon CAGR", cagr_display, cagr_delta))

    second_metrics = []
    if has_rjpp:
        cagr_display = _fmt_val(cagr * 100, "{:.2f}%") if cagr is not None else "n/a"
        cagr_delta = _fmt_delta_value(cagr * 100 if cagr is not None else np.nan, cagr_adj * 100 if cagr_adj is not None else np.nan, ".2f", "%") if cagr_adj is not None else None
        second_metrics.append(("Horizon CAGR", cagr_display, cagr_delta))
        second_metrics.append(("Band Calibration (p10-90)", band_display, None))
        second_metrics.append(("Validation RMSE", rmse_value, rmse_delta))
    else:
        second_metrics.append(("Band Calibration (p10-90)", band_display, None))
        second_metrics.append(("Validation RMSE", rmse_value, rmse_delta))

    trend_msgs = []
    if cagr is not None and np.isfinite(cagr):
        trend_msgs.append(_trend_label_from_cagr(cagr))
    if cagr_adj is not None and np.isfinite(cagr_adj):
        trend_msgs.append(f"Scenario {_trend_label_from_cagr(cagr_adj)}")

    st.subheader("Main forecast")
    if trend_msgs:
        st.caption("Trend snapshot: " + " | ".join(trend_msgs))

    plot_hero(
        comb_for_plot.to_frame("Combined"),
        fc,
        rjpp_full if has_rjpp else None,
        key_suffix="overview",
        show_bands=False,
        bounded=bounded_bands,
        max_dev=forecaster.max_dev,
        fc_adj=fc_adj,
        combined_adj=st.session_state.combined_adjusted,
        planned_overlay=planned_overlay,
        forecast_start=forecast_start_ts,
    )
    # Explain for hero chart
    with st.expander("ℹ Explain", expanded=False):
        txt = GLOSSARY.get("Band calibration (p10–p90)") or "Forecast bands show p10–p90 uncertainty."
        if has_rjpp:
            txt = txt + " " + (GLOSSARY.get("RJPP compliance") or "RJPP shows regulator baseline.")
        st.write(txt)

    primary_metrics = []
    primary_metrics.append(("Next-12m Energy (GWh)", _fmt_val(next_12_energy_gwh, "{:,.1f}"), _fmt_delta_value(next_12_energy_gwh, adj_next_12_energy_gwh, ".1f", " GWh") if adj_horizon is not None else None))
    primary_metrics.append(("Next-12m Avg (p50)", _fmt_val(next_12_avg, "{:,.0f}"), _fmt_delta_value(next_12_avg, adj_next_12_avg, ".0f", " MW") if adj_horizon is not None else None))
    if has_rjpp:
        comp_delta = _fmt_delta_value(comp_rate, comp_rate_adj, ".1f", "%") if comp_rate_adj is not None else None
        primary_metrics.append(("RJPP Compliance", comp_display, comp_delta))

    st.markdown("### Primary KPIs")
    cols_primary = st.columns(len(primary_metrics)) if primary_metrics else []
    for col, (label, value, delta) in zip(cols_primary, primary_metrics):
        if delta is not None:
            show_metric(col, label, value, delta=delta, explain_on=explain_on)
        else:
            show_metric(col, label, value, explain_on=explain_on)

    secondary_metrics = []
    if has_rjpp:
        cagr_display = _fmt_val(cagr * 100, "{:.2f}%") if cagr is not None else "n/a"
        cagr_delta = _fmt_delta_value(cagr * 100 if cagr is not None else np.nan, cagr_adj * 100 if cagr_adj is not None else np.nan, ".2f", "%") if cagr_adj is not None else None
        secondary_metrics.append(("Horizon CAGR", cagr_display, cagr_delta))
        secondary_metrics.append(("Band Calibration (p10-90)", band_display, None))
        secondary_metrics.append(("Validation RMSE", rmse_value, rmse_delta))
        worst_gap_display = _fmt_val(worst_gap, "{:.2f}%") if np.isfinite(worst_gap) else "n/a"
        secondary_metrics.append(("Worst Annual Gap", worst_gap_display, None))
        clamp_display = "n/a"
        if clamped_months and np.isfinite(clamp_pct):
            clamp_display = f"{clamped_months} ({clamp_pct:.1f}%)"
        elif np.isfinite(clamp_pct):
            clamp_display = "0"
        secondary_metrics.append(("Clamped Months", clamp_display, None))
    else:
        cagr_display = _fmt_val(cagr * 100, "{:.2f}%") if cagr is not None else "n/a"
        cagr_delta = _fmt_delta_value(cagr * 100 if cagr is not None else np.nan, cagr_adj * 100 if cagr_adj is not None else np.nan, ".2f", "%") if cagr_adj is not None else None
        secondary_metrics.append(("Horizon CAGR", cagr_display, cagr_delta))
        secondary_metrics.append(("Band Calibration (p10-90)", band_display, None))
        secondary_metrics.append(("Validation RMSE", rmse_value, rmse_delta))

    with st.expander("More KPIs", expanded=False):
        if secondary_metrics:
            for chunk_start in range(0, len(secondary_metrics), 3):
                cols_sec = st.columns(min(3, len(secondary_metrics) - chunk_start))
                for col, (label, value, delta) in zip(cols_sec, secondary_metrics[chunk_start:chunk_start + 3]):
                    if delta is not None:
                        show_metric(col, label, value, delta=delta, explain_on=explain_on)
                    else:
                        show_metric(col, label, value, explain_on=explain_on)
        else:
            st.caption("No secondary KPIs available.")

    caption_bits = []
    caption_bits.append(f"Ensemble selection: **{ensemble_choice}**")
    if has_rjpp and comp_rate is not None:
        compliance_text = f"RJPP compliance base **{comp_display}**"
        if comp_rate_adj is not None and np.isfinite(comp_rate_adj):
            compliance_text += f" | scenario **{comp_display_adj}**"
        caption_bits.append(compliance_text)
    if has_rjpp and clamped_months:
        clamp_window = f"{first_clamp:%Y-%m} – {last_clamp:%Y-%m}" if first_clamp and last_clamp else "multiple months"
        clamp_pct_display = _fmt_val(clamp_pct, "{:.1f}%")
        caption_bits.append(f"Clamped months **{clamped_months}** (~{clamp_pct_display}) {clamp_window}")
    if has_rjpp and np.isfinite(worst_gap):
        caption_bits.append(f"Worst annual gap **{_fmt_val(worst_gap, '{:.2f}%')}**")
    st.caption("; ".join(caption_bits))

    kpi_lines = [
        "# Key KPIs",
        f"- Next-12m Energy (GWh): {fmt_gwh(next_12_energy_gwh)}",
        f"- Next-12m Avg (p50): {fmt_mw(next_12_avg)}",
        f"- RJPP Compliance: {fmt_pct(comp_rate if comp_rate is not None else np.nan)}",
        f"- Clamped Months: {clamped_months} ({fmt_pct(clamp_pct)})",
        f"- Band Calibration (p10-90): {fmt_pct(calibration_pct if calibration_pct is not None else np.nan)}",
    ]
    if ensemble_rmse is not None:
        kpi_lines.append(f"- Validation RMSE: {ensemble_rmse:.2f}")
    else:
        kpi_lines.append("- Validation RMSE: n/a")
    kpi_md = "\n".join(kpi_lines)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Copy KPIs"):
            try:
                import importlib
                pyperclip = importlib.import_module("pyperclip")
                pyperclip.copy(kpi_md)
                st.success("KPIs copied to clipboard")
            except Exception:
                st.info("KPIs available for download")
                st.download_button("Download KPIs (.md)", kpi_md, f"kpis_{datetime.now().strftime('%Y-%m-%d')}.md", "text/markdown")
    with c2:
        if st.button("Reset to defaults"):
            for k in ["display_mode", "show_bands", "bounded_bands", "explain_mode"]:
                if k in st.session_state:
                    st.session_state.pop(k, None)
            st.experimental_rerun()

    anchor_df = pd.DataFrame(anchor_rows).set_index("Year") if anchor_rows else pd.DataFrame()
    if not annual_summary.empty or (has_rjpp and not anchor_df.empty):
        st.subheader("Annual Summary & Health Checks")
        col_summary, col_health = st.columns([2, 1])
        if not annual_summary.empty:
            col_summary.dataframe(
                annual_summary.reset_index().style.format({
                    "Average_MW": "{:,.0f}",
                    "Energy_GWh": "{:,.1f}",
                    "Peak_MW": "{:,.0f}",
                    "Load_Factor": "{:.3f}"
                }),
                use_container_width=True,
            )
        if has_rjpp and not anchor_df.empty:
            show_metric(col_health, "Worst Annual Gap", _fmt_val(worst_gap, "{:.2f}%"), explain_on=explain_on)
            col_health.dataframe(anchor_df.style.format({"Gap_%": "{:.2f}"}), use_container_width=True)
        else:
            col_health.caption("No RJPP anchor diagnostics available.")

with tab_forecast:
    st.subheader("Forecast")
    plot_hero(
        combined.to_frame("Combined"),
        fc,
        rjpp_full if has_rjpp else None,
        key_suffix="forecast",
        show_bands=show_bands,
        bounded=bounded_bands,
        max_dev=forecaster.max_dev,
        fc_adj=fc_adj,
        combined_adj=st.session_state.combined_adjusted,
        planned_overlay=planned_overlay,
        forecast_start=forecast_start_ts,
    )
    with st.expander("ℹ Explain", expanded=False):
        st.write(GLOSSARY.get("Band calibration (p10–p90)") or "Forecast shows p50 with p10–p90 uncertainty bands. Units: MW.")

    # Annual Delta Chart (Base vs Scenario-Adjusted)
    if fc_adj is not None and len(fc.index):
        base_col = "p50" if "p50" in fc.columns else "Ensemble"
        adj_col = "p50_adj" if "p50_adj" in fc_adj.columns else "Ensemble_adj"
        if adj_col in fc_adj.columns:
            st.subheader("Annual Impact: Scenario vs Base")
            
            # Calculate annual means
            fc_annual = fc.groupby(fc.index.year)[base_col].mean()
            fc_adj_annual = fc_adj.groupby(fc_adj.index.year)[adj_col].mean()
            
            annual_delta_mw = fc_adj_annual - fc_annual.reindex(fc_adj_annual.index)
            annual_delta_pct = (annual_delta_mw / fc_annual.reindex(fc_adj_annual.index) * 100).replace([np.inf, -np.inf], np.nan)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_delta_mw = go.Figure(layout=PLOTLY_BASE_LAYOUT)
                colors = ['green' if x >= 0 else 'red' for x in annual_delta_mw.values]
                fig_delta_mw.add_trace(go.Bar(
                    x=annual_delta_mw.index.astype(str),
                    y=annual_delta_mw.values,
                    marker_color=colors,
                    name="Delta MW"
                ))
                fig_delta_mw.update_yaxes(title_text="Annual Average Delta (MW)")
                fig_delta_mw.update_layout(title_text="Scenario Impact (MW)")
                st.plotly_chart(fig_delta_mw, use_container_width=True, key="annual_delta_mw")
            
            with col2:
                fig_delta_pct = go.Figure(layout=PLOTLY_BASE_LAYOUT)
                colors = ['green' if x >= 0 else 'red' for x in annual_delta_pct.dropna().values]
                fig_delta_pct.add_trace(go.Bar(
                    x=annual_delta_pct.dropna().index.astype(str),
                    y=annual_delta_pct.dropna().values,
                    marker_color=colors,
                    name="Delta %"
                ))
                fig_delta_pct.update_yaxes(title_text="Annual Average Delta (%)", ticksuffix=" %")
                fig_delta_pct.update_layout(title_text="Scenario Impact (%)")
                st.plotly_chart(fig_delta_pct, use_container_width=True, key="annual_delta_pct")

    if has_rjpp and "Ensemble" in fc:
        st.subheader("RJPP Deviation by Month (signed)")
        dev = (fc["Ensemble"] - fc["RJPP"]) / fc["RJPP"] * 100
        highlight_top = st.toggle("Show only worst deviations", value=False, key="rjpp_highlight_top")
        plot_series = dev
        title_suffix = ""
        if highlight_top:
            top_n = st.selectbox("Top N months", options=[6, 12, 24], index=0, key="rjpp_top_n")
            selected_index = dev.abs().sort_values(ascending=False).index[:top_n]
            plot_series = dev.loc[selected_index].sort_index()
            title_suffix = f" (top {top_n})"
        colors = [PALETTE["good"] if abs(v) <= forecaster.max_dev * 100 else (PALETTE["warn"] if abs(v) <= forecaster.max_dev * 150 else PALETTE["bad"]) for v in plot_series]
        figd = go.Figure(layout=PLOTLY_BASE_LAYOUT)
        figd.add_trace(go.Bar(x=plot_series.index, y=plot_series.values, marker_color=colors, name="Deviation %"))
        figd.add_hline(y=forecaster.max_dev*100, line_dash="dot", line_color="red", annotation_text="+/- max_dev")
        figd.add_hline(y=-forecaster.max_dev*100, line_dash="dot", line_color="red")
        figd.update_yaxes(ticksuffix=" %", title_text="Deviation (%)")
        figd.update_layout(title_text=f"RJPP Deviation by Month{title_suffix}")
        st.plotly_chart(figd, use_container_width=True, key="rjpp_dev_month_bar")
        with st.expander("ℹ Explain", expanded=False):
            st.write("Bars show monthly % deviation vs RJPP. Dotted lines are +/- bound. Colors: green within bound, amber near limit, red outside.")


# ---------- Models ----------
with tab_models:
    st.subheader("Training controls")
    prev_params = st.session_state.training_params.copy()
    col_tc1, col_tc2, col_tc3, col_tc4 = st.columns(4)
    val_months_sel = col_tc1.slider(
        "Validation months",
        min_value=6,
        max_value=36,
        value=int(prev_params.get("val_months", DEFAULT_VAL_MONTHS)),
        step=1,
    )
    rolling_k_sel = col_tc2.slider(
        "Rolling windows (K)",
        min_value=1,
        max_value=6,
        value=int(prev_params.get("rolling_k", DEFAULT_ROLLING_K)),
        step=1,
    )
    mom_quant_sel = col_tc3.slider(
        "MoM cap quantile",
        min_value=0.80,
        max_value=0.98,
        value=float(prev_params.get("mom_limit_quantile", 0.90)),
        step=0.01,
    )
    max_dev_sel = col_tc4.slider(
        "RJPP ± deviation",
        min_value=0.01,
        max_value=0.10,
        value=float(prev_params.get("max_dev", DEFAULT_MAX_DEV)),
        step=0.005,
        format="%.3f",
    )
    st.session_state.training_params.update(
        {
            "val_months": int(val_months_sel),
            "rolling_k": int(rolling_k_sel),
            "mom_limit_quantile": float(round(mom_quant_sel, 2)),
            "max_dev": float(round(max_dev_sel, 3)),
        }
    )
    if st.session_state.training_params != prev_params:
        st.session_state.training_params_dirty = True
    if st.session_state.training_params_dirty:
        st.info("Training settings changed. Re-run the forecast to apply updates.")

    st.subheader("Per-Model Forecasts (horizon only)")
    st.caption("Raw model outputs before RJPP alignment, noise, and clipping.")
    plot_per_model(combined, fc, fc_raw, rjpp_full if has_rjpp else None, forecaster.slot_info)

    metric_options = {"RMSE": "RMSE", "MAE": "MAE", "sMAPE": "sMAPE"}
    st.session_state.setdefault("leaderboard_metric", "RMSE")
    selected_metric_label = st.session_state.get("leaderboard_metric", "RMSE")
    st.subheader(f"Model Leaderboard ({selected_metric_label})")
    st.caption("Sorted ascending; lower values indicate better validation fit.")
    with st.expander("More metrics", expanded=False):
        st.markdown("Switch the leaderboard metric or review metric definitions.")
        selected_metric_label = st.radio(
            "Leaderboard metric",
            list(metric_options.keys()),
            index=list(metric_options.keys()).index(st.session_state.get("leaderboard_metric", "RMSE")),
            key="leaderboard_metric",
        )
        gloss_lines = []
        for label, column in metric_options.items():
            gloss_key = _find_glossary_key(column)
            desc = GLOSSARY.get(gloss_key) if gloss_key else None
            if desc:
                gloss_lines.append(f"- **{label}**: {desc}")
        if gloss_lines:
            st.markdown("\n".join(gloss_lines))
    metric_key = metric_options[st.session_state.get("leaderboard_metric", "RMSE")]
    if has_rjpp and rjpp_proximity:
        prox_df = pd.DataFrame(rjpp_proximity).T.rename(columns={"mean_signed_pct": "Mean signed %", "pct_within_5": "% within \u00b15%"})
        st.caption("RJPP proximity diagnostics (raw model outputs)")
        st.dataframe(prox_df.style.format({"Mean signed %": "{:.2f}%", "% within \u00b15%": "{:.1f}%"}), use_container_width=True)
    if val_results:
        show_baseline = st.checkbox("Show baseline (Naive) in leaderboard", value=False, key="show_baseline_leaderboard")
        dfv = pd.DataFrame(val_results).T
        if not show_baseline and "Naive" in dfv.index:
            dfv = dfv.drop(index="Naive")
        if not dfv.empty:
            sort_metric = metric_key if metric_key in dfv.columns else "RMSE"
            dfv = dfv.sort_values(sort_metric)
            chart_df = dfv.head(5).reset_index()
            y_axis = sort_metric
            chart_df["Model"] = chart_df["index"].map(lambda name: forecaster.slot_info.get(name, {}).get("impl", name))
            st.plotly_chart(
                px.bar(chart_df, x="Model", y=y_axis, labels={"Model": "Model", y_axis: y_axis}),
                use_container_width=True,
                key="model_leaderboard_bar",
            )
            impl_col = [forecaster.slot_info.get(idx, {}).get("impl", "") for idx in dfv.index]
            dfv.insert(0, "Implementation", impl_col)
            numeric_cols = dfv.select_dtypes(include=[np.number]).columns
            st.dataframe(dfv.style.format({col: "{:.3f}" for col in numeric_cols}), use_container_width=True)
        else:
            st.info("No validation metrics available after filters.")
    else:
        st.info("No validation metrics available.")

    st.subheader("Model Matrix & Ensemble Participation")
    presence = []
    for m in MODEL_SLOTS:
        impl = forecaster.slot_info.get(m, {}).get("impl", "")
        raw_status = forecaster.slot_info.get(m, {}).get("status", forecaster.model_status.get(m, "unknown"))
        if raw_status == "unavailable_fallback_naive":
            status_label = "Fallback: Seasonal Naive"
        elif raw_status == "fallback":
            status_label = "Fallback"
        else:
            status_label = raw_status
        presence.append({
            "Slot": m,
            "Model": "SARIMA(X)" if m == "SARIMAX" else ("Regression (Ridge)" if m == "Ridge" else m),
            "Implementation": impl,
            "Status": status_label,
            "Available": raw_status != "unavailable",
            "Used in Ensemble": bool(weights.get(m, 0.0) > 0),
            "Weight": weights.get(m, 0.0),
            "Val RMSE": val_results.get(m, {}).get("RMSE"),
            "RJPP dev %": val_results.get(m, {}).get("RJPP_dev%"),
            "Corr vs Ensemble": (
                fc["Ensemble"].reindex(fc_raw.index).corr(fc_raw[m])
                if m in fc_raw.columns and len(fc_raw) > 1
                else np.nan
            ),
            "Error": model_errors.get(m, ""),
            "_raw_status": raw_status,
        })
    model_matrix = pd.DataFrame(presence)
    if "Error" in model_matrix.columns:
        model_matrix["Error"] = model_matrix["Error"].replace({"": np.nan})
    formatter = {"Val RMSE": "{:.2f}", "Corr vs Ensemble": "{:.2f}", "Weight": "{:.2f}"}
    if has_rjpp and "RJPP dev %" in model_matrix.columns:
        formatter["RJPP dev %"] = "{:.2f}"
    elif "RJPP dev %" in model_matrix.columns:
        model_matrix = model_matrix.drop(columns=["RJPP dev %"])
    ordered_cols = ["Slot", "Implementation", "Status", "Available", "Used in Ensemble", "Weight", "Val RMSE"]
    if has_rjpp and "RJPP dev %" in model_matrix.columns:
        ordered_cols.append("RJPP dev %")
    ordered_cols += ["Corr vs Ensemble", "Error"]
    ordered_cols = [c for c in ordered_cols if c in model_matrix.columns]
    st.dataframe(model_matrix[ordered_cols].style.format(formatter), use_container_width=True)

    raw_status_series = model_matrix["_raw_status"] if "_raw_status" in model_matrix.columns else pd.Series(dtype=str)
    fallback_naive_slots = model_matrix.loc[raw_status_series == "unavailable_fallback_naive", "Slot"].tolist()
    if fallback_naive_slots:
        names = ", ".join(fallback_naive_slots)
        st.warning(
            f"Seasonal naive fallback active for: {names}. Install LightGBM (`pip install lightgbm`) to unlock gradient boosting models.",
            icon="⚠️",
        )

    optional_slots = ["Prophet", "LGB", "XGB"]
    missing = [m for m in optional_slots if forecaster.slot_info.get(m, {}).get("status") == "unavailable"]
    fallbacks = [m for m in optional_slots if forecaster.slot_info.get(m, {}).get("status", "").startswith("fallback")]
    if missing:
        install_map = {"Prophet": "prophet", "LGB": "lightgbm", "XGB": "xgboost"}
        needed = sorted({install_map.get(m, m.lower()) for m in missing})
        st.info(
            "Optional models unavailable: " + ", ".join(missing) + f". Install with: `pip install {' '.join(needed)}`"
        )
    elif fallbacks:
        labels = [forecaster.slot_info.get(m, {}).get("impl", m) for m in fallbacks]
        st.caption("Fallback implementations in use: " + ", ".join(labels))

    st.markdown("**Preview custom weighted ensemble (horizon)**")
    available_weight_models = [m for m in MODEL_SLOTS if m in fc_raw.columns]
    if available_weight_models:
        cols = st.columns(len(available_weight_models)) if len(available_weight_models) <= 4 else None
        custom_weights = {}
        for idx, m in enumerate(available_weight_models):
            impl = forecaster.slot_info.get(m, {}).get("impl", "")
            label = impl if impl else ("SARIMA(X)" if m == "SARIMAX" else ("Regression (Ridge)" if m == "Ridge" else m))
            slider_args = {"label": label, "min_value": 0.0, "max_value": 1.0, "value": float(round(weights.get(m, 0.0), 2)), "key": f"w_{m}", "step": 0.05}
            if cols:
                with cols[idx]:
                    custom_weights[m] = st.slider(**slider_args)
            else:
                custom_weights[m] = st.slider(**slider_args)
        if st.button("Preview reweighted ensemble", key="preview_reweight"):
            total = sum(custom_weights.values())
            if total <= 0:
                st.warning("Assign at least one positive weight to preview the ensemble.")
            else:
                norm_weights = {m: w / total for m, w in custom_weights.items()}
                ens = np.zeros(len(fc_raw.index))
                for m, w in norm_weights.items():
                    ens += w * np.nan_to_num(fc_raw[m].values)
                fc_re = fc.copy()
                if has_rjpp and "RJPP" in fc_re.columns:
                    fc_re["Ensemble"] = np.clip(ens, fc_re["RJPP"] * (1 - forecaster.max_dev), fc_re["RJPP"] * (1 + forecaster.max_dev))
                else:
                    fc_re["Ensemble"] = np.maximum(0.0, ens)
                st.caption("Weighted preview uses sliders above (normalized to 1.0).")
                plot_hero(
                    combined.to_frame("Combined"),
                    fc_re,
                    rjpp_full if has_rjpp else None,
                    key_suffix="reweighted",
                    show_bands=show_bands,
                    bounded=bounded_bands,
                    max_dev=forecaster.max_dev,
                    forecast_start=forecast_start_ts,
                )
    else:
        st.caption("Trained model predictions are required to preview custom weights.")

    with st.expander("Overlay selected model outputs (optional)", expanded=False):
        st.caption("Overlay lets you visually compare chosen raw model outputs to the ensemble.")
        available_models = [m for m in MODEL_SLOTS if m in fc_raw.columns]
        if available_models:
            default_overlay_slots: List[str] = []
            if val_results:
                val_df_sorted = pd.DataFrame(val_results).T
                if "RMSE" in val_df_sorted.columns:
                    val_df_sorted = val_df_sorted.sort_values("RMSE")
                    val_df_sorted = val_df_sorted.loc[val_df_sorted.index != "Naive"]
                    default_overlay_slots = [m for m in val_df_sorted.index if m in available_models][:2]
            if not default_overlay_slots:
                default_overlay_slots = available_models[:2]
            overlay_labels = {}
            for m in available_models:
                base_label = forecaster.slot_info.get(m, {}).get("impl")
                if not base_label:
                    base_label = "SARIMA(X)" if m == "SARIMAX" else ("Regression (Ridge)" if m == "Ridge" else m)
                overlay_labels[m] = f"{base_label} [{m}]"
            label_list = [overlay_labels[m] for m in available_models]
            default_labels = [overlay_labels[m] for m in default_overlay_slots]
            chosen_labels = st.multiselect("Overlay models", label_list, default=default_labels, key="overlay_pick")
            chosen_slots = [m for m, lbl in overlay_labels.items() if lbl in chosen_labels]
            if chosen_slots:
                fig_overlay = go.Figure(layout=PLOTLY_BASE_LAYOUT)
                shade_start = fc.index.min() if len(fc.index) else None
                if shade_start is not None:
                    shaded(fig_overlay, shade_start, fc.index.max())
                for m in chosen_slots:
                    fig_overlay.add_trace(
                        go.Scatter(
                            x=fc_raw.index,
                            y=fc_raw[m],
                            name=overlay_labels[m],
                            mode="lines",
                            line=dict(width=1.8),
                        )
                    )
                fig_overlay.add_trace(go.Scatter(x=fc.index, y=fc["Ensemble"], name="Ensemble", line=dict(width=3, color=PALETTE["ensemble"])))
                if shade_start is not None:
                    fig_overlay.add_vline(x=shade_start, line_width=1, line_dash="dot", line_color="orange")
                fig_overlay.update_yaxes(ticksuffix=" MW", title_text="Load (MW)")
                st.plotly_chart(fig_overlay, use_container_width=True, key="overlay_models")
        else:
            st.info("No trained model predictions available for overlay.")

    with st.expander("Correlation heatmap (advanced)", expanded=False):
        cols_for_corr = [c for c in MODEL_SLOTS if c in fc_raw.columns] + ["Ensemble"]
        if len(cols_for_corr) >= 2:
            st.caption("Correlation of model outputs across the forecast horizon.")
            corr_heatmap(pd.concat([fc_raw[[c for c in cols_for_corr if c != "Ensemble"]], fc["Ensemble"]], axis=1), cols_for_corr)
        else:
            st.info("Need at least two model outputs to compute correlations.")

    if "RF" in fc_raw.columns and "XGB" in fc_raw.columns:
        common_idx = fc_raw.index
        rf_aligned = fc_raw["RF"].reindex(common_idx)
        xgb_aligned = fc_raw["XGB"].reindex(common_idx)
        c = float(rf_aligned.corr(xgb_aligned)) if len(common_idx) > 1 else np.nan
        if np.isfinite(c) and c >= 0.98:
            st.warning(f"High similarity detected between RF and XGB (corr={c:.3f}). Check feature leakage or scaler reuse.")

# ---------- Validation ----------
with tab_validation:
    st.subheader("MAE by Month")
    mae_rendered = False
    if (
        ens_val_metrics
        and ens_val_pred is not None
        and val_df_for_bundle is not None
        and val_index is not None
        and "Average_Load" in val_df_for_bundle
        and hasattr(ens_val_pred, "reindex")
    ):
        actual = val_df_for_bundle["Average_Load"].reindex(val_index)
        forecast_err = (actual - ens_val_pred.reindex(val_index)).dropna()
        if len(forecast_err):
            mo = forecast_err.groupby(forecast_err.index.month).apply(lambda s: float(np.mean(np.abs(s))))
            fig_mae = px.bar(mo, labels={"value": "MAE", "index": "Month"})
            fig_mae.update_layout(PLOTLY_BASE_LAYOUT | {"title": "Monthly validation MAE"})
            fig_mae.update_yaxes(ticksuffix=" MW")
            st.plotly_chart(fig_mae, use_container_width=True, key="val_mae_moy_bar")
            st.caption("Validation mean absolute error by calendar month (lower is better).")
            mae_rendered = True
    if not mae_rendered:
        st.info("MAE by Month becomes available after running validation with holdout windows.")

    if ens_val_metrics and ens_val_pred is not None:
        naive_rmse_val = naive_metrics.get("RMSE") if naive_metrics else None
        has_naive_rmse = naive_rmse_val is not None and np.isfinite(naive_rmse_val)
        columns = st.columns(6) if has_naive_rmse else st.columns(5)
        c1, c2, c3, c4, c5 = columns[:5]
        show_metric(c1, "Ensemble RMSE", f"{ens_val_metrics['RMSE']:.2f}", explain_on=st.session_state.get("explain_mode", False))
        show_metric(c2, "MAE", f"{ens_val_metrics['MAE']:.2f}", explain_on=st.session_state.get("explain_mode", False))
        show_metric(c3, "MAPE", f"{ens_val_metrics['MAPE']:.2f}%", explain_on=st.session_state.get("explain_mode", False))
        show_metric(c4, "WAPE", f"{ens_val_metrics['WAPE']:.2f}%", explain_on=st.session_state.get("explain_mode", False))
        show_metric(c5, "sMAPE", f"{ens_val_metrics['sMAPE']:.2f}%", explain_on=st.session_state.get("explain_mode", False))
        if has_naive_rmse:
            delta_pct = ((naive_rmse_val - ens_val_metrics['RMSE']) / naive_rmse_val) * 100 if naive_rmse_val else np.nan
            label = f"{delta_pct:+.1f}%" if np.isfinite(delta_pct) else "n/a"
            show_metric(columns[5], "RMSE vs Naive", label, explain_on=st.session_state.get("explain_mode", False))

        cal = calibration_pct
        if cal is not None and np.isfinite(cal):
            if 70 <= cal <= 90:
                show_metric(st, "Calibration (p10-90 hit rate)", f"{cal:.1f}%", explain_on=st.session_state.get("explain_on", False))
            elif 60 <= cal < 70 or 90 < cal <= 95:
                st.markdown(f"<div style='color:#F59E0B;font-weight:600'>Calibration (p10-90): {cal:.1f}% - tune bands</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:#EF4444;font-weight:700'>Calibration (p10-90): {cal:.1f}% - revisit band construction</div>", unsafe_allow_html=True)

    if val_bands is not None and not val_bands.empty and val_df_for_bundle is not None and "Average_Load" in val_df_for_bundle:
        st.markdown("#### Band coverage by quarter")
        plot_reliability(
            val_df_for_bundle["Average_Load"].reindex(val_bands.index),
            val_bands["p10"],
            val_bands["p90"],
        )

    with st.expander("Rolling Window Metrics (averaged)", expanded=False):
        if val_results:
            st.caption("Average validation metrics across rolling windows (lower is better).")
            st.dataframe(pd.DataFrame(val_results).T.style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("No validation metrics available.")

    with st.expander("Rolling metrics trend", expanded=False):
        if validation_windows:
            st.caption("Trend of validation errors over time; default metric is RMSE.")
            metric_choice = st.selectbox("Metric", ["RMSE", "MAE", "MAPE", "WAPE", "sMAPE"], index=0, key="trend_metric")
            available_models_trend = list(validation_windows.keys())
            default_models = available_models_trend[:3] if len(available_models_trend) >= 3 else available_models_trend
            selected_models_trend = st.multiselect("Models", available_models_trend, default=default_models, key="trend_models")
            trend_frames = []
            for m in selected_models_trend:
                dfm = validation_windows.get(m)
                if dfm is None or metric_choice not in dfm.columns:
                    continue
                temp = dfm.dropna(subset=["window_end", metric_choice]).copy()
                if temp.empty:
                    continue
                label = forecaster.slot_info.get(m, {}).get("impl", None)
                if not label:
                    label = "SARIMA(X)" if m == "SARIMAX" else ("Regression (Ridge)" if m == "Ridge" else m)
                temp["Model"] = label
                trend_frames.append(temp[["window_end", metric_choice, "Model"]])
            if trend_frames:
                trend_df = pd.concat(trend_frames).sort_values("window_end")
                fig_trend = px.line(trend_df, x="window_end", y=metric_choice, color="Model")
                fig_trend.update_layout(PLOTLY_BASE_LAYOUT)
                fig_trend.update_traces(mode="lines+markers")
                st.plotly_chart(fig_trend, use_container_width=True, key="val_trend")
            else:
                st.info("No rolling metrics available for the selected models.")
        else:
            st.info("Rolling window diagnostics will appear after validation is computed.")


# ---------- Data Quality ----------
with tab_quality:
    dq_summary = getattr(forecaster, "data_quality_summary", {}) or {}
    dq_cols = st.columns(4)
    explain_flag = st.session_state.get("explain_mode", False)
    show_metric(dq_cols[0], "Negative values", dq_summary.get("negatives", 0), explain_on=explain_flag)
    show_metric(dq_cols[1], "Missing values", dq_summary.get("missing", 0), explain_on=explain_flag)
    show_metric(dq_cols[2], "Duplicate dates", dq_summary.get("dupe_dates", 0), explain_on=explain_flag)
    show_metric(dq_cols[3], "Near-zero entries", dq_summary.get("near_zero", 0), explain_on=explain_flag)
    if dq_summary.get("negatives", 0):
        st.warning(f"Detected {dq_summary['negatives']} negative load values; they were clipped during preprocessing.")
    if dq_summary.get("dupe_dates", 0):
        st.warning(f"Found {dq_summary['dupe_dates']} duplicated timestamps; consider deduplicating upstream.")
    if dq_summary.get("near_zero", 0):
        st.info(f"{dq_summary['near_zero']} entries are near zero (<1e-6 MW); double-check units.")
    src = forecaster.load_df["Average_Load"]
    full = pd.date_range(src.index.min(), src.index.max(), freq="MS")
    missing = sorted(list(set(full) - set(src.index)))
    dups = int(src.index.duplicated().sum())
    z = (src - src.rolling(12, min_periods=6).median())/src.rolling(12, min_periods=6).std()
    outliers = int((z.abs()>3).sum())
    a,b,c = st.columns(3)
    show_metric(a, "Missing Months", len(missing), explain_on=st.session_state.get("explain_mode", False))
    show_metric(b, "Duplicate Timestamps", dups, explain_on=st.session_state.get("explain_mode", False))
    show_metric(c, "Potential Outliers (|z|>3)", outliers, explain_on=st.session_state.get("explain_mode", False))
    if len(missing):
        missing_df = pd.DataFrame({"Date": pd.to_datetime(missing)})
        st.write("Missing months:")
        st.dataframe(missing_df.style.format({"Date": "{:%Y-%m}"}), use_container_width=True)
        st.download_button("Download missing months", missing_df.to_csv(index=False), "missing_months.csv", "text/csv")
    outlier_points = z.loc[z.abs()>3].dropna()
    if not outlier_points.empty:
        out_df = outlier_points.to_frame(name="z_score")
        st.write("Potential outliers (|z|>3):")
        st.dataframe(out_df.style.format({"z_score": "{:.2f}"}), use_container_width=True)
        st.download_button("Download outliers", out_df.to_csv(), "outliers.csv", "text/csv")

    if has_rjpp:
        fc_end = fc.index.max() if len(fc.index) else None
        rjpp_end = rjpp_full.index.max() if rjpp_full is not None and len(rjpp_full.index) else None
        if fc_end is not None and rjpp_end is not None and rjpp_end < fc_end:
            st.warning(f"RJPP data ends at {rjpp_end:%Y-%m}; forecast extends to {fc_end:%Y-%m} (RJPP forward-filled).")
        core_rjpp = getattr(forecaster, "rjpp", None)
        if core_rjpp is not None and not core_rjpp.empty:
            overlap_idx = forecaster.load_df.index.intersection(core_rjpp.index)
            if len(overlap_idx):
                load_mean = forecaster.load_df.loc[overlap_idx, "Average_Load"].mean()
                rjpp_mean = core_rjpp.loc[overlap_idx, "RJPP"].mean()
                if rjpp_mean and np.isfinite(load_mean) and np.isfinite(rjpp_mean):
                    ratio = load_mean / rjpp_mean
                    if np.isfinite(ratio) and (ratio < 0.1 or ratio > 10):
                        st.warning(f"Historical load vs RJPP mean ratio ~{ratio:.2f}; check for unit mismatch (MW vs kW).")

# ---------- Exports ----------
with tab_exports:
    st.subheader("Downloads")
    base_col = "p50" if "p50" in fc.columns else "Ensemble"
    minimal_out = pd.DataFrame(columns=["Date", "Forecast_MW"])
    annual_energy_summary = pd.DataFrame()
    if len(fc):
        minimal_out = pd.DataFrame({"Date": fc.index, "Forecast_MW": fc[base_col].values})
        if has_rjpp and "RJPP" in fc:
            minimal_out["RJPP_MW"] = fc["RJPP"].values
        hours = pd.Series(fc.index.days_in_month, index=fc.index, dtype=float) * 24
        energy_mwh = fc[base_col] * hours
        annual_energy_summary = pd.DataFrame({
            "Energy_GWh": energy_mwh.groupby(fc.index.year).sum() / 1000,
            "Average_MW": fc[base_col].groupby(fc.index.year).mean(),
            "Peak_MW": fc[base_col].groupby(fc.index.year).max(),
        })
        if not annual_energy_summary.empty:
            annual_energy_summary["Load_Factor"] = annual_energy_summary["Average_MW"] / annual_energy_summary["Peak_MW"].replace(0, np.nan)
            annual_energy_summary.index.name = "Year"
    today = datetime.now().strftime("%Y-%m-%d")
    st.download_button("Forecast CSV", fc.to_csv(), f"forecast_base_{today}.csv", "text/csv")
    st.download_button("Combined CSV", combined.to_csv(), f"combined_{today}.csv", "text/csv")
    if has_rjpp and rjpp_full is not None:
        st.download_button("RJPP CSV", rjpp_full.to_csv(), "rjpp_full.csv", "text/csv")
    if not minimal_out.empty:
        st.download_button("Minimal Forecast CSV", minimal_out.to_csv(index=False), "forecast_minimal.csv", "text/csv")
    if not annual_energy_summary.empty:
        st.download_button("Annual Energy Summary CSV", annual_energy_summary.reset_index().to_csv(index=False), "annual_energy_summary.csv", "text/csv")
    
    # Scenario-related exports
    scenarios_json = json.dumps(st.session_state.scenarios, indent=2, ensure_ascii=False) if st.session_state.get("scenarios") else "[]"
    st.download_button("Scenarios JSON", scenarios_json, f"scenarios_{today}.json", "application/json")
    
    if fc_adj is not None:
        adj_cols = [col for col in fc_adj.columns if col.endswith('_adj')]
        if adj_cols:
            st.download_button("Forecast Adjusted CSV", fc_adj[adj_cols].to_csv(), f"forecast_adjusted_{today}.csv", "text/csv")
    
    # Delta comparison export
    if fc_adj is not None and "Ensemble" in fc.columns:
        base_col = "p50" if "p50" in fc.columns else "Ensemble"
        adj_col = "p50_adj" if "p50_adj" in fc_adj.columns else "Ensemble_adj"
        if adj_col in fc_adj.columns:
            delta_df = pd.DataFrame({
                "Date": fc.index,
                "Base_MW": fc[base_col].values,
                "Adjusted_MW": fc_adj[adj_col].values,
                "Delta_MW": (fc_adj[adj_col] - fc[base_col]).values,
                "Delta_Pct": ((fc_adj[adj_col] - fc[base_col]) / fc[base_col] * 100).replace([np.inf, -np.inf], np.nan).values
            })
            
            # Add annual summary
            delta_annual = pd.DataFrame({
                "Year": fc.index.year,
                "Base_MW": fc[base_col].values,
                "Adjusted_MW": fc_adj[adj_col].values,
                "Delta_MW": (fc_adj[adj_col] - fc[base_col]).values
            }).groupby("Year").agg({
                "Base_MW": "mean",
                "Adjusted_MW": "mean", 
                "Delta_MW": "mean"
            })
            delta_annual["Delta_Pct"] = (delta_annual["Delta_MW"] / delta_annual["Base_MW"] * 100).replace([np.inf, -np.inf], np.nan)
            
            st.download_button("Delta Base vs Adjusted (Monthly) CSV", delta_df.to_csv(index=False), f"scenario_A_vs_B_{today}_monthly.csv", "text/csv")
            st.download_button("Delta Base vs Adjusted (Annual) CSV", delta_annual.reset_index().to_csv(index=False), f"scenario_A_vs_B_{today}_annual.csv", "text/csv")

    clamp_mask = None
    clamped_months = 0
    clamp_pct = np.nan
    if has_rjpp and "RJPP" in fc:
        lo = fc["RJPP"] * (1 - forecaster.max_dev)
        hi = fc["RJPP"] * (1 + forecaster.max_dev)
        tol = np.maximum(fc["RJPP"].abs() * 0.002, 0.5)
        clamp_mask = ((fc["Ensemble"] - lo).abs() <= tol) | ((fc["Ensemble"] - hi).abs() <= tol)
        clamped_months = int(clamp_mask.sum())
        clamp_pct = float(clamp_mask.mean() * 100) if len(clamp_mask) else np.nan

    qa_bytes = io.BytesIO()
    with zipfile.ZipFile(qa_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("forecast.csv", fc.to_csv())
        zf.writestr("forecast_raw_before_align_noise_clip.csv", fc_raw.to_csv())
        zf.writestr("combined.csv", combined.to_csv())
        if not minimal_out.empty:
            zf.writestr("forecast_minimal.csv", minimal_out.to_csv(index=False))
        if not annual_energy_summary.empty:
            zf.writestr("annual_energy_summary.csv", annual_energy_summary.reset_index().to_csv(index=False))
        tmp = fc.copy(); tmp["Year"] = tmp.index.year
        cols = [c for c in ["Ensemble", "p10", "p50", "p90"] if c in tmp.columns]
        annual = tmp.groupby("Year")[cols].mean()
        annual["YoY_%"] = annual[( "p50" if "p50" in annual.columns else "Ensemble")].pct_change()*100
        zf.writestr("yearly_means_p10_p50_p90.csv", annual.to_csv())
        if val_results:
            zf.writestr("validation_metrics_averaged.csv", pd.DataFrame(val_results).T.to_csv())
        if ens_val_metrics and ens_val_pred is not None and val_df_for_bundle is not None and results.get("val_index") is not None:
            val_idx = results["val_index"]
            val_df_out = pd.DataFrame(index=val_idx)
            val_df_out["Actual"] = val_df_for_bundle["Average_Load"].reindex(val_idx)
            for m, arr in forecaster.val_preds.items():
                s = pd.Series(arr); val_df_out[m] = s.reindex(val_idx)
            val_df_out["Ensemble_val"] = ens_val_pred.reindex(val_idx)
            zf.writestr("validation_window_preds.csv", val_df_out.to_csv())
        if validation_windows:
            frames = []
            for name, dfm in validation_windows.items():
                df_local = dfm.copy()
                df_local["Model"] = name
                frames.append(df_local)
            if frames:
                vw_df = pd.concat(frames)
                zf.writestr("validation_windows_metrics.csv", vw_df.to_csv(index=False))
        models_used = {
            slot: {
                "implementation": forecaster.slot_info.get(slot, {}).get("impl", ""),
                "status": forecaster.slot_info.get(slot, {}).get("status", forecaster.model_status.get(slot, "unknown")),
                "weight": float(weights.get(slot, 0.0))
            }
            for slot in MODEL_SLOTS
        }
        zf.writestr("models_used.json", json.dumps(models_used, indent=2))
        if has_rjpp and "RJPP" in fc:
            dev = (fc["Ensemble"] - fc["RJPP"]) / fc["RJPP"] * 100
            zf.writestr("rjpp_monthly_deviation.csv", pd.DataFrame({"Date": dev.index, "dev%": dev.values, "abs_dev%": dev.abs().values}).to_csv(index=False))
        calibration_display = f"{calibration_pct:.1f}%" if calibration_pct is not None and np.isfinite(calibration_pct) else "n/a"
        clamp_line = "n/a" if not has_rjpp else (f"{clamped_months}/{len(fc)} ({clamp_pct:.1f}%)" if len(fc) else "0")
        settings = f"""Run Settings:
- Forecast: {_month_start(forecast_start):%b %Y} -> {_month_start(forecast_end):%b %Y}
- Ensemble method: {ensemble_choice} (auto-selected from PRESETS)
- Annual anchor tolerance: ±{PRESETS['annual_anchor_tolerance']*100:.1f}%
- Horizon-aware caps: h1-3={PRESETS['horizon_max_dev']['h1_3']*100:.0f}%, h4-6={PRESETS['horizon_max_dev']['h4_6']*100:.0f}%, h7-12={PRESETS['horizon_max_dev']['h7_12']*100:.0f}%, h13+={PRESETS['horizon_max_dev']['h13p']*100:.0f}%
- MoM caps (horizon-aware): h1-12={PRESETS['mom_cap']['h1_12']*100:.0f}%, h13+={PRESETS['mom_cap']['h13p']*100:.0f}%
- Residual noise: {"On" if forecaster.add_noise else "Off"} (scale={forecaster.noise_scale:.2f})
- Use RJPP: {"Yes" if forecaster.use_rjpp else "No"}
- Residualize (effective): {"Yes" if getattr(forecaster, '_residualize_effective', False) else "No"}
- Seasonal strength: {getattr(forecaster, 'seasonal_strength', 0.0):.3f}
- Guard scope: {forecaster.apply_guards_to}
- Smoothing alpha: {PRESETS['smoothing_alpha']:.2f} (disabled)
- Calibration (p10-90 target): {PRESETS['band_target_coverage']*100:.0f}%
- Calibration (actual): {calibration_display}
- Band calibration k: {forecaster.band_k:.3f}
- Validation RMSE (weighted): {getattr(forecaster, 'val_rmse_weighted', np.nan):.2f}
- Validation RMSE (meta): {getattr(forecaster, 'val_rmse_meta', np.nan):.2f}
- Models available: {", ".join([m for m,s in model_status.items() if s=="ok"])}
- Models unavailable: {", ".join([m for m,s in model_status.items() if s!="ok"])}
- Python: {sys.version.split()[0]}
- numpy: {np.__version__}; pandas: {pd.__version__}; plotly: {plotly.__version__}; statsmodels: {sm.__version__}; sklearn: {sklearn_version}
"""
        zf.writestr("settings.txt", settings)
    st.download_button("QA bundle (.zip)", qa_bytes.getvalue(), f"qa_bundle_{today}.zip", "application/zip")

    if has_rjpp and "RJPP" in fc and "Ensemble" in fc:
        st.subheader("Annual Re-anchor Check (target <=1.0%)")
        rows=[]
        for y in sorted(fc.index.year.unique()):
            mask = fc.index.year==y
            yr_mean = fc.loc[mask,"Ensemble"].mean()
            dec_val = fc.loc[fc.index==pd.Timestamp(y,12,1),"RJPP"]
            if len(dec_val):
                rjpp_dec = float(dec_val.values[0])
                gap = abs((yr_mean - rjpp_dec)/rjpp_dec)*100 if rjpp_dec else np.nan
                status = "Pass" if gap <= 1.0 else ("Watch" if gap <= 1.5 else "Fail")
                rows.append({"Year":y,"Gap_%":gap,"Status":status})
        if rows:
            df_anchor = pd.DataFrame(rows)
            st.dataframe(df_anchor.style.format({"Gap_%":"{:.2f}"}), use_container_width=True)

    if ui_log_handler.buffer:
        st.subheader("Pipeline Logs")
        st.code("\n".join(ui_log_handler.buffer[-200:]), language="text")

# ---------- Scenarios tab (consolidated builder, manager, compare) ----------
#############################
# Redesigned Scenario Tab   #
#############################

# --- Helper: Month utilities for new UI ---
def _ym_to_first_of_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-01"

def _parse_month(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).to_period("M").to_timestamp()

def validate_scenario_dict(d: dict) -> list[str]:
    errors = []
    if not d.get("name"):
        errors.append("Name is required.")
    # Accept either capacity_mw or mw
    capacity = d.get("capacity_mw") or d.get("mw", 0)
    if capacity <= 0:
        errors.append("Capacity must be > 0.")
    profile = d.get("profile")
    start = d.get("start_month")
    peak = d.get("peak_month")
    plateau = d.get("plateau_end")
    retire = d.get("retire_month")
    def _to_ts(x):
        return pd.to_datetime(x).to_period("M").to_timestamp() if x else None
    ts_start, ts_peak, ts_plateau, ts_retire = map(_to_ts, [start, peak, plateau, retire])
    if profile in ("ramp", "s_curve") and not peak:
        errors.append("Peak month required for Ramp / S-curve.")
    ordering = [(ts_start, "Start"), (ts_peak, "Peak"), (ts_plateau, "PlateauEnd"), (ts_retire, "Retire")]
    prev = None
    for ts, label in ordering:
        if ts is None:
            continue
        if prev and ts < prev:
            errors.append(f"Month ordering invalid: {label} before previous milestone.")
        prev = ts
    if profile == "custom_csv":
        if not d.get("_uploaded_valid", False):
            errors.append("Custom CSV not uploaded or invalid.")
    return errors

@st.cache_data(show_spinner=False)
def _hash_scenario_for_cache(d: dict) -> str:
    keys = [k for k in d.keys() if not k.startswith("_") and k not in ("active", "priority")]
    blob = json.dumps({k: d.get(k) for k in sorted(keys)}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()

def _index_signature(index: pd.DatetimeIndex | None) -> str:
    """Return a compact, hashable signature describing a DatetimeIndex."""
    try:
        if index is None or len(index) == 0:
            return "0:None:None:None"
        first = pd.to_datetime(index[0]).strftime('%Y-%m-%d')
        last = pd.to_datetime(index[-1]).strftime('%Y-%m-%d')
        # freqstr may be None for irregular indexes
        freq = getattr(index, 'freqstr', None) or (str(getattr(index, 'freq', None)) if getattr(index, 'freq', None) else 'None')
        return f"{len(index)}:{first}:{last}:{freq}"
    except Exception:
        # As a last resort, use md5 of the stringified values (avoid huge keys)
        try:
            vals = ",".join(pd.to_datetime(index).strftime('%Y-%m-%d').tolist()) if index is not None else ""
            return hashlib.md5(vals.encode()).hexdigest()
        except Exception:
            return "unknown_index"

@st.cache_data(show_spinner=False)
def build_scenario_series(d: dict, _index: pd.DatetimeIndex, index_sig: str) -> pd.Series:
    """Return a monthly MW series for the scenario aligned to index."""
    profile = d.get("profile")
    cap = float(d.get("capacity_mw", 0.0) or 0.0)
    start = d.get("start_month")
    peak = d.get("peak_month")
    plateau = d.get("plateau_end")
    retire = d.get("retire_month")
    if profile == "step":
        # Use existing step implementation logic (reuse scenario_engine if possible)
        from scenario_engine import step_profile
        return step_profile(cap, _index, start, retire)
    elif profile == "ramp":
        from scenario_engine import ramp_profile
        return ramp_profile(cap, _index, start, peak, plateau, retire)
    elif profile == "s_curve":
        from scenario_engine import s_curve_profile
        if not peak:
            return pd.Series(0.0, index=_index)
        return s_curve_profile(cap, _index, start, peak)
    elif profile == "custom_csv":
        path = d.get("_custom_csv_path")
        if not path:
            return pd.Series(0.0, index=_index)
        try:
            df = pd.read_csv(path)
            if not {"month", "mw"}.issubset({c.lower() for c in df.columns}):
                return pd.Series(0.0, index=_index)
            # Normalize column names
            col_map = {c: c.lower() for c in df.columns}
            df.columns = [c.lower() for c in df.columns]
            df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
            monthly = df.set_index("month")["mw"].astype(float).reindex(_index).fillna(0.0)
            # Enforce non-negative
            return monthly.clip(lower=0.0)
        except Exception:
            return pd.Series(0.0, index=_index)
    else:
        return pd.Series(0.0, index=_index)

def compute_planned_components(scenarios: list[dict], index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    active = [s for s in scenarios if s.get("active", True)]
    # Normalize priority: existing integer or fallback to list order
    for i, s in enumerate(active):
        if "priority" not in s or s["priority"] is None or not isinstance(s["priority"], int):
            s["priority"] = i
    ordered = sorted(active, key=lambda x: x.get("priority", 0))
    comp = {}
    idx_sig = _index_signature(index)
    for scn in ordered:
        series = build_scenario_series(scn, index, idx_sig)
        comp[scn["name"]] = series
    return comp

def sum_components(components: dict[str, pd.Series]) -> pd.Series:
    if not components:
        return pd.Series(dtype=float)
    total = None
    for s in components.values():
        if total is None:
            total = s.astype(float)
        else:
            total = total.add(s.astype(float), fill_value=0.0)
    return total.clip(lower=0.0) if total is not None else pd.Series(dtype=float)

def plot_stacked_overlay(components: dict[str, pd.Series], base_df: pd.DataFrame, adjusted_df: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure(layout=PLOTLY_BASE_LAYOUT)
    # Stacked components
    for name, series in components.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, mode="lines", stackgroup="scn", line=dict(width=1)))
    base_col = "p50" if "p50" in base_df.columns else "Ensemble"
    fig.add_trace(go.Scatter(x=base_df.index, y=base_df[base_col], name="Base p50", mode="lines", line=dict(width=2, dash="dot", color="#555")))
    if adjusted_df is not None:
        adj_col = "p50_adj" if "p50_adj" in adjusted_df.columns else ("Ensemble_adj" if "Ensemble_adj" in adjusted_df.columns else None)
        if adj_col:
            fig.add_trace(go.Scatter(x=adjusted_df.index, y=adjusted_df[adj_col], name="Adjusted p50", mode="lines", line=dict(width=3, color=PALETTE.get("scenario", "#D97706"))))
    fig.update_yaxes(title_text="MW", ticksuffix=" MW", rangemode="tozero")
    fig.update_layout(title="Stacked Scenario Overlay")
    return fig

def plot_annual_impact(adjusted_df: pd.DataFrame, base_df: pd.DataFrame) -> go.Figure:
    base_col = "p50" if "p50" in base_df.columns else "Ensemble"
    adj_col = "p50_adj" if "p50_adj" in adjusted_df.columns else ("Ensemble_adj" if "Ensemble_adj" in adjusted_df.columns else None)
    fig = go.Figure(layout=PLOTLY_BASE_LAYOUT)
    if adj_col is None:
        return fig
    b = base_df[base_col].groupby(base_df.index.year).mean()
    a = adjusted_df[adj_col].groupby(adjusted_df.index.year).mean()
    years = sorted(set(b.index).union(a.index))
    delta_mw = []
    for y in years:
        bv = b.get(y, np.nan)
        av = a.get(y, np.nan)
        delta_mw.append(av - bv if (np.isfinite(av) and np.isfinite(bv)) else np.nan)
    fig.add_trace(go.Bar(x=[str(y) for y in years], y=delta_mw, name="ΔMW", marker_color=["green" if (v or 0) >=0 else "red" for v in delta_mw]))
    fig.update_yaxes(title_text="Δ Avg MW")
    fig.update_layout(title="Annual Impact (Adjusted - Base)")
    return fig

def compute_kpis(base_df: pd.DataFrame, adjusted_df: pd.DataFrame | None, rjpp_df: pd.DataFrame | None) -> dict:
    base_col = "p50" if "p50" in base_df.columns else "Ensemble"
    adj_col = None
    if adjusted_df is not None:
        adj_col = "p50_adj" if "p50_adj" in adjusted_df.columns else ("Ensemble_adj" if "Ensemble_adj" in adjusted_df.columns else None)
    out = {}
    base_peak = base_df[base_col].max() if len(base_df) else np.nan
    base_peak_month = base_df[base_col].idxmax() if len(base_df) else None
    out["base_peak_mw"] = float(base_peak) if np.isfinite(base_peak) else np.nan
    out["base_peak_month"] = base_peak_month
    if adj_col:
        adj_peak = adjusted_df[adj_col].max()
        adj_peak_month = adjusted_df[adj_col].idxmax()
    else:
        adj_peak = np.nan; adj_peak_month = None
    out["delta_peak_mw"] = (adj_peak - base_peak) if np.isfinite(adj_peak) and np.isfinite(base_peak) else np.nan
    out["delta_peak_month"] = adj_peak_month
    # Average next 12 months
    def _avg12(df, col):
        if df is None or col is None or col not in df.columns or df.empty:
            return np.nan
        return float(df[col].iloc[:12].mean()) if len(df) >= 1 else np.nan
    base_avg12 = _avg12(base_df, base_col)
    adj_avg12 = _avg12(adjusted_df, adj_col) if adj_col else np.nan
    out["delta_avg_mw"] = (adj_avg12 - base_avg12) if np.isfinite(base_avg12) and np.isfinite(adj_avg12) else np.nan
    out["delta_avg_pct"] = ((adj_avg12 / base_avg12) - 1) * 100 if np.isfinite(base_avg12) and base_avg12>0 and np.isfinite(adj_avg12) else np.nan
    # RJPP compliance simple calc
    def _compliance(df, col):
        if df is None or col is None or "RJPP" not in df.columns:
            return np.nan
        series = (df[col] - df["RJPP"]).abs() / df["RJPP"].replace(0, np.nan)
        return float((series <= getattr(st.session_state.forecaster, 'max_dev', 0.05)).mean()*100)
    if rjpp_df is not None:
        base_tmp = base_df.copy(); base_tmp["RJPP"] = rjpp_df["RJPP"].reindex(base_df.index)
        out["rjpp_base_pct"] = _compliance(base_tmp, base_col)
        if adj_col and adjusted_df is not None:
            adj_tmp = adjusted_df.copy(); adj_tmp["RJPP"] = rjpp_df["RJPP"].reindex(adjusted_df.index)
            out["rjpp_adj_pct"] = _compliance(adj_tmp, adj_col)
            if np.isfinite(out.get("rjpp_base_pct", np.nan)) and np.isfinite(out.get("rjpp_adj_pct", np.nan)):
                out["rjpp_delta_pp"] = out["rjpp_adj_pct"] - out["rjpp_base_pct"]
            else:
                out["rjpp_delta_pp"] = np.nan
        else:
            out["rjpp_adj_pct"] = np.nan; out["rjpp_delta_pp"] = np.nan
    else:
        out.update({"rjpp_base_pct": np.nan, "rjpp_adj_pct": np.nan, "rjpp_delta_pp": np.nan})
    # Placeholder clamped metrics (reuse if base_df already has them calculated externally)
    out["clamped_months_base"] = np.nan
    out["clamped_months_adj"] = np.nan
    out["worst_anchor_gap_category"] = None
    return out

def ui_quick_planner_form():
    st.markdown("#### Quick Planner")
    with st.container(border=True):
        colA, colB = st.columns([1,1])
        with colA:
            name = st.text_input("Name", key="qp_name", help="Unique scenario name.")
        with colB:
            capacity = st.number_input("Capacity (MW)", min_value=0.0, value=0.0, step=10.0, key="qp_cap", help="Peak incremental load contribution in MW.")
        # Start month pickers
        now_year = datetime.now().year
        years = list(range(now_year, now_year+20))
        colY, colM = st.columns([1,1])
        with colY:
            start_year = st.selectbox("Start year", years, index=0, key="qp_start_year")
        with colM:
            start_month = st.selectbox("Start month", list(range(1,13)), index=0, key="qp_start_month")
        start_month_str = _ym_to_first_of_month(start_year, start_month)
        profile = st.selectbox("Profile type", ["Step","Ramp","S-curve","Custom CSV"], key="qp_profile", help="Shape of build-up over time.")
        # Conditional fields
        peak_month = plateau_end = retire_month = None
        if profile in ("Ramp", "S-curve"):
            col1, col2 = st.columns([1,1])
            with col1:
                peak_year = st.selectbox("Peak year", years, index=min(1, len(years)-1), key="qp_peak_year", help="Month when full capacity is first reached.")
            with col2:
                peak_m = st.selectbox("Peak month", list(range(1,13)), index=5, key="qp_peak_month")
            peak_month = _ym_to_first_of_month(peak_year, peak_m)
            colp1, colp2 = st.columns([1,1])
            with colp1:
                plateau_flag = st.checkbox("Plateau end?", value=False, key="qp_plateau_flag")
            if plateau_flag:
                with colp2:
                    plateau_year = st.selectbox("Plateau end year", years, index=min(2,len(years)-1), key="qp_plateau_year")
                plateau_m = st.selectbox("Plateau end month", list(range(1,13)), index=0, key="qp_plateau_month")
                plateau_end = _ym_to_first_of_month(st.session_state.qp_plateau_year, plateau_m)
            retire_flag = st.checkbox("Retire?", value=False, key="qp_retire_flag")
            if retire_flag:
                retire_year = st.selectbox("Retire year", years, index=min(3,len(years)-1), key="qp_retire_year")
                retire_m = st.selectbox("Retire month", list(range(1,13)), index=0, key="qp_retire_month")
                retire_month = _ym_to_first_of_month(retire_year, retire_m)
        uploaded_path = None
        upload_valid = False
        if profile == "Custom CSV":
            csv_file = st.file_uploader("Upload custom CSV", type=["csv"], key="qp_upload", help="Columns: month (YYYY-MM-01), mw (float).")
            if csv_file is not None:
                try:
                    tmp_path = _save_file(csv_file, "custom_scn_")
                    df = pd.read_csv(tmp_path)
                    if {"month", "mw"}.issubset({c.lower() for c in df.columns}):
                        uploaded_path = tmp_path
                        upload_valid = True
                    else:
                        st.warning("CSV must contain month,mw columns.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
        profile_key = profile.lower().replace(" ", "_")
        scenario_preview = None
        if capacity>0:
            temp_dict = {
                "name": name or "(preview)",
                "capacity_mw": capacity,
                "start_month": start_month_str,
                "profile": profile_key if profile_key != "custom_csv" else "custom_csv",
                "peak_month": peak_month,
                "plateau_end": plateau_end,
                "retire_month": retire_month,
                "_custom_csv_path": uploaded_path,
                "_uploaded_valid": upload_valid,
                "active": True,
                "priority": 0,
            }
            errs = validate_scenario_dict(temp_dict)
            if not errs:
                idx = st.session_state.results["forecast"].index if st.session_state.get("results") else pd.date_range(start_month_str, periods=24, freq="MS")
                scenario_preview = build_scenario_series(temp_dict, idx, _index_signature(idx))
            else:
                for e in errs:
                    st.error(e)
        st.markdown("Preview")
        if scenario_preview is not None:
            fig_prev = go.Figure(layout=PLOTLY_BASE_LAYOUT)
            fig_prev.add_trace(go.Scatter(x=scenario_preview.index, y=scenario_preview.values, mode="lines", name="Profile"))
            fig_prev.update_layout(height=180, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig_prev, use_container_width=True)
        else:
            st.caption("Fill required fields to preview profile…")
        save_btn, reset_btn = st.columns([1,1])
        with save_btn:
            do_save = st.button("Save scenario", type="primary", key="qp_save")
        with reset_btn:
            do_reset = st.button("Reset form", key="qp_reset")
    # Handle actions
    saved = None
    if do_reset:
        for k in [k for k in st.session_state.keys() if k.startswith("qp_")]:
            st.session_state.pop(k, None)
        st.experimental_rerun()
    if do_save:
        record = {
            "name": name.strip() if name else f"Scenario {len(st.session_state.scenarios)+1}",
            "capacity_mw": capacity,
            "start_month": start_month_str,
            "profile": profile_key if profile_key != "custom_csv" else "custom_csv",
            "peak_month": peak_month,
            "plateau_end": plateau_end,
            "retire_month": retire_month,
            "probability": None,
            "priority": len(st.session_state.scenarios),
            "active": True,
            "_custom_csv_path": uploaded_path,
            "_uploaded_valid": upload_valid,
        }
        errs = validate_scenario_dict(record)
        existing_names = {s.get("name", "").lower() for s in st.session_state.scenarios}
        if record["name"].lower() in existing_names:
            # auto suffix
            base = record["name"]
            i = 2
            while f"{base} ({i})".lower() in existing_names:
                i += 1
            record["name"] = f"{base} ({i})"
        if errs:
            for e in errs: st.error(e)
        else:
            st.session_state.scenarios.append(record)
            st.success(f"Saved scenario '{record['name']}'.")
            saved = record
    return saved

def ui_scenarios_tab():
    st.subheader("Scenario Planning")
    top_cols = st.columns([1,1,1])
    with top_cols[0]:
        st.session_state.apply_scenarios = st.toggle("Apply scenarios to forecast", value=st.session_state.get("apply_scenarios", False), key="apply_scenarios_master")
    with top_cols[1]:
        explain_flag = st.toggle("Explain mode", value=st.session_state.get("scn_explain", False), key="scn_explain")
    with top_cols[2]:
        st.caption(f"Active scenarios: {sum(1 for s in st.session_state.scenarios if s.get('active'))}")
    # Two-column layout
    col_left, col_right = st.columns([1,1])
    with col_left:
        saved = ui_quick_planner_form()
    with col_right:
        st.markdown("#### Library")
        if not st.session_state.scenarios:
            st.info("No scenarios yet. Create one using Quick Planner.")
        else:
            # Render cards
            for i, scn in enumerate(st.session_state.scenarios):
                with st.container(border=True):
                    header_cols = st.columns([0.6,0.4])
                    with header_cols[0]:
                        st.markdown(f"**{scn.get('name')}**")
                    with header_cols[1]:
                        scn["active"] = st.checkbox("Active", value=scn.get("active", True), key=f"scn_active_{i}")
                    chips = []
                    chips.append(f"{int(scn.get('capacity_mw', scn.get('mw',0)))} MW")
                    chips.append(scn.get("profile","?").replace("_"," "))
                    if scn.get("probability") is not None:
                        chips.append(f"P={scn['probability']:.2f}")
                    st.caption(" | ".join(chips))
                    dates_line = []
                    for lbl, key in [("Start","start_month"),("Peak","peak_month"),("Plateau","plateau_end"),("Retire","retire_month")]:
                        if scn.get(key):
                            dates_line.append(f"{lbl}:{scn[key][:7]}")
                    if dates_line:
                        st.write("; ".join(dates_line))
                    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([0.25,0.25,0.25,0.25])
                    with btn_c1:
                        if st.button("↑", key=f"scn_up_{i}", help="Higher priority") and i>0:
                            st.session_state.scenarios[i-1], st.session_state.scenarios[i] = st.session_state.scenarios[i], st.session_state.scenarios[i-1]
                    with btn_c2:
                        if st.button("↓", key=f"scn_down_{i}", help="Lower priority") and i < len(st.session_state.scenarios)-1:
                            st.session_state.scenarios[i+1], st.session_state.scenarios[i] = st.session_state.scenarios[i], st.session_state.scenarios[i+1]
                    with btn_c3:
                        if st.button("Duplicate", key=f"scn_dup_{i}"):
                            dup = scn.copy(); dup["name"] = f"{dup['name']} (copy)"; st.session_state.scenarios.append(dup)
                    with btn_c4:
                        if st.button("Delete", key=f"scn_del_{i}"):
                            st.session_state.scenarios.pop(i); st.experimental_rerun()
    # Impact & Compare full width
    st.markdown("#### Impact & Compare")
    forecast_idx = st.session_state.results["forecast"].index
    components = {}
    overlay = None
    adjusted = None
    if st.session_state.apply_scenarios and any(s.get("active", True) for s in st.session_state.scenarios):
        components = compute_planned_components(st.session_state.scenarios, forecast_idx)
        overlay = sum_components(components)
        if overlay is not None and len(overlay):
            # For scenario analysis, adjustments must reflect the physical addition
            # of load without policy clamps. We therefore do NOT bound the
            # adjusted series to RJPP here (clamp_to_rjpp=False). This ensures a
            # 100 MW scenario peak truly adds ~100 MW to the corresponding
            # months in the adjusted p50/Ensemble, matching the stacked overlay.
            rjpp_series = (
                st.session_state.results["forecast"].get("RJPP")
                if "RJPP" in st.session_state.results["forecast"].columns
                else None
            )
            adjusted = apply_scenarios_to_bands(
                st.session_state.results["forecast"],
                overlay,
                rjpp_series,
                st.session_state.forecaster.max_dev,
                clamp_to_rjpp=False,
            )
    if not st.session_state.apply_scenarios:
        st.info("Scenarios not applied. Toggle 'Apply scenarios to forecast' above.")
    elif not any(s.get("active", True) for s in st.session_state.scenarios):
        st.warning("No active scenarios. Activate at least one in the library.")
    # Charts
    if components:
        fig_overlay = plot_stacked_overlay(components, st.session_state.results["forecast"], adjusted)
        st.plotly_chart(fig_overlay, use_container_width=True)
    st.markdown("#### Scenario comparison")
    base_forecast = st.session_state.results["forecast"]
    combined_series = st.session_state.combined
    rjpp_full_local = st.session_state.rjpp_full if st.session_state.rjpp_full is not None else None
    if isinstance(combined_series, pd.Series):
        combined_frame = combined_series.to_frame(name="Combined")
    elif isinstance(combined_series, pd.DataFrame):
        combined_frame = combined_series
    else:
        combined_frame = pd.DataFrame()
    col_base, col_adjusted = st.columns(2)
    with col_base:
        st.markdown("**Base forecast**")
        plot_hero(
            combined_frame,
            base_forecast,
            rjpp_full_local,
            key_suffix="scenario_base",
            show_bands=show_bands,
            bounded=bounded_bands,
            max_dev=forecaster.max_dev,
            forecast_start=forecast_start_ts,
        )
    with col_adjusted:
        st.markdown("**Scenario-adjusted**")
        if adjusted is not None:
            combined_adjusted = st.session_state.combined_adjusted if st.session_state.combined_adjusted is not None else combined_series
            if isinstance(combined_adjusted, pd.Series):
                combined_adjusted_frame = combined_adjusted.to_frame(name="Combined")
            elif isinstance(combined_adjusted, pd.DataFrame):
                combined_adjusted_frame = combined_adjusted
            else:
                combined_adjusted_frame = combined_frame
            plot_hero(
                combined_adjusted_frame,
                base_forecast,
                rjpp_full_local,
                key_suffix="scenario_adjusted",
                show_bands=show_bands,
                bounded=bounded_bands,
                max_dev=forecaster.max_dev,
                fc_adj=adjusted,
                combined_adj=combined_adjusted,
                planned_overlay=overlay,
                forecast_start=forecast_start_ts,
            )
        else:
            st.info("Activate scenarios to see adjusted forecast impacts.")
    # KPI tiles
    kpi_cols = st.columns(4)
    kpis = compute_kpis(st.session_state.results["forecast"], adjusted, st.session_state.rjpp_full if st.session_state.rjpp_full is not None else None)
    show_metric(kpi_cols[0], "ΔPeak (MW)", f"{kpis.get('delta_peak_mw', float('nan')):,.0f}" if np.isfinite(kpis.get('delta_peak_mw', np.nan)) else "n/a")
    show_metric(kpi_cols[1], "ΔAvg 12m (MW)", f"{kpis.get('delta_avg_mw', float('nan')):,.1f}" if np.isfinite(kpis.get('delta_avg_mw', np.nan)) else "n/a")
    show_metric(kpi_cols[2], "ΔAvg 12m (%)", f"{kpis.get('delta_avg_pct', float('nan')):,.2f}%" if np.isfinite(kpis.get('delta_avg_pct', np.nan)) else "n/a")
    if np.isfinite(kpis.get("rjpp_delta_pp", np.nan)):
        show_metric(kpi_cols[3], "RJPP Δ (pp)", f"{kpis['rjpp_delta_pp']:.2f}pp")
    # Annual impact bars
    if adjusted is not None:
        st.plotly_chart(plot_annual_impact(adjusted, st.session_state.results["forecast"]), use_container_width=True)
    # Export active scenarios overlay and components
    if overlay is not None and len(overlay):
        stub = pd.DataFrame({"Date": overlay.index, "Planned_MW": overlay.values})
        st.download_button(
            "Scenario overlay CSV",
            stub.to_csv(index=False),
            "scenario_overlay.csv",
            "text/csv",
        )
        # Components CSV (each active scenario as a column + overlay)
        try:
            comp_df = pd.DataFrame(index=overlay.index)
            for name, s in (components or {}).items():
                comp_df[name] = s.reindex(overlay.index).astype(float)
            comp_df["Overlay"] = overlay.astype(float)
            comp_df_reset = comp_df.reset_index().rename(columns={comp_df.index.name or 'index': 'Date'})
            st.download_button(
                "Scenario components CSV",
                comp_df_reset.to_csv(index=False),
                "scenario_components.csv",
                "text/csv",
            )
        except Exception:
            pass
    # Import/Export scenarios (JSON)
    with st.expander("Import / Export Scenarios", expanded=False):
        c1, c2 = st.columns([1,1])
        with c1:
            try:
                def _clean_for_export(s: dict) -> dict:
                    # drop transient/internal keys
                    return {k: v for k, v in s.items() if not str(k).startswith('_')}
                export_payload = json.dumps([_clean_for_export(s) for s in st.session_state.scenarios], indent=2)
                st.download_button(
                    "Export scenarios JSON",
                    export_payload,
                    "scenarios.json",
                    "application/json",
                )
            except Exception:
                st.caption("No scenarios to export.")
        with c2:
            up = st.file_uploader("Import scenarios JSON", type=["json"], key="scn_import")
            if up is not None:
                try:
                    data = json.load(up)
                    if isinstance(data, dict):
                        data = data.get("scenarios", [])
                    added = 0
                    for rec in data if isinstance(data, list) else []:
                        if not isinstance(rec, dict):
                            continue
                        errs = validate_scenario_dict(rec)
                        if errs:
                            continue
                        st.session_state.scenarios.append(rec)
                        added += 1
                    if added:
                        st.success(f"Imported {added} scenarios.")
                        st.experimental_rerun()
                    else:
                        st.info("No valid scenarios found in file.")
                except Exception as e:
                    st.error(f"Import failed: {e}")

with tab_scenarios:
    ui_scenarios_tab()

# Prefer module-level AI functions if available (override local stubs)
try:
    if 'ai_mod' in globals() and ai_mod is not None:
        ollama_analyze = ai_mod.ollama_analyze  # type: ignore
        build_analysis_payload = ai_mod.build_analysis_payload  # type: ignore
except Exception:
    pass

# ---------- AI Insights ----------
with tab_ai:
    # --- AI Insight (chat) ---
    st.subheader("AI Insight")
    if st.session_state.pop("ai_focus_banner", False):
        st.info("Quick action triggered. Scroll here to prompt the AI assistant below.")

    # Optional imports from ai_insights; guard to avoid hard failures
    try:
        from ai_insights import (
            build_chat_facts as _build_chat_facts_mod,
            ollama_chat as _ollama_chat_mod,
            build_peak_trend_payload as _build_peak_trend_payload_mod,
            ollama_analyze_narrative as _ollama_analyze_narrative_mod,
        )
    except Exception:
        _build_chat_facts_mod = None
        _ollama_chat_mod = None
        _build_peak_trend_payload_mod = None
        _ollama_analyze_narrative_mod = None

    # Utilities for local fallbacks
    def _list_ollama_models_local() -> list[str]:
        try:
            import json, urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("models", []) or []
                return [m.get("name") for m in items if isinstance(m, dict) and m.get("name")]
        except Exception:
            return ["qwen3:8b", "deepseek-r1:8b", "llama3.2:3b"]

    def _ollama_chat_local(model: str, question: str, facts: dict, temperature: float = 0.1, num_predict: int = 450, num_ctx: int = 2048, timeout_s: int = 180) -> str:
        try:
            import json, urllib.request
            
            # Extract key metrics and convert to natural language facts
            meta = facts.get("meta", {})
            next12 = facts.get("next_12m", {})
            planning = facts.get("planning", {})
            validation = facts.get("validation", {})
            peaks = facts.get("peaks", {})
            uncertainty = facts.get("uncertainty", {})
            annual = facts.get("annual", {})
            weights = facts.get("weights", {})
            history = facts.get("history", {})
            scenarios = facts.get("scenarios", [])
            
            # Build comprehensive natural language context
            facts_narrative = []
            
            # Historical context
            if history.get("last_actual_mw"):
                facts_narrative.append(f"Current actual load is {history['last_actual_mw']:.1f} MW.")
            if history.get("last_12m_avg_mw"):
                facts_narrative.append(f"The last 12 months averaged {history['last_12m_avg_mw']:.1f} MW with total energy of {history.get('last_12m_energy_gwh', 0):.1f} GWh.")
            
            # Forecast period
            if meta.get("forecast_start") and meta.get("forecast_end"):
                facts_narrative.append(f"The forecast covers {meta['months']} months from {meta['forecast_start']} to {meta['forecast_end']}.")
            
            # Growth trend
            if meta.get("cagr_pct_per_year") is not None:
                cagr = meta['cagr_pct_per_year']
                trend_word = "growing" if cagr > 0 else ("declining" if cagr < 0 else "stable")
                facts_narrative.append(f"Overall demand is {trend_word} at {abs(cagr):.2f}% per year (CAGR).")
            
            # Next 12 months key metrics
            if next12.get("avg_mw") and next12.get("peak_mw") and next12.get("peak_month"):
                facts_narrative.append(f"In the next 12 months, average demand will be {next12['avg_mw']:.1f} MW, with a peak of {next12['peak_mw']:.1f} MW expected in {next12['peak_month']}.")
                if next12.get("energy_gwh"):
                    facts_narrative.append(f"Total energy consumption for the next 12 months is projected at {next12['energy_gwh']:.1f} GWh.")
            
            # Top peaks
            if peaks.get("top3") and len(peaks["top3"]) > 0:
                peak_list = ", ".join([f"{pk['month']} ({pk['mw']:.1f} MW)" for pk in peaks["top3"][:3]])
                facts_narrative.append(f"The three highest peak months are: {peak_list}.")
            
            # Model ensemble information
            if weights:
                top_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
                model_desc = ", ".join([f"{m} ({w*100:.1f}%)" for m, w in top_models])
                facts_narrative.append(f"The forecast uses an ensemble of models weighted as: {model_desc}.")
            
            # Uncertainty bands
            if uncertainty.get("avg_band_width_mw") and uncertainty.get("max_band_width_mw"):
                facts_narrative.append(f"Forecast uncertainty: average band width is {uncertainty['avg_band_width_mw']:.1f} MW, with maximum reaching {uncertainty['max_band_width_mw']:.1f} MW.")
            
            # Validation performance
            if validation.get("band_calibration_pct"):
                facts_narrative.append(f"Model validation shows {validation['band_calibration_pct']:.1f}% of historical months fall within the p10-p90 uncertainty bands.")
            
            # RJPP compliance
            if planning.get("rjpp_compliance_pct"):
                facts_narrative.append(f"RJPP regulatory compliance: {planning['rjpp_compliance_pct']:.1f}% of forecast months are within {meta.get('max_dev_pct', 5):.1f}% deviation from regulatory baseline.")
            
            # Planning reserve analysis
            if planning.get("avg_monthly_shortfall_mw_vs_p90_prm"):
                prm = planning.get("assumed_prm_pct", 15)
                shortfall = planning['avg_monthly_shortfall_mw_vs_p90_prm']
                if shortfall > 0:
                    facts_narrative.append(f"Capacity planning: assuming {prm:.0f}% planning reserve margin, average monthly shortfall vs. p90 demand is {shortfall:.1f} MW.")
            
            # Annual breakdown (first 3-5 years)
            if annual:
                years_to_show = sorted(annual.keys())[:5]
                facts_narrative.append(f"Annual projections for the next {len(years_to_show)} years:")
                for year in years_to_show:
                    yr_data = annual[year]
                    facts_narrative.append(f"  - {year}: Average demand {yr_data.get('avg_mw', 0):.1f} MW, peak {yr_data.get('peak_mw', 0):.1f} MW, total energy {yr_data.get('energy_gwh', 0):.1f} GWh.")
            
            # Scenarios summary
            if scenarios and len(scenarios) > 0:
                active_scenarios = [s for s in scenarios if s.get('active', True)]
                if active_scenarios:
                    total_capacity = sum(s.get('capacity_mw', 0) or s.get('mw', 0) for s in active_scenarios)
                    facts_narrative.append(f"There are {len(active_scenarios)} active scenarios adding {total_capacity:.1f} MW of capacity.")
                    top_scenarios = sorted(active_scenarios, key=lambda s: s.get('capacity_mw', 0) or s.get('mw', 0), reverse=True)[:3]
                    for scn in top_scenarios:
                        name = scn.get('name', 'Unnamed')
                        cap = scn.get('capacity_mw', 0) or scn.get('mw', 0)
                        start = scn.get('start_month', 'TBD')
                        facts_narrative.append(f"  - {name}: {cap:.1f} MW starting {start}")

            # Scenario-adjusted forecast deltas (compare adjusted vs. base)
            try:
                base_series = (facts.get("series", {}) or {}).get("p50", []) or []
                adj_series = (facts.get("adjusted", {}) or {}).get("p50_adj", []) or []
                if base_series and adj_series:
                    base_map = {
                        item.get("date"): float(item.get("mw"))
                        for item in base_series
                        if isinstance(item, dict) and item.get("date") and item.get("mw") is not None
                    }
                    deltas: list[tuple[str, float, float, float]] = []
                    for item in adj_series:
                        if not isinstance(item, dict):
                            continue
                        date = item.get("date")
                        adj_val = item.get("mw")
                        base_val = base_map.get(date)
                        if date is None or adj_val is None or base_val is None:
                            continue
                        delta = float(adj_val) - float(base_val)
                        if abs(delta) < 1e-5:
                            continue
                        deltas.append((date, delta, float(base_val), float(adj_val)))
                    if deltas:
                        # Keep chronological order (series already ordered)
                        next_12 = deltas[:12]
                        total_delta_12 = sum(d[1] for d in next_12) if next_12 else 0.0
                        avg_delta_12 = (total_delta_12 / len(next_12)) if next_12 else 0.0
                        largest_delta = max(deltas, key=lambda x: abs(x[1]))
                        direction_word = "increase" if total_delta_12 >= 0 else "decrease"
                        facts_narrative.append(
                            f"Scenario adjustments {direction_word} the forecast by {total_delta_12:+.1f} MW across the next 12 months "
                            f"(average shift {avg_delta_12:+.1f} MW). Largest change is {largest_delta[1]:+.1f} MW in {largest_delta[0]}, "
                            f"moving from {largest_delta[2]:.1f} MW to {largest_delta[3]:.1f} MW."
                        )
                        first_material = next((d for d in deltas if abs(d[1]) >= 0.5), None)
                        if first_material:
                            facts_narrative.append(
                                f"First material divergence occurs in {first_material[0]} at {first_material[1]:+.1f} MW vs. base."
                            )
            except Exception:
                pass
            
            # Combine all facts into a readable narrative
            facts_text = "\n".join(facts_narrative)
            
            # Create a professional system prompt that acts as a consultant
            system = (
                "You are a Senior Power Systems Planning Engineer with 20+ years of experience. "
                "You have just completed a detailed load forecast analysis and are now briefing the utility's planning committee.\n\n"
                "Your role is to:\n"
                "- Interpret the forecast results and provide actionable insights\n"
                "- Highlight critical findings, risks, and opportunities\n"
                "- Make specific recommendations with numbers, months, and magnitudes\n"
                "- Write in a professional, consultant-style narrative\n\n"
                "DO NOT describe data structures or JSON formats. "
                "DO NOT explain what the analysis contains. "
                "Instead, USE the data to tell a story about what the forecast reveals and what actions should be taken.\n\n"
                "CRITICAL RULES:\n"
                "1. Write as if speaking to executives - focus on business impact\n"
                "2. Use specific numbers, months (YYYY-MM format), and MW/GWh units\n"
                "3. Be concise but substantive - every sentence should add value\n"
                "4. Focus on actionable insights, not data description\n"
                "5. Never say 'the data shows' or 'according to the JSON' - just state the findings directly\n"
            )
            
            # Decide response style based on whether user kept the default template
            def _normalize_prompt_text(text: str) -> str:
                return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip()) if text else ""

            normalized_question = _normalize_prompt_text(question or "")
            normalized_default = _normalize_prompt_text(DEFAULT_AI_QUERY or "")
            use_structured_summary = bool(normalized_question) and normalized_question == normalized_default

            if use_structured_summary:
                task = (
                    "Based on the forecast analysis, craft a concise briefing using this structure:\n\n"
                    "## Summary\n"
                    "3-5 bullets on the most critical findings (peak demands, growth trends, compliance status)\n\n"
                    "## Key Risks & Concerns\n"
                    "Highlight 3-4 specific risks (capacity shortfalls, compliance issues, uncertainty factors)\n\n"
                    "## Recommendations\n"
                    "List 4-6 concrete action items (capacity additions, monitoring priorities, scenario analysis)\n\n"
                    "Write professionally and confidently. Be specific with numbers, timeframes, and units."
                )
            else:
                task = (
                    "Using the context above, answer the user's question directly. "
                    "Cite specific months (YYYY-MM), MW / GWh values, and scenario-adjusted impacts when relevant. "
                    "If information is unavailable, state 'n/a' rather than guessing."
                )
            
            question_clean = (question or "").strip()

            # The actual prompt - facts as narrative, not JSON
            prompt = (
                f"{system}\n\n"
                f"=== FORECAST ANALYSIS RESULTS ===\n\n"
                f"{facts_text}\n\n"
                f"=== END OF ANALYSIS DATA ===\n\n"
                f"{task}\n\n"
                f"User question: {question_clean or 'Provide a structured summary of the forecast.'}\n\n"
                f"Provide your expert analysis now:"
            )
            body = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(num_predict),
                    "num_ctx": int(num_ctx),
                    "top_p": 0.9,
                    "top_k": 40,
                    "stop": ["</think>"]
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as e:
            return f"AI call failed: {e}"

    # Build comprehensive, tidy facts payload (fallback or augmentation)
    def _build_chat_facts_tidy(results: dict, scenarios: list, *, compact: bool = True, max_months: int = 24, top_features: int = 8, prm_pct: float = 0.15) -> dict:
        out: dict[str, object] = {"error": None}
        try:
            fc: pd.DataFrame | None = results.get("forecast") if isinstance(results, dict) else None
            if fc is None or fc.empty:
                return {"error": "no_forecast"}
            base_col = "p50" if "p50" in fc.columns else "Ensemble"
            start = fc.index.min().strftime("%Y-%m")
            end = fc.index.max().strftime("%Y-%m")
            out["context"] = {
                "units": {"power": "MW", "energy": "GWh"},
                "interval": "monthly",
                "definitions": {
                    "p50": "Median monthly load forecast (MW)",
                    "p10": "Lower uncertainty bound (10th percentile, MW)",
                    "p90": "Upper uncertainty bound (90th percentile, MW)",
                    "RJPP": "Regulatory baseline monthly load (MW)",
                    "band_calibration": "Share of validation months within p10–p90",
                    "RJPP_compliance": "Share of months within RJPP ± bound",
                },
            }
            # Add headline meta and trend
            cagr = None
            try:
                cagr = _calc_cagr_series(fc[base_col])
            except Exception:
                cagr = None
            out["meta"] = {
                "forecast_start": start,
                "forecast_end": end,
                "ensemble_method": results.get("ensemble_choice"),
                "band_k": float(getattr(st.session_state.forecaster, "band_k", 1.0)),
                "max_dev_pct": float(results.get("max_dev", getattr(st.session_state.forecaster, "max_dev", 0.05)) * 100.0),
                "months": int(len(fc.index)),
                "cagr_pct_per_year": float(cagr * 100.0) if cagr is not None else None,
            }
            # Series (trim to last max_months when compact)
            def _series(df: pd.DataFrame, col: str) -> list[dict]:
                if col not in df.columns:
                    return []
                s = df[col].astype(float)
                if compact and isinstance(max_months, int) and max_months > 0:
                    s = s.iloc[-max_months:]
                return [{"date": d.strftime("%Y-%m"), "mw": float(v)} for d, v in s.items()]
            out["series"] = {
                "p50": _series(fc, base_col),
                "p10": _series(fc, "p10"),
                "p90": _series(fc, "p90"),
                "rjpp": _series(fc, "RJPP") if "RJPP" in fc.columns else [],
            }
            # History snapshot
            try:
                comb = results.get("combined_forecast")
                if isinstance(comb, pd.DataFrame) and "Combined" in comb.columns:
                    hist_end = fc.index.min() - pd.offsets.MonthBegin(1)
                    hist = comb["Combined"].loc[:hist_end].dropna()
                    last_actual = float(hist.iloc[-1]) if len(hist) else None
                    hist12 = hist.tail(12)
                    out["history"] = {
                        "last_actual_mw": last_actual,
                        "last_12m_avg_mw": float(hist12.mean()) if len(hist12) else None,
                        "last_12m_energy_gwh": float((hist12 * pd.Series(hist12.index.days_in_month, index=hist12.index) * 24).sum() / 1000) if len(hist12) else None,
                    }
            except Exception:
                out["history"] = {}

            # Next 12 months summary
            try:
                nxt = fc[base_col].astype(float).iloc[:12]
                hours = pd.Series(fc.index.days_in_month, index=fc.index, dtype=float).iloc[:12] * 24
                out["next_12m"] = {
                    "avg_mw": float(nxt.mean()) if len(nxt) else None,
                    "energy_gwh": float((nxt * hours).sum() / 1000) if len(nxt) else None,
                    "peak_mw": float(nxt.max()) if len(nxt) else None,
                    "peak_month": nxt.idxmax().strftime("%Y-%m") if len(nxt) else None,
                }
            except Exception:
                out["next_12m"] = {}

            # Peaks and uncertainty
            try:
                p50s = fc[base_col].astype(float)
                p10s = fc["p10"].astype(float) if "p10" in fc.columns else None
                p90s = fc["p90"].astype(float) if "p90" in fc.columns else None
                top_idx = p50s.nlargest(3).index if len(p50s) else []
                out["peaks"] = {
                    "top3": [
                        {"month": i.strftime("%Y-%m"), "mw": float(p50s.loc[i])}
                        for i in top_idx
                    ]
                }
                if p10s is not None and p90s is not None:
                    width = (p90s - p10s)
                    out["uncertainty"] = {
                        "avg_band_width_mw": float(width.mean()),
                        "max_band_width_mw": float(width.max()),
                    }
            except Exception:
                pass

            # Planning + compliance
            try:
                out["planning"] = {"assumed_prm_pct": float(prm_pct * 100)}
                if "RJPP" in fc.columns:
                    r = fc["RJPP"].replace(0, np.nan)
                    dev = ((fc[base_col] - r) / r).abs()
                    comp = float((dev <= (out["meta"]["max_dev_pct"] / 100.0)).mean() * 100)
                    out["planning"]["rjpp_compliance_pct"] = comp
                    # Rough monthly reserve check vs p90 + PRM
                    if "p90" in fc.columns:
                        need = fc["p90"] * (1.0 + prm_pct)
                        short = (need - fc[base_col]).clip(lower=0)
                        out["planning"]["avg_monthly_shortfall_mw_vs_p90_prm"] = float(short.mean())
            except Exception:
                pass
            # Validation & calibration
            out["validation"] = {
                "metrics_by_model": results.get("validation_results", {}),
                "ensemble_val_metrics": results.get("ensemble_val_metrics", {}),
                "band_calibration_pct": float(results.get("calibration_pct")) if results.get("calibration_pct") is not None else None,
            }
            # Annual summary
            try:
                hours = pd.Series(fc.index.days_in_month, index=fc.index, dtype=float) * 24
                energy_mwh = fc[base_col].astype(float) * hours
                annual = pd.DataFrame({
                    "avg_mw": fc[base_col].groupby(fc.index.year).mean().astype(float),
                    "energy_gwh": (energy_mwh.groupby(fc.index.year).sum() / 1000).astype(float),
                    "peak_mw": fc[base_col].groupby(fc.index.year).max().astype(float),
                })
                out["annual"] = {int(y): {k: float(v) for k, v in row.items()} for y, row in annual.iterrows()}
            except Exception:
                out["annual"] = {}
            # Scenarios
            fc_adj: pd.DataFrame | None = st.session_state.get("fc_adjusted")
            if isinstance(fc_adj, pd.DataFrame) and not fc_adj.empty:
                adj_col = "p50_adj" if "p50_adj" in fc_adj.columns else ("Ensemble_adj" if "Ensemble_adj" in fc_adj.columns else None)
                out["adjusted"] = {"p50_adj": _series(fc_adj, adj_col)} if adj_col else {}
            else:
                out["adjusted"] = {}
            out["scenarios"] = [{k: v for k, v in s.items() if not str(k).startswith("_")} for s in (scenarios or [])]
            # Weights & importance
            out["weights"] = results.get("model_weights", {})
            # Feature importance (top N per model if compact)
            feat = results.get("feature_importance", {}) or {}
            if compact and isinstance(feat, dict):
                trimmed = {}
                for m, d in feat.items():
                    if isinstance(d, dict):
                        top = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max(1, int(top_features))]
                        trimmed[m] = {k: float(v) for k, v in top}
                out["feature_importance"] = trimmed
            else:
                out["feature_importance"] = feat
            return out
        except Exception:
            return {"error": "build_failed"}

    # Clean, defaulted AI (no UI toggles)
    # Chat state
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    # Subtle styling for a cleaner AI section
    st.markdown(
        """
        <style>
        .stMetric { border: 1px solid rgba(255,255,255,0.08); padding: 8px 10px; border-radius: 10px; }
        .stTextArea textarea { font-size: 0.95rem; }
        .stButton>button { height: 42px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build compact facts once for AI context
    results_for_ai = st.session_state.get("results", {})
    scenarios_for_ai = st.session_state.get("scenarios", [])

    # Show existing chat
    for msg in st.session_state.ai_chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    # Simple input
    user_q = st.text_area("Your question to AI", value=DEFAULT_AI_QUERY, height=160)
    colx1, colx2 = st.columns([1, 0.5])
    ask_clicked = colx1.button("Ask AI", type="primary", use_container_width=True)
    clear_clicked = colx2.button("Clear chat", use_container_width=True)

    if clear_clicked:
        st.session_state.ai_chat_history = []
        st.rerun()

    if ask_clicked and user_q.strip():
        st.session_state.ai_chat_history.append({"role": "user", "content": user_q.strip()})
        st.session_state.ai_chat_history = st.session_state.ai_chat_history[-20:]
        with st.chat_message("assistant"):
            status = st.status("Generating…", expanded=True)
            with status:
                status.write("Extracting forecast metrics...")
                facts = _build_chat_facts_tidy(results_for_ai, scenarios_for_ai, compact=True, max_months=24, prm_pct=0.15)
                model_opts = _list_ollama_models_local()
                model = (model_opts[0] if model_opts else "llama3.2:3b")
                status.write(f"Using AI model: {model}")
                status.write("Building narrative context from forecast data...")
                try:
                    # Increased num_predict for longer responses and num_ctx for better context understanding
                    answer = _ollama_chat_local(model, user_q.strip(), facts, temperature=0.1, num_predict=800, num_ctx=4096, timeout_s=180)
                except Exception as e:
                    answer = f"AI call failed: {e}"
                status.update(label="Answer received", state="complete")
            final = sanitize_ai(answer)
            st.markdown(final or "_No response._")
        st.session_state.ai_chat_history.append({"role": "assistant", "content": final or ""})

    # Transcript download
    transcript_lines = []
    for m in st.session_state.ai_chat_history:
        who = "You" if m["role"] == "user" else "AI"
        transcript_lines.append(f"**{who}:**\n\n{m['content']}\n")
    st.download_button("Download transcript (.md)", "\n\n---\n\n".join(transcript_lines) or "", file_name="ai_insight_chat.md", mime="text/markdown", use_container_width=True)

    # Check if forecast is available
    if not results_for_ai or not results_for_ai.get("forecast") is not None:
        st.info("Run the forecast first to generate AI insights.")
