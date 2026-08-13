"""
Advanced ML Engine for Octavian
Implements massive and complex neural network models within an ensemble framework.
Includes LSTM, Transformer, and Deep Dense Networks weighted dynamically.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import warnings
import threading
import time
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OctavianML")

warnings.filterwarnings('ignore')

# --- Per-symbol analysis cache ---
# analyze_symbol_ensemble re-fits LSTM + Transformer + MLP + RF + GBM on
# EVERY call (~20s), so repeated analyses of identical price data (re-runs,
# multiple target checks on the same symbol) were paying the full training
# cost each time. A fingerprint-keyed, TTL-bounded cache makes identical
# inputs resolve instantly while genuinely new data still gets a fresh model.
_ANALYSIS_CACHE: Dict[str, Any] = {}
_ANALYSIS_CACHE_LOCK = threading.Lock()
_ANALYSIS_CACHE_TTL = 600.0   # seconds
_ANALYSIS_CACHE_MAX = 512


def _analysis_fingerprint(df: pd.DataFrame, symbol: str) -> str:
    """Stable fingerprint of the actual price data feeding the model."""
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return f"{symbol}:empty"
    close = close.round(4)
    tail = close.tail(120).values
    digest = __import__("hashlib").md5(tail.tobytes()).hexdigest()[:16]
    return f"{symbol}:{len(close)}:{digest}"


# --- PyTorch Deep Learning Models ---

class MarketLSTM(nn.Module):
    """
    Complex Long Short-Term Memory Network for Time Series Prediction.
    Captures temporal dependencies in price action.
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=3, output_dim=1):
        super(MarketLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc_1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_1(out[:, -1, :]) # Take last time step
        out = self.relu(out)
        out = self.fc_2(out)
        return out

class MarketTransformer(nn.Module):
    """
    Transformer Encoder for Financial Time Series.
    Uses self-attention to identify non-linear relationships across timeframes.
    """
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=3, output_dim=1):
        super(MarketTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model)) # Max sequence length 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Global Average Pooling
        x = self.fc_out(x)
        return x

# --- Ensemble Manager ---

class AdvancedEnsembleEngine:
    """
    Master Ensemble Model managing multiple complex neural networks and ML models.
    Weights outcomes based on a variety of factors including historical accuracy and volatility.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Deterministic initialization: online-learning nets are random-init and
        # undertrained by design (fast path), so seed them to keep forecasts
        # reproducible across calls/instances rather than coin-flip directions.
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize Neural Networks (8 massive comprehensive features)
        self.lstm_model = MarketLSTM(input_dim=8).to(self.device)
        self.transformer_model = MarketTransformer(input_dim=8).to(self.device)
        
        # Initialize Scikit-Learn Models (Deep Dense & Tree-based)
        self.mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=120, random_state=42, alpha=0.01) # Added L2 Regularization
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=4, random_state=42) # Reduced depth, added min_samples
        self.gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_leaf=4, random_state=42) # Reduced depth, added min_samples
        
        # Pre-train / Initialize weights with synthetic data to avoid random garbage in Fast Mode
        # self._pretrain_general_model() # DISABLED for performance - relies on online learning or saved state

        # Dynamic Weighting System (Initial weights)
        self.model_weights = {
            'LSTM_Neural_Net': 0.25,
            'Transformer_Attention': 0.25,
            'Deep_MLP_Network': 0.20,
            'Random_Forest_Ensemble': 0.15,
            'Gradient_Boosting_Machine': 0.15
        }
        
        self.scaler = MinMaxScaler()
        
    def auto_tune_weights(self, model_performance_history: Dict[str, List[float]]):
        """
        Phase 4: AutoML Model Auto-Tuning
        Actively re-weights sub-engines (LSTM vs. Transformer vs. Dense) based on their
        historical predictive accuracy (e.g., hit rate or Sharpe contribution).
        """
        if not model_performance_history:
            return
            
        total_score = 0.0
        updated_weights = {}
        
        # Calculate recent performance score (exponential moving average approximation)
        for model_name, history in model_performance_history.items():
            if not history:
                updated_weights[model_name] = self.model_weights.get(model_name, 0.2)
                continue
            
            # Weigh recent performance heavier
            weights = np.exp(np.linspace(-1, 0, len(history)))
            score = np.average(history, weights=weights)
            
            # Penalty for consistently bad performance (Pruning)
            if score < 0.45: # e.g., Worse than a coin flip
                score = score * 0.5 
                
            updated_weights[model_name] = max(0.01, score)
            total_score += updated_weights[model_name]
            
        # Normalize weights
        if total_score > 0:
            for model_name in updated_weights:
                self.model_weights[model_name] = updated_weights[model_name] / total_score
        
        logger.info(f"AutoML tuning complete. New weights: {self.model_weights}")

    def _pretrain_general_model(self):
        """Pre-train models on synthetic data to establish baseline weights."""
        try:
            # Generate synthetic market-like data (sine wave + noise + trend)
            t = np.linspace(0, 100, 500)
            price = 100 + 10 * np.sin(t/5) + 0.5 * t + np.random.normal(0, 2, 500)
            
            df_synth = pd.DataFrame({
                'Open': price + np.random.normal(0, 1, 500),
                'High': price + 2,
                'Low': price - 2,
                'Close': price,
                'Volume': np.abs(np.random.normal(10000, 2000, 500))
            })
            
            # Temporary scaler for pre-training
            temp_scaler = MinMaxScaler()
            
            # Feature engineering for synth data
            df_synth['Returns'] = df_synth['Close'].pct_change()
            df_synth['Volatility'] = df_synth['Returns'].rolling(window=20).std()
            df_synth = df_synth.fillna(0)
            
            features = ['Close', 'High', 'Low', 'Volume', 'Volatility']
            dataset = df_synth[features].values
            scaled_data = temp_scaler.fit_transform(dataset)
            
            lookback = 30
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i])
                y.append(scaled_data[i, 0])
                
            X, y = np.array(X), np.array(y)
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
            
            # Train PyTorch models briefly
            criterion = nn.MSELoss()
            optimizer_lstm = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)
            optimizer_trans = torch.optim.Adam(self.transformer_model.parameters(), lr=0.01)
            
            self.lstm_model.train()
            self.transformer_model.train()
            
            for _ in range(10): # 10 epochs
                optimizer_lstm.zero_grad()
                out = self.lstm_model(X_tensor)
                loss = criterion(out, y_tensor)
                loss.backward()
                optimizer_lstm.step()
                
                optimizer_trans.zero_grad()
                out = self.transformer_model(X_tensor)
                loss = criterion(out, y_tensor)
                loss.backward()
                optimizer_trans.step()
                
            # Train Sklearn models
            X_flat = X.reshape(X.shape[0], -1)
            self.mlp_model.fit(X_flat, y)
            self.rf_model.fit(X_flat, y)
            self.gbm_model.fit(X_flat, y)
            
        except Exception as e:
            logger.warning(f"Pre-training failed: {e}")

    def _prepare_data(self, df: pd.DataFrame, symbol: str, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare massive institutional data for training/inference from a DataFrame."""
        if len(df) < lookback + 20:
            return None, None, None
            
        # Feature Engineering — robust to price-only inputs (e.g. a bare Close series
        # from a scanner/backtester). Missing OHLC is synthesized from Close using the
        # standard convention Open=prev Close, High/Low = small band around Close,
        # which keeps the 8-feature model pipeline fully functional for any input.
        cols_to_get = []
        if 'Close' in df.columns:
            cols_to_get.append('Close')
        if 'Open' in df.columns:
            cols_to_get.append('Open')
        if 'High' in df.columns:
            cols_to_get.append('High')
        if 'Low' in df.columns:
            cols_to_get.append('Low')
        if 'Volume' in df.columns:
            cols_to_get.append('Volume')
        if not cols_to_get:
            # No recognized columns at all — fall back to the first numeric column
            cols_to_get = [df.columns[0]]

        data = df[cols_to_get].copy()

        # Normalize column presence so downstream code can rely on a canonical schema
        if 'Close' not in data.columns:
            data['Close'] = data.iloc[:, 0]
        if 'Open' not in data.columns:
            # Standard convention: Open = previous Close (first row = Close)
            data['Open'] = data['Close'].shift(1).fillna(data['Close'])
        if 'High' not in data.columns:
            data['High'] = data[['Open', 'Close']].max(axis=1) * 1.001
        if 'Low' not in data.columns:
            data['Low'] = data[['Open', 'Close']].min(axis=1) * 0.999

        # Ensure Volume exists for the models even if not in data
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        
        # Add basic technical features for the ML models
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Massive comprehensive indicators
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        
        # Incorporate Trades Filed (Institutional 13F Flow) directly into training data
        try:
            from sec_13f_engine import get_sec_13f_engine
            sec_engine = get_sec_13f_engine()
            flows = sec_engine.get_global_smart_money_flow()
            net_flow_map = flows.get("net_flow_map", {})
            symbol_flow = net_flow_map.get(symbol, 0)
        except:
            symbol_flow = 0
            
        data['Inst_Flow'] = symbol_flow / 1e9 # Normalize to Billions for the neural network
        
        data = data.fillna(0)
        
        # Select 8 core features for massive comprehensive input dimension
        features = ['Close', 'High', 'Low', 'Volume', 'Volatility', 'RSI', 'MACD', 'Inst_Flow']
        dataset = data[features].values
        
        scaled_data = self.scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0]) # Predicting next Close (normalized)
            
        X, y = np.array(X), np.array(y)
        
        # Last sequence for future prediction
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, len(features))
        
        return X, y, last_sequence

    def analyze_symbol_ensemble(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        asset_type: str = "STOCK",
        options_context: Dict[str, Any] = None,
        fast_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Backward-compatible ensemble interface used across quant/optimizer/portal.
        """
        df = data
        if df is None or df.empty:
            return {
                "decision": "NEUTRAL",
                "confidence": 0.0,
                "predicted_return": 0.0,
                "alpha_score": 50.0,
                "driving_factor": "FUNDAMENTAL",
                "weights": {"fundamental": 0.45, "technical": 0.30, "sentiment": 0.15, "options": 0.10},
                "factors": {"fundamental": 50.0, "technical": 50.0, "sentiment": 50.0, "options": 50.0},
                "options_edge": None,
                "volatility": 0.25,
                "model_breakdown": {},
            }

        lookback = 30
        X, y, last_seq = self._prepare_data(df, symbol=symbol, lookback=lookback)
        
        if X is None or len(X) < 10:
            return {
                "decision": "NEUTRAL",
                "confidence": 0.0,
                "predicted_return": 0.0,
                "alpha_score": 50.0,
                "driving_factor": "FUNDAMENTAL",
                "weights": {"fundamental": 0.45, "technical": 0.30, "sentiment": 0.15, "options": 0.10},
                "factors": {"fundamental": 50.0, "technical": 50.0, "sentiment": 50.0, "options": 50.0},
                "options_edge": None,
                "volatility": 0.25,
                "model_breakdown": {},
            }
            
        # --- Fast path: identical data already analyzed recently ---
        fp = _analysis_fingerprint(df, symbol)
        _now = time.time()
        with _ANALYSIS_CACHE_LOCK:
            _hit = _ANALYSIS_CACHE.get(fp)
            if _hit is not None and (_now - _hit[0]) < _ANALYSIS_CACHE_TTL:
                return _hit[1]

        # --- Online Learning (Rapid Adaptation) ---
        # We train the models on the specific asset's recent history to adapt to its current regime.
        
        # Convert numpy arrays to PyTorch Tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
        last_seq_tensor = torch.FloatTensor(last_seq).to(self.device)
        criterion = nn.MSELoss()
        
        # Train PyTorch Models (Short epochs for speed)
        # Incorporating L2 Regularization (weight_decay)
        optimizer_lstm = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01, weight_decay=1e-4)
        optimizer_trans = torch.optim.Adam(self.transformer_model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # Reduce epochs for speed — fast_mode uses 2 epochs, normal uses 3 max.
        # Direction is anchored by the Kalman-smoothed trend (65% weight), so
        # a couple of online-learning epochs is enough; the old 5-epoch default
        # roughly doubled training time for no measurable directional gain.
        # IMPORTANT: Sklearn models fit ONCE outside the loop (not O(epochs) times)
        epochs = 2 if fast_mode else 3
        
        # Train LSTM
        self.lstm_model.train()
        for _ in range(epochs):
            optimizer_lstm.zero_grad()
            out = self.lstm_model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer_lstm.step()

        # Train Transformer (separate loop — not nested!)
        self.transformer_model.train()
        for _ in range(epochs):
            optimizer_trans.zero_grad()
            out = self.transformer_model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer_trans.step()

        # Train Sklearn Models ONCE (not inside PyTorch loop)
        X_flat = X.reshape(X.shape[0], -1)
        self.mlp_model.fit(X_flat, y)
        self.rf_model.fit(X_flat, y)
        self.gbm_model.fit(X_flat, y)
        
        # --- Ensemble Inference ---
        self.lstm_model.eval()
        self.transformer_model.eval()
        
        with torch.no_grad():
            pred_lstm = self.lstm_model(last_seq_tensor).item()
            pred_trans = self.transformer_model(last_seq_tensor).item()
            
        last_seq_flat = last_seq.reshape(1, -1)
        
        # Check if sklearn models are fitted
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.mlp_model)
            pred_mlp = self.mlp_model.predict(last_seq_flat)[0]
            pred_rf = self.rf_model.predict(last_seq_flat)[0]
            pred_gbm = self.gbm_model.predict(last_seq_flat)[0]
        except:
            # If not fitted (should happen only if pretrain failed), fit on current batch
            X_flat = X.reshape(X.shape[0], -1)
            self.mlp_model.fit(X_flat, y)
            self.rf_model.fit(X_flat, y)
            self.gbm_model.fit(X_flat, y)
            pred_mlp = self.mlp_model.predict(last_seq_flat)[0]
            pred_rf = self.rf_model.predict(last_seq_flat)[0]
            pred_gbm = self.gbm_model.predict(last_seq_flat)[0]
        
        # --- Inverse Transform Predictions ---
        # We need to invert the scaling to get actual price targets
        # The scaler was fit on [Close, High, Low, Vol, Volatility]
        # We only care about the first column (Close)
        
        current_close = df['Close'].iloc[-1]
        
        def inverse_price(pred_val):
            # Create a dummy row with the predicted close and 0s for others
            dummy = np.zeros((1, 8))
            dummy[0, 0] = pred_val
            return self.scaler.inverse_transform(dummy)[0, 0]
            
        target_lstm = inverse_price(pred_lstm)
        target_trans = inverse_price(pred_trans)
        target_mlp = inverse_price(pred_mlp)
        target_rf = inverse_price(pred_rf)
        target_gbm = inverse_price(pred_gbm)
        
        predictions = {
            'LSTM_Neural_Net': target_lstm,
            'Transformer_Attention': target_trans,
            'Deep_MLP_Network': target_mlp,
            'Random_Forest_Ensemble': target_rf,
            'Gradient_Boosting_Machine': target_gbm
        }
        
        # --- Institutional Enrichment ---
        # 1. Denoise the price action before final decision
        kalman_series = self._kalman_filter(df['Close'].values)
        kalman_price = kalman_series[-1]
        vol_forecast = self._calculate_garch_vol(df)
        
        # --- Regression-to-mean bias correction ---
        # Undertrained regressors (short online-learning horizon) predict near the
        # training-window mean close, which mechanically biases forecasts DOWNWARD on
        # uptrends and UPWARD on downtrends (a strong linear uptrend would otherwise
        # print a false SELL). De-mean each model target against the window mean so the
        # ML contribution is residual alpha around the level, not a level artifact.
        window_mean_close = float(df['Close'].iloc[-min(len(df), 90):].mean())
        raw_predictions = dict(predictions)  # keep raw targets for transparency
        for _name in list(predictions.keys()):
            _denom = max(abs(window_mean_close), 1e-9)
            predictions[_name] = current_close * (1.0 + (predictions[_name] - window_mean_close) / _denom)
        
        # 2. Robust Weighted Voting (trimmed mean) — Regime-Aware
        # Undertrained online-learning nets can emit pathological outliers (e.g. a
        # Transformer that hasn't converged in a handful of epochs). A trimmed
        # ensemble drops the best and worst sub-model target each forecast, then
        # weights the remaining members — robust to any single model failure.
        dynamic_weights = self.model_weights.copy()
        if vol_forecast > 0.30: # High vol regime
            dynamic_weights['Transformer_Attention'] *= 1.2
            dynamic_weights['Random_Forest_Ensemble'] *= 1.2
            dynamic_weights['Deep_MLP_Network'] *= 0.8
            
        _items = sorted(predictions.items(), key=lambda kv: kv[1])
        _keep = _items[1:-1]  # drop the single best and worst sub-model target
        if len(_keep) < 2:
            _keep = _items
        weighted_sum = 0.0
        total_weight = 0.0
        for name, pred in _keep:
            w = dynamic_weights.get(name, 0.15)
            weighted_sum += pred * w
            total_weight += w
            
        ensemble_forecast = weighted_sum / max(total_weight, 1e-9)
        
        # 3. Trend anchor: extrapolate the Kalman-smoothed drift over a short lookback.
        # Keeps the forecast aligned with the actual recent trend even when the
        # ML sub-models are uncertain (institutional fallback for low-signal regimes).
        drift_lookback = min(10, max(1, len(kalman_series) - 1))
        kalman_prev = kalman_series[-1 - drift_lookback]
        kalman_drift = (kalman_price - kalman_prev) / max(abs(kalman_prev), 1e-9)
        kalman_drift = float(np.clip(kalman_drift, -0.15, 0.15))
        trend_anchor = current_close * (1.0 + kalman_drift)
        
        # Blend the ensemble forecast with the trend anchor.
        # The anchor gets the larger weight: online-learning sub-models are
        # undertrained by design (fast path), so the Kalman-smoothed technical
        # trend is the more reliable directional backbone; the ensemble adds
        # residual alpha rather than dictating direction.
        final_predicted_price = (ensemble_forecast * 0.35) + (trend_anchor * 0.65)
        
        # --- 13F Smart Money Flow Integration ---
        try:
            from sec_13f_engine import get_sec_13f_engine
            sec_engine = get_sec_13f_engine()
            flows = sec_engine.get_global_smart_money_flow()
            net_flow_map = flows.get("net_flow_map", {})
            symbol_flow = net_flow_map.get(symbol, 0)
            
            # Adjust price target based on massive institutional flow
            # A $1B inflow pushes price target slightly higher
            if symbol_flow > 0:
                flow_boost = min(0.02, (symbol_flow / 1e10)) # Cap at 2% boost
                final_predicted_price *= (1 + flow_boost)
            elif symbol_flow < 0:
                flow_drag = min(0.02, (abs(symbol_flow) / 1e10))
                final_predicted_price *= (1 - flow_drag)
        except Exception as e:
            logger.warning(f"Failed to integrate 13F smart money flow: {e}")
            
        predicted_return = (final_predicted_price - current_close) / current_close
        
        # 3. Monte Carlo Probabilistic Forecast
        mc_data = self.compute_probabilistic_targets(current_close, vol_forecast)
        
        # Determine Confidence based on directional model agreement and volatility
        # Use the trimmed (non-outlier) sub-model targets for a robust dispersion signal
        trimmed_values = [p for _, p in _keep]
        pred_std = np.std(trimmed_values)
        direction_votes = sum(1 for v in trimmed_values if (v - current_close) * (final_predicted_price - current_close) > 0)
        agreement = direction_votes / max(len(trimmed_values), 1)
        confidence = 0.15 + agreement * 0.65 - min(0.2, vol_forecast)
        confidence = float(np.clip(confidence, 0.10, 0.95))
        
        decision = "HOLD"
        if predicted_return > 0.005: 
            decision = "BUY"
        elif predicted_return < -0.005: 
            decision = "SELL"
            
        out = {
            "decision": decision,
            "confidence": float(confidence),
            "predicted_return": float(predicted_return),
            "final_predicted_price": float(final_predicted_price),
            "current_price": float(current_close),
            "model_breakdown": predictions,
            "raw_model_targets": raw_predictions,
            "volatility_forecast": vol_forecast,
            "monte_carlo": mc_data,
            "is_institutional": True
        }

        # Normalize decision to platform standard
        raw_decision = str(out.get("decision", "HOLD")).upper()
        if raw_decision == "BUY":
            norm_decision = "BULLISH"
        elif raw_decision == "SELL":
            norm_decision = "BEARISH"
        else:
            norm_decision = "NEUTRAL"

        vol = float(out.get("volatility_forecast", 0.25))
        alpha_score = float(np.clip(50.0 + out.get("predicted_return", 0.0) * 300.0, 0.0, 100.0))

        # Derive factor/weight compatibility fields
        model_weights = self.get_model_summary() if hasattr(self, "get_model_summary") else {}
        w_sum = max(sum(model_weights.values()), 1e-9)
        weights = {
            "fundamental": 0.45,
            "technical": 0.30,
            "sentiment": 0.15,
            "options": 0.10,
        }
        factors = {
            "fundamental": float(np.clip(50 + out.get("predicted_return", 0) * 120, 0, 100)),
            "technical": float(np.clip(50 + out.get("predicted_return", 0) * 180, 0, 100)),
            "sentiment": float(np.clip(50 + (out.get("confidence", 0.5) - 0.5) * 80, 0, 100)),
            "options": 50.0,
        }

        # Optional options overlay
        options_edge = None
        if options_context:
            try:
                import options_engine
                oe = options_engine.get_options_engine()
                options_edge = oe.predict_option_edge(
                    spot=float(options_context["spot"]),
                    strike=float(options_context["strike"]),
                    dte_days=int(options_context["dte_days"]),
                    iv=float(options_context["iv"]),
                    option_type=str(options_context.get("option_type", "call")),
                    market_view=float(options_context.get("market_view", 0.0)),
                    confidence=float(options_context.get("confidence", 0.5)),
                )
                factors["options"] = float(np.clip(50 + options_edge["edge_score"] * 45, 0, 100))
            except Exception:
                pass

        out.update({
            "decision": norm_decision,
            "alpha_score": alpha_score,
            "driving_factor": "TECHNICAL" if vol > 0.35 else "FUNDAMENTAL",
            "weights": weights,
            "factors": factors,
            "options_edge": options_edge,
            "volatility": vol,
        })

        # Store in the bounded TTL cache so identical re-analyses are instant.
        with _ANALYSIS_CACHE_LOCK:
            _ANALYSIS_CACHE[fp] = (time.time(), out)
            if len(_ANALYSIS_CACHE) > _ANALYSIS_CACHE_MAX:
                _old = sorted(
                    _ANALYSIS_CACHE, key=lambda k: _ANALYSIS_CACHE[k][0]
                )[: len(_ANALYSIS_CACHE) // 4]
                for _k in _old:
                    _ANALYSIS_CACHE.pop(_k, None)
        return out

    def get_model_summary(self) -> Dict[str, Any]:
        """Return summary of model weights."""
        return self.model_weights

    # 
    # INSTITUTIONAL MODULES
    # 

    def _kalman_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Denoise price action using a steady-state Kalman Filter.
        Removes 'market noise' while preserving structural trend changes.
        """
        if len(data) < 2: return data
        n_iter = len(data)
        sz = (n_iter,)
        xhat = np.zeros(sz)      # a posteri estimate of x
        P = np.zeros(sz)         # a posteri error estimate
        xhatminus = np.zeros(sz) # a priori estimate of x
        Pminus = np.zeros(sz)    # a priori error estimate
        K = np.zeros(sz)         # gain or blending factor

        Q = 1e-5 # process variance
        R = 0.01**2 # estimate of measurement variance
        
        xhat[0] = data[0]
        P[0] = 1.0

        for k in range(1, n_iter):
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
            
        return xhat

    def _calculate_garch_vol(self, df: pd.DataFrame) -> float:
        """
        Recursive GARCH(1,1) variance estimation.
        Institutional-standard volatility forecasting.
        """
        try:
            rets = df["Close"].pct_change().dropna()
            if len(rets) < 50: return 0.20
            
            omega, alpha, beta = 1e-6, 0.1, 0.8
            sigma_sq = rets.var()
            
            for r in rets.values[-100:]:
                sigma_sq = omega + alpha * (r**2) + beta * sigma_sq
                
            return float(np.sqrt(sigma_sq * 252))
        except:
            return 0.20

    def compute_probabilistic_targets(self, current_price: float, vol: float) -> dict:
        """
        Monte Carlo price path simulation (5,000 paths).
        Provides a probabilistic distribution of potential returns.
        """
        horizon = 21 # 1 month
        sims = 5000
        daily_vol = vol / np.sqrt(252)
        
        # Log-normal paths
        sim_rets = np.random.normal(0, daily_vol, (horizon, sims))
        paths = current_price * np.exp(np.cumsum(sim_rets, axis=0))
        
        final_prices = paths[-1, :]
        return {
            "expected_target": float(np.mean(final_prices)),
            "upside_95": float(np.percentile(final_prices, 95)),
            "downside_5": float(np.percentile(final_prices, 5)),
            "prob_profit": float(np.mean(final_prices > current_price))
        }

# --- Singleton Instance (Lazy Loading handled by caller or Streamlit cache) ---
# Removed global instantiation to prevent import-time training lag
# ensemble_engine = AdvancedEnsembleEngine() 

@st.cache_resource
def get_ensemble_engine():
    """Factory to get singleton instance."""
    return AdvancedEnsembleEngine()

def get_ml_engine():
    """Compatibility alias for legacy callers."""
    return get_ensemble_engine()
