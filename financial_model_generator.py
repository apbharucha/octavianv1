"""
Octavian Financial Model Generator
Institutional-grade DCF valuation engine
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import io
from historical_data_engine import get_historical_engine
import logging
import importlib
import inspect
import sys
from functools import wraps

# --- Runtime-safe DCF patch ---
def _apply_financial_runtime_safety_patch():
    """
    Additive runtime patch:
    - Wrap InstitutionalDCFEngine.run_dcf
    - Guarantee stable DCFResult shape
    - Prevent downstream crashes from partial/legacy paths
    """
    engine_cls = InstitutionalDCFEngine
    run_fn = getattr(engine_cls, "run_dcf", None)
    if run_fn is None or getattr(run_fn, "_octavian_runtime_safe", False):
        return

    @wraps(run_fn)
    def _wrapped_run_dcf(self, assumptions: DCFAssumptions) -> DCFResult:
        try:
            result = run_fn(self, assumptions)
            if result is None:
                raise ValueError("run_dcf returned None")

            # Normalize key fields (non-destructive)
            if result.assumptions is None:
                result.assumptions = assumptions
            if result.trade_signal is None:
                result.trade_signal = TradeSignal(
                    ticker=assumptions.ticker,
                    fair_value=float(result.fair_value_per_share or 0.0),
                    market_price=float(assumptions.current_price or 0.0),
                    upside_pct=(
                        ((float(result.fair_value_per_share or 0.0) / float(assumptions.current_price)) - 1.0) * 100.0
                        if float(assumptions.current_price or 0.0) > 0 else 0.0
                    ),
                    signal="Neutral",
                    confidence_pct=50.0,
                    risk_level="Medium",
                    risk_adjusted_return=0.0,
                    position_size_pct=0.0,
                    rationale="Runtime fallback signal generated for structural safety.",
                )

            result.trade_signal.ticker = assumptions.ticker
            result.scenarios = result.scenarios or []
            result.relative_valuation = result.relative_valuation or {}
            result.wacc_breakdown = result.wacc_breakdown or {}
            result.expected_returns = result.expected_returns or {}
            result.mc_distribution = result.mc_distribution or [float(assumptions.current_price or 0.0)]
            result.mc_percentiles = result.mc_percentiles or {
                "p5": float(np.percentile(result.mc_distribution, 5)),
                "p25": float(np.percentile(result.mc_distribution, 25)),
                "p50": float(np.percentile(result.mc_distribution, 50)),
                "p75": float(np.percentile(result.mc_distribution, 75)),
                "p95": float(np.percentile(result.mc_distribution, 95)),
            }
            result.visualization_payload = result.visualization_payload or {}
            result.probabilistic_forecast = result.probabilistic_forecast or {}

            return result

        except Exception as e:
            logging.getLogger(__name__).exception("run_dcf failed; returning safe fallback: %s", e)
            safe_price = float(assumptions.current_price or 0.0)
            fallback = DCFResult(
                ticker=assumptions.ticker,
                assumptions=assumptions,
                fair_value_per_share=safe_price,
                enterprise_value=0.0,
                equity_value=0.0,
                wacc=float(getattr(assumptions, "risk_free_rate", 0.10) + 0.05),
                cost_of_equity=float(getattr(assumptions, "risk_free_rate", 0.04) + getattr(assumptions, "equity_risk_premium", 0.06)),
                line_items=pd.DataFrame(),
                scenarios=[],
                mc_distribution=[safe_price],
                mc_median=safe_price,
                mc_mean=safe_price,
                mc_std=0.0,
                mc_upside_prob=0.5,
                mc_downside_prob=0.5,
                mc_percentiles={"p5": safe_price, "p25": safe_price, "p50": safe_price, "p75": safe_price, "p95": safe_price},
                relative_valuation={},
                sensitivity=pd.DataFrame(),
                catalysts=[],
                wacc_breakdown={},
                expected_returns={},
                historical_data=pd.DataFrame(),
                historical_stats={},
                forecast_vs_historical={},
                stress_tests=[],
                regime_adjusted_values={},
                diagnostic_flags=["run_dcf_exception_fallback"],
                visualization_payload={},
                probabilistic_forecast={
                    "horizon_years": assumptions.projection_years,
                    "expected_price": safe_price,
                    "prob_up_10pct": 0.0,
                    "prob_down_10pct": 0.0,
                    "distribution_moments": {"mean": safe_price, "std": 0.0, "skew_proxy": 0.0},
                },
                trade_signal=TradeSignal(
                    ticker=assumptions.ticker,
                    fair_value=safe_price,
                    market_price=safe_price,
                    upside_pct=0.0,
                    signal="Neutral",
                    confidence_pct=0.0,
                    risk_level="High",
                    risk_adjusted_return=0.0,
                    position_size_pct=0.0,
                    rationale="Fallback result due to internal DCF runtime error.",
                ),
            )
            return fallback

    _wrapped_run_dcf._octavian_runtime_safe = True
    setattr(engine_cls, "run_dcf", _wrapped_run_dcf)


# Auto-apply once on import (additive, idempotent)
try:
    _apply_financial_runtime_safety_patch()
except Exception:
    pass

# 
# DATA CLASSES
# 

def _patch_alt_data_signal() -> Dict[str, Any]:
    """
    Add backward-compatible `signal_type` property to AltDataSignal if missing.
    """
    patched = False
    details = []
    candidate_modules = [
        "alternative_data_engine",
        "alt_data_engine",
        "market_alt_data",
        "advanced_backtester",
        "backtester",
        "simulation_hub",
    ]

    # include already-loaded modules too
    loaded = list(sys.modules.keys())
    for mod_name in list(dict.fromkeys(candidate_modules + loaded)):
        try:
            mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        except Exception:
            continue

        cls = getattr(mod, "AltDataSignal", None)
        if cls is None:
            continue

        if hasattr(cls, "signal_type"):
            details.append(f"{mod_name}.AltDataSignal already has signal_type")
            continue

        def _get_signal_type(self):
            for attr in ("type", "signal", "category", "event_type", "label"):
                if hasattr(self, attr):
                    v = getattr(self, attr)
                    if v is not None:
                        return v
            return "unknown"

        def _set_signal_type(self, value):
            for attr in ("type", "signal", "category", "event_type", "label"):
                if hasattr(self, attr):
                    setattr(self, attr, value)
                    return
            # fallback: create dynamic attr
            setattr(self, "_signal_type_compat", value)

        try:
            setattr(cls, "signal_type", property(_get_signal_type, _set_signal_type))
            patched = True
            details.append(f"patched {mod_name}.AltDataSignal.signal_type")
        except Exception as e:
            details.append(f"failed patch {mod_name}.AltDataSignal: {e}")

    return {"patched": patched, "details": details}


def _patch_advanced_backtester_init() -> Dict[str, Any]:
    """
    Make AdvancedBacktester.init backward-compatible with `symbol=...`.
    """
    patched = False
    details = []
    candidate_modules = [
        "advanced_backtester",
        "backtester",
        "simulation_hub",
    ]

    loaded = list(sys.modules.keys())
    for mod_name in list(dict.fromkeys(candidate_modules + loaded)):
        try:
            mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        except Exception:
            continue

        cls = getattr(mod, "AdvancedBacktester", None)
        if cls is None:
            continue

        init_fn = getattr(cls, "init", None)
        if init_fn is None or getattr(init_fn, "_octavian_symbol_compat", False):
            continue

        try:
            sig = inspect.signature(init_fn)
            params = sig.parameters
            if "symbol" in params:
                details.append(f"{mod_name}.AdvancedBacktester.init already supports symbol")
                continue

            @wraps(init_fn)
            def _wrapped_init(self, *args, **kwargs):
                if "symbol" in kwargs:
                    symbol_value = kwargs.pop("symbol")
                    if "ticker" in params and "ticker" not in kwargs:
                        kwargs["ticker"] = symbol_value
                    elif "asset" in params and "asset" not in kwargs:
                        kwargs["asset"] = symbol_value
                    elif "instrument" in params and "instrument" not in kwargs:
                        kwargs["instrument"] = symbol_value
                    # else: silently drop symbol for strict signatures
                return init_fn(self, *args, **kwargs)

            setattr(_wrapped_init, "_octavian_symbol_compat", True)
            setattr(cls, "init", _wrapped_init)
            patched = True
            details.append(f"patched {mod_name}.AdvancedBacktester.init symbol compatibility")
        except Exception as e:
            details.append(f"failed patch {mod_name}.AdvancedBacktester.init: {e}")

    return {"patched": patched, "details": details}


def apply_backward_compatibility_patches() -> Dict[str, Any]:
    """
    Apply runtime compatibility patches for cross-module API mismatches.
    Safe, additive, and idempotent.
    """
    alt_data_result = _patch_alt_data_signal()
    backtester_result = _patch_advanced_backtester_init()
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "alt_data_signal": alt_data_result,
        "advanced_backtester": backtester_result,
    }

    logger = logging.getLogger(__name__)
    logger.info("Compatibility patch summary: %s", summary)
    return summary

_dcf_engine_instance = None
_financial_generator_instance = None
_compat_patch_summary = None


@dataclass
class DCFAssumptions:
    """Full set of DCF model assumptions."""

    ticker: str
    # Revenue
    base_revenue: float  # $M - most recent annual revenue
    revenue_growth_rates: List[
        float
    ]  # year-by-year growth rates (list len = projection_years)
    # Margins
    ebit_margin: float  # EBIT / Revenue
    tax_rate: float  # effective tax rate
    da_pct_revenue: float  # D&A as % of revenue
    capex_pct_revenue: float  # CapEx as % of revenue
    nwc_change_pct_revenue: float  # NWC as % of revenue
    # WACC components
    equity_value_market: float  # market cap $M  (E)
    debt_value: float  # total debt $M  (D)
    cost_of_debt: float  # pre-tax cost of debt Rd
    risk_free_rate: float  # Rf (10Y Treasury yield)
    equity_risk_premium: float  # Rm - Rf
    beta: float  # levered beta
    # Terminal value
    terminal_growth_rate: float  # g (Gordon Growth)
    # Balance sheet
    cash: float  # $M
    shares_outstanding: float  # millions
    current_price: float  # market price per share
    # Optional: peer multiples for relative valuation
    peer_pe: float = 25.0
    peer_ev_ebitda: float = 15.0
    peer_ev_fcf: float = 20.0
    peg_ratio: float = 1.5
    projection_years: int = 5


@dataclass
class ScenarioResult:
    """Scenario analysis result."""

    label: str
    probability: float
    fair_value: float
    upside: float
    revenue_growth_avg: float
    ebit_margin: float
    wacc: float
    terminal_growth: float  # ADD THIS FIELD

    # Institutional scenario engine (Section 8/9 of the upgrade spec):
    # each scenario is a *meaningfully different and internally coherent*
    # outcome with transparent, explainable probability drivers.
    evidence: str = ""
    trigger: str = ""
    key_risks: str = ""
    key_catalysts: str = ""
    rationale: str = ""
    driver_changes: str = ""  # which variables move vs base
    probability_basis: str = ""  # how the probability was derived


@dataclass
class CatalystEvent:
    name: str
    date: str
    impact: str  # "High" / "Medium" / "Low"
    direction: str  # "Positive" / "Negative" / "Neutral"
    description: str


@dataclass
class TradeSignal:
    ticker: str
    fair_value: float
    market_price: float
    upside_pct: float
    signal: str  # "Strong Long" / "Long" / "Neutral" / "Short" / "Strong Short"
    confidence_pct: float
    risk_level: str  # "Low" / "Medium" / "High"
    risk_adjusted_return: float
    position_size_pct: float
    rationale: str


@dataclass
class DCFResult:
    """Complete DCF analysis result with all institutional features."""

    ticker: str
    assumptions: DCFAssumptions = None  # Make optional with default
    
    # 20-Line IB Template
    line_items: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Valuation summary
    sum_pv_fcf: float = 0.0
    pv_terminal_value: float = 0.0
    enterprise_value: float = 0.0
    net_debt: float = 0.0
    equity_value: float = 0.0
    shares_outstanding: float = 0.0  # Make optional with default
    fair_value_per_share: float = 0.0

    # Scenario analysis
    scenarios: List[ScenarioResult] = field(default_factory=list)
    scenario_weighted_value: float = 0.0

    # Monte Carlo
    mc_median: float = 0.0
    mc_mean: float = 0.0
    mc_std: float = 0.0  # Add default value
    mc_upside_prob: float = 0.0
    mc_downside_prob: float = 0.0
    mc_percentiles: Dict[str, float] = field(default_factory=dict)
    mc_distribution: List[float] = field(default_factory=list)

    # Relative valuation
    relative_valuation: Dict[str, float] = field(default_factory=dict)

    # Sensitivity table
    sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Catalyst events
    catalysts: List[CatalystEvent] = field(default_factory=list)

    # Trade signal
    trade_signal: TradeSignal = None

    # WACC detail
    wacc: float = 0.0
    cost_of_equity: float = 0.0
    wacc_breakdown: Dict[str, float] = field(default_factory=dict)

    # NEW: Expected Investor Return fields
    expected_returns: Dict[str, float] = field(default_factory=dict)
    probability_weighted_irr: float = 0.0
    investment_recommendation: str = ""
    recommendation_color: str = "#aaaaaa"

    # NEW: Reverse DCF fields
    market_implied_growth: float = 0.0
    market_implied_ebit_margin: float = 0.0
    market_implied_fcf_growth: float = 0.0
    market_expectation_label: str = ""
    market_expectation_color: str = "#aaaaaa"

    # NEW: Historical anchoring fields
    historical_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_stats: Dict[str, float] = field(default_factory=dict)
    forecast_vs_historical: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # NEW: Institutional additive outputs
    stress_tests: List[Dict[str, Any]] = field(default_factory=list)
    regime_adjusted_values: Dict[str, float] = field(default_factory=dict)
    diagnostic_flags: List[str] = field(default_factory=list)
    visualization_payload: Dict[str, Any] = field(default_factory=dict)

    # NEW: additive probabilistic forecast block
    probabilistic_forecast: Dict[str, Any] = field(default_factory=dict)

    # NEW: institutional upgrade outputs (Sections 15/24/25/26/27)
    terminal_diagnostics: Dict[str, Any] = field(default_factory=dict)
    key_assumption_attribution: Dict[str, Any] = field(default_factory=dict)
    consensus_layer: Dict[str, Any] = field(default_factory=dict)
    dynamic_scenario_notes: str = ""


# 
# CORE ENGINE
# 

from openpyxl import Workbook

class InstitutionalDCFEngine:
    """Full institutional DCF engine with all advanced features."""

    #  WACC 

    def compute_wacc(self, a: DCFAssumptions) -> Tuple[float, float, Dict]:
        """
        WACC = (E/(D+E)) * Re + (D/(D+E)) * Rd * (1 - T)
        Re = Rf +  * (Rm - Rf)   [CAPM]
        """
        re = a.risk_free_rate + a.beta * a.equity_risk_premium
        total = a.equity_value_market + a.debt_value
        if total <= 0:
            total = 1.0
        w_e = a.equity_value_market / total
        w_d = a.debt_value / total
        wacc = w_e * re + w_d * a.cost_of_debt * (1 - a.tax_rate)
        detail = {
            "cost_of_equity": re,
            "cost_of_debt_pretax": a.cost_of_debt,
            "cost_of_debt_aftertax": a.cost_of_debt * (1 - a.tax_rate),
            "weight_equity": w_e,
            "weight_debt": w_d,
            "wacc": wacc,
            "risk_free_rate": a.risk_free_rate,
            "equity_risk_premium": a.equity_risk_premium,
            "beta": a.beta,
        }
        return wacc, re, detail

    #  FCF PROJECTION 


    def project_fcf(self, a: DCFAssumptions, wacc: float) -> pd.DataFrame:
        """
        Build the 20-line IB DCF template.

        Lines:
         1  Revenue
         2  EBIT
         3  Taxes on EBIT
         4  NOPAT  (= EBIT * (1 - T))
         5  D&A
         6  CapEx
         7  NWC
         8  Free Cash Flow
         9  Discount Factor  (= 1 / (1 + WACC)^t)
        10  PV of FCF
        """
        rows = {}
        prev_revenue = a.base_revenue
        prev_nwc = a.base_revenue * a.nwc_change_pct_revenue  # seed

        years = list(range(1, a.projection_years + 1))
        g_rates = list(a.revenue_growth_rates)
        # Pad or trim growth rates to match projection_years
        while len(g_rates) < a.projection_years:
            g_rates.append(g_rates[-1] if g_rates else 0.05)
        g_rates = g_rates[: a.projection_years]

        revenues, ebits, taxes, nopats, das, capexes, dnwcs, fcfs, dfs, pvfcfs = (
            [] for _ in range(10)
        )

        for i, yr in enumerate(years):
            g = g_rates[i]
            rev = prev_revenue * (1 + g)
            ebit = rev * a.ebit_margin
            tax = ebit * a.tax_rate
            nopat = ebit * (1 - a.tax_rate)
            da = rev * a.da_pct_revenue
            capex = rev * a.capex_pct_revenue
            curr_nwc = rev * a.nwc_change_pct_revenue
            dnwc = curr_nwc - prev_nwc

            # FCF = NOPAT + D&A - CapEx - NWC
            fcf = nopat + da - capex - dnwc

            # Discount factor: 1 / (1 + WACC)^t  (DECREASING over time - correctly represents PV factor)
            df_factor = 1.0 / (1.0 + wacc) ** yr
            pv_fcf = fcf * df_factor

            revenues.append(rev)
            ebits.append(ebit)
            taxes.append(tax)
            nopats.append(nopat)
            das.append(da)
            capexes.append(capex)
            dnwcs.append(dnwc)
            fcfs.append(fcf)
            dfs.append(df_factor)
            pvfcfs.append(pv_fcf)

            prev_revenue = rev
            prev_nwc = curr_nwc

        df = pd.DataFrame(
            {
                "Year": years,
                "Revenue ($M)": revenues,
                "EBIT ($M)": ebits,
                "Taxes ($M)": taxes,
                "NOPAT ($M)": nopats,
                "D&A ($M)": das,
                "CapEx ($M)": capexes,
                "NWC ($M)": dnwcs,
                "Free Cash Flow ($M)": fcfs,
                "Discount Factor": dfs,
                "PV of FCF ($M)": pvfcfs,
            }
        )
        df = df.set_index("Year")
        return df


    def compute_terminal_value(
        self, last_fcf: float, wacc: float, terminal_growth: float, years: int
    ) -> Tuple[float, float]:
        """
        TV = FCF_n x (1 + g) / (WACC - g)    [Gordon Growth Model]
        PV(TV) = TV / (1 + WACC)^n
        """
        if wacc <= terminal_growth:
            # Guard: WACC must exceed terminal growth
            wacc = terminal_growth + 0.01
        tv = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv = tv / (1.0 + wacc) ** years
        return tv, pv_tv

    #  FULL DCF 


    def run_dcf(self, assumptions: DCFAssumptions) -> DCFResult:
        """Run complete institutional DCF with all advanced features."""
        
        # Calculate WACC and cost of equity
        wacc = self._calculate_wacc(assumptions)
        cost_of_equity = self._calculate_cost_of_equity(assumptions)
        
        # Defensive: ensure wacc is a single float
        if isinstance(wacc, (tuple, list)):
            wacc = float(wacc[0])
        else:
            wacc = float(wacc)
        
        if isinstance(cost_of_equity, (tuple, list)):
            cost_of_equity = float(cost_of_equity[0])
        else:
            cost_of_equity = float(cost_of_equity)
        
        # Sanity check
        if wacc <= 0 or wacc > 1:
            wacc = 0.10
        if cost_of_equity <= 0 or cost_of_equity > 1:
            cost_of_equity = 0.12
        
        # Project FCFs
        fcfs = []
        revenues = []
        ebits = []
        nopats = []
        discount_factors = []
        pv_fcfs = []  # Add this to match _build_line_items_df signature
        
        revenue = assumptions.base_revenue
        for i, growth_rate in enumerate(assumptions.revenue_growth_rates):
            revenue *= (1 + growth_rate)
            revenues.append(revenue)
            
            ebit = revenue * assumptions.ebit_margin
            ebits.append(ebit)
            
            nopat = ebit * (1 - assumptions.tax_rate)
            nopats.append(nopat)
            
            da = revenue * assumptions.da_pct_revenue
            capex = revenue * assumptions.capex_pct_revenue
            nwc_change = revenue * assumptions.nwc_change_pct_revenue
            
            fcf = nopat + da - capex - nwc_change
            fcfs.append(fcf)
            
            # Calculate discount factor and PV for this period
            year = i + 1
            df = 1 / (1 + wacc) ** year
            discount_factors.append(df)
            pv_fcf = fcf * df
            pv_fcfs.append(pv_fcf)
        
        # Terminal value
        terminal_fcf = fcfs[-1] * (1 + assumptions.terminal_growth_rate)
        
        # Safe division with type checking
        wacc_for_terminal = float(wacc)
        terminal_growth = float(assumptions.terminal_growth_rate)
        
        if wacc_for_terminal <= terminal_growth:
            # Invalid: WACC must be > terminal growth
            wacc_for_terminal = terminal_growth + 0.02  # Add 2% buffer
        
        terminal_value = terminal_fcf / (wacc_for_terminal - terminal_growth)
        
        # PV of terminal value
        pv_terminal_value = terminal_value * discount_factors[-1]
        
        # Enterprise and equity value
        sum_pv_fcf = sum(pv_fcfs)
        enterprise_value = sum_pv_fcf + pv_terminal_value
        net_debt = assumptions.debt_value - assumptions.cash
        equity_value = enterprise_value - net_debt
        fair_value_per_share = equity_value / assumptions.shares_outstanding if assumptions.shares_outstanding > 0 else 0
        
        # Build line items for 20-line DCF - NOW ALL VARIABLES ARE DEFINED
        line_items = self._build_line_items_df(
            revenues, ebits, nopats, fcfs, discount_factors, 
            pv_fcfs, terminal_value, pv_terminal_value
        )
        
        # Scenario analysis
        scenarios = self._run_scenarios(assumptions, enterprise_value, fair_value_per_share)
        scenarios = self._normalize_scenario_probabilities(scenarios)
        scenario_weighted_value = sum(s.fair_value * s.probability for s in scenarios)
        
        # Monte Carlo simulation
        mc_results = self._run_monte_carlo(assumptions, iterations=10000)
        mc_distribution = mc_results['distribution']
        mc_median = np.median(mc_distribution)
        mc_mean = np.mean(mc_distribution)
        mc_std = float(mc_results.get('std', np.std(mc_distribution) if len(mc_distribution) else 0.0))
        mc_upside_prob = sum(1 for v in mc_distribution if v > assumptions.current_price) / len(mc_distribution)
        mc_downside_prob = 1 - mc_upside_prob
        mc_percentiles = {
            'p5': np.percentile(mc_distribution, 5),
            'p25': np.percentile(mc_distribution, 25),
            'p50': mc_median,
            'p75': np.percentile(mc_distribution, 75),
            'p95': np.percentile(mc_distribution, 95),
        }
        
        # Sensitivity analysis
        sensitivity = self._build_sensitivity_table(assumptions)
        
        # Relative valuation
        relative_valuation = self._calculate_relative_valuation(assumptions, fcfs[-1], ebits[-1])
        
        # Catalysts
        catalysts = self._generate_catalysts(assumptions)
        
        # WACC breakdown
        wacc_breakdown = {
            'wacc': wacc,
            'cost_of_equity': cost_of_equity,
            'cost_of_debt_pretax': assumptions.cost_of_debt,
            'cost_of_debt_aftertax': assumptions.cost_of_debt * (1 - assumptions.tax_rate),
            'risk_free_rate': assumptions.risk_free_rate,
            'equity_risk_premium': assumptions.equity_risk_premium,
            'beta': assumptions.beta,
            'weight_equity': assumptions.equity_value_market / (assumptions.equity_value_market + assumptions.debt_value),
            'weight_debt': assumptions.debt_value / (assumptions.equity_value_market + assumptions.debt_value),
        }
        
        # Trade signal
        trade_signal = self._generate_trade_signal(
            fair_value_per_share, assumptions.current_price, 
            mc_upside_prob, scenarios
        )
        
        # NEW FEATURES: Expected Returns, Reverse DCF, Historical Anchoring
        expected_returns_data = self._calculate_expected_returns(
            assumptions, scenarios, assumptions.current_price, assumptions.projection_years
        )
        
        # FIX: Pass fair_value_per_share (not base_fv) to _calculate_reverse_dcf
        reverse_dcf_data = self._calculate_reverse_dcf(
            assumptions, enterprise_value
        )
        
        historical_data = self._fetch_historical_financials(assumptions.ticker)
        forecast_comparison = self._compare_forecast_to_historical(assumptions, historical_data)
        
        # Build complete result
        result = DCFResult(
            ticker=assumptions.ticker,
            assumptions=assumptions,
            fair_value_per_share=fair_value_per_share,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            wacc=wacc,
            cost_of_equity=cost_of_equity,
            net_debt=net_debt,
            sum_pv_fcf=sum_pv_fcf,
            pv_terminal_value=pv_terminal_value,
            line_items=line_items,
            scenarios=scenarios,
            scenario_weighted_value=scenario_weighted_value,
            mc_distribution=mc_distribution,
            mc_median=mc_median,
            mc_mean=mc_mean,
            mc_upside_prob=mc_upside_prob,
            mc_downside_prob=mc_downside_prob,
            mc_percentiles=mc_percentiles,
            sensitivity=sensitivity,
            relative_valuation=relative_valuation,
            catalysts=catalysts,
            wacc_breakdown=wacc_breakdown,
            trade_signal=trade_signal,
            mc_std=mc_std,
            
            # NEW: Expected investor returns
            expected_returns=expected_returns_data.get('irrs', {}),
            probability_weighted_irr=expected_returns_data.get('weighted_irr', 0.0),
            investment_recommendation=expected_returns_data.get('recommendation', 'Neutral'),
            recommendation_color=expected_returns_data.get('recommendation_color', '#aaaaaa'),
            
            # NEW: Reverse DCF
            market_implied_growth=reverse_dcf_data.get('implied_growth', 0.0),
            market_implied_ebit_margin=reverse_dcf_data.get('implied_ebit', 0.0),
            market_implied_fcf_growth=reverse_dcf_data.get('implied_fcf_growth', 0.0),
            market_expectation_label=reverse_dcf_data.get('expectation_label', 'Unknown'),
            market_expectation_color=reverse_dcf_data.get('expectation_color', '#aaaaaa'),
            
            # NEW: Historical anchoring
            historical_data=historical_data.get('data', pd.DataFrame()),
            historical_stats=historical_data.get('stats', {}),
            forecast_vs_historical=forecast_comparison,
        )

        # --- Additive institutional layers ---
        result.trade_signal.ticker = assumptions.ticker if result.trade_signal else assumptions.ticker
        result.stress_tests = self._run_stress_tests(assumptions)
        result.regime_adjusted_values = self._compute_regime_adjusted_values(result)
        result.diagnostic_flags = self._compute_diagnostic_flags(result, assumptions)
        result.visualization_payload = self._build_visualization_payload(result)
        result.probabilistic_forecast = self._build_probabilistic_forecast(assumptions, mc_distribution)

        # --- Institutional upgrade layers (all defensive — never break the run) ---
        try:
            result.terminal_diagnostics = self._terminal_diagnostics(result, assumptions)
        except Exception:
            result.terminal_diagnostics = {}
        try:
            result.key_assumption_attribution = self._key_assumption_attribution(assumptions)
        except Exception:
            result.key_assumption_attribution = {}
        try:
            result.consensus_layer = self._consensus_layer(assumptions, result)
        except Exception:
            result.consensus_layer = {}
        try:
            result.dynamic_scenario_notes = self._dynamic_scenario_notes(assumptions, result)
        except Exception:
            result.dynamic_scenario_notes = ""

        return result
    


    def _calculate_relative_valuation(self, assumptions: DCFAssumptions, terminal_fcf: float, terminal_ebit: float) -> Dict[str, Any]:
        """
        Calculate implied share prices based on peer trading multiples.
        """
        # We need net debt to convert EV to Equity Value
        net_debt = assumptions.debt_value - assumptions.cash
        shares = max(assumptions.shares_outstanding, 1.0)
        
        # 1. PE Multiple (requires Net Income approximation -> NOPAT - Interest)
        interest_exp = assumptions.debt_value * assumptions.cost_of_debt
        net_income = max(0, (terminal_ebit - interest_exp) * (1 - assumptions.tax_rate))
        implied_equity_pe = net_income * assumptions.peer_pe
        price_pe = max(0.0, implied_equity_pe / shares)
        
        # 2. EV/EBITDA Multiple
        # Approximate EBITDA = EBIT + D&A
        terminal_da = assumptions.base_revenue * ((1 + assumptions.revenue_growth_rates[0]) ** assumptions.projection_years) * assumptions.da_pct_revenue
        ebitda = terminal_ebit + terminal_da
        implied_ev_ebitda = ebitda * assumptions.peer_ev_ebitda
        price_ev_ebitda = max(0.0, (implied_ev_ebitda - net_debt) / shares)
        
        # 3. EV/FCF Multiple
        implied_ev_fcf = terminal_fcf * assumptions.peer_ev_fcf
        price_ev_fcf = max(0.0, (implied_ev_fcf - net_debt) / shares)
        
        # 4. PEG Ratio
        eps = net_income / shares if shares > 0 else 0
        eps_growth = assumptions.revenue_growth_rates[0] * 100 # Approx growth whole numbers
        price_peg = max(0.0, assumptions.peg_ratio * eps_growth * eps)
        
        return {
            'pe_implied_fv': float(price_pe),
            'ev_ebitda_implied_fv': float(price_ev_ebitda),
            'ev_fcf_implied_fv': float(price_ev_fcf),
            'peg_implied_fv': float(price_peg),
            'peer_pe': float(assumptions.peer_pe),
            'peer_ev_ebitda': float(assumptions.peer_ev_ebitda),
            'peer_ev_fcf': float(assumptions.peer_ev_fcf),
            'peg_ratio': float(assumptions.peg_ratio),
            'ebitda_M': float(ebitda),
            'last_fcf_M': float(terminal_fcf),
            'eps_proxy': float(eps),
            'blended_relative_value': float((price_pe + price_ev_ebitda + price_ev_fcf + price_peg) / 4)
        }


    def _run_scenarios(self, assumptions: DCFAssumptions, base_ev: float, base_fv: float) -> List[ScenarioResult]:
        """Run bear/base/bull scenario analysis with asymmetric skewed weighting."""
        scenarios = []
        
        # Bear case: lower growth, wider spread
        bear_assumptions = self._create_scenario_assumptions(
            assumptions, 
            growth_multiplier=0.6,
            margin_delta=-0.03,
            wacc_delta=0.015
        )
        bear_ev, bear_fv = self._quick_dcf(bear_assumptions)
        scenarios.append(ScenarioResult(
            label='Bear',
            probability=0.20,  # Asymmetric Skewed: 20%
            fair_value=bear_fv,
            upside=(bear_fv - assumptions.current_price) / assumptions.current_price if assumptions.current_price > 0 else 0,
            revenue_growth_avg=bear_assumptions.revenue_growth_rates[0],
            ebit_margin=bear_assumptions.ebit_margin,
            wacc=self._calculate_wacc(bear_assumptions),
            terminal_growth=bear_assumptions.terminal_growth_rate,
        ))
        
        # Base case
        scenarios.append(ScenarioResult(
            label='Base',
            probability=0.70,  # Asymmetric Skewed: 70%
            fair_value=base_fv,
            upside=(base_fv - assumptions.current_price) / assumptions.current_price if assumptions.current_price > 0 else 0,
            revenue_growth_avg=assumptions.revenue_growth_rates[0],
            ebit_margin=assumptions.ebit_margin,
            wacc=self._calculate_wacc(assumptions),
            terminal_growth=assumptions.terminal_growth_rate,
        ))
        
        # Bull case: higher growth, tighter spread
        bull_assumptions = self._create_scenario_assumptions(
            assumptions,
            growth_multiplier=1.4,
            margin_delta=0.03,
            wacc_delta=-0.01
        )
        bull_ev, bull_fv = self._quick_dcf(bull_assumptions)
        scenarios.append(ScenarioResult(
            label='Bull',
            probability=0.10,  # Asymmetric Skewed: 10%
            fair_value=bull_fv,
            upside=(bull_fv - assumptions.current_price) / assumptions.current_price if assumptions.current_price > 0 else 0,
            revenue_growth_avg=bull_assumptions.revenue_growth_rates[0],
            ebit_margin=bull_assumptions.ebit_margin,
            wacc=self._calculate_wacc(bull_assumptions),
            terminal_growth=bull_assumptions.terminal_growth_rate,
        ))
        
        return scenarios
    

    def _create_scenario_assumptions(self, base: DCFAssumptions, growth_multiplier: float, 
                                    margin_delta: float, wacc_delta: float) -> DCFAssumptions:
        """Create modified assumptions for scenario analysis."""
        return DCFAssumptions(
            ticker=base.ticker,
            base_revenue=base.base_revenue,
            revenue_growth_rates=[g * growth_multiplier for g in base.revenue_growth_rates],
            ebit_margin=max(0.05, min(0.60, base.ebit_margin + margin_delta)),
            tax_rate=base.tax_rate,
            da_pct_revenue=base.da_pct_revenue,
            capex_pct_revenue=base.capex_pct_revenue,
            nwc_change_pct_revenue=base.nwc_change_pct_revenue,
            equity_value_market=base.equity_value_market,
            debt_value=base.debt_value,
            cost_of_debt=base.cost_of_debt + wacc_delta,
            risk_free_rate=base.risk_free_rate + wacc_delta,
            equity_risk_premium=base.equity_risk_premium,
            beta=base.beta,
            terminal_growth_rate=base.terminal_growth_rate,
            cash=base.cash,
            shares_outstanding=base.shares_outstanding,
            current_price=base.current_price,
            peer_pe=base.peer_pe,
            peer_ev_ebitda=base.peer_ev_ebitda,
            peer_ev_fcf=base.peer_ev_fcf,
            peg_ratio=base.peg_ratio,
            projection_years=base.projection_years,
        )
    

    def _quick_dcf(self, assumptions: DCFAssumptions) -> Tuple[float, float]:
        """Quick DCF calculation for scenarios/monte carlo."""
        try:
            wacc_result = self._calculate_wacc(assumptions)
            
            # Ensure wacc is a single float, not a tuple
            if isinstance(wacc_result, (tuple, list)):
                wacc = float(wacc_result[0])
            else:
                wacc = float(wacc_result)
            
            # Ensure wacc is valid
            if wacc <= 0 or wacc > 1 or not isinstance(wacc, float):
                wacc = 0.10
            
            revenue = assumptions.base_revenue
            fcfs = []
            for growth_rate in assumptions.revenue_growth_rates:
                revenue *= (1 + growth_rate)
                ebit = revenue * assumptions.ebit_margin
                nopat = ebit * (1 - assumptions.tax_rate)
                da = revenue * assumptions.da_pct_revenue
                capex = revenue * assumptions.capex_pct_revenue
                nwc_change = revenue * assumptions.nwc_change_pct_revenue
                fcf = nopat + da - capex - nwc_change
                fcfs.append(fcf)
            
            # Terminal value with safety checks
            terminal_fcf = fcfs[-1] * (1 + assumptions.terminal_growth_rate)
            terminal_growth = float(assumptions.terminal_growth_rate)
            
            # Ensure wacc > terminal growth
            if wacc <= terminal_growth:
                wacc = terminal_growth + 0.02
            
            terminal_value = terminal_fcf / (wacc - terminal_growth)
            
            # PV
            pv_fcfs = sum(fcf / (1 + wacc) ** i for i, fcf in enumerate(fcfs, 1))
            pv_terminal = terminal_value / (1 + wacc) ** len(fcfs)
            
            enterprise_value = pv_fcfs + pv_terminal
            net_debt = assumptions.debt_value - assumptions.cash
            equity_value = enterprise_value - net_debt
            fair_value_per_share = equity_value / assumptions.shares_outstanding if assumptions.shares_outstanding > 0 else 0
            
            return float(enterprise_value), float(fair_value_per_share)
        except Exception as e:
            print(f"Quick DCF error: {e}")
            return 0.0, 0.0
    

    def _calculate_expected_returns(self, assumptions: DCFAssumptions, scenarios: List,
                                   current_price: float, projection_years: int) -> Dict[str, Any]:
        """Calculate expected investor returns (IRR) for each scenario."""
        if not scenarios or projection_years <= 0 or current_price <= 0:
            return {
                'irrs': {'Bear': 0.0, 'Base': 0.0, 'Bull': 0.0},
                'weighted_irr': 0.0,
                'recommendation': 'Insufficient Data',
                'recommendation_color': '#aaaaaa'
            }
        
        irrs = {}
        for scenario in scenarios:
            future_price = scenario.fair_value
            if current_price > 0 and projection_years > 0 and future_price > 0:
                irr = (future_price / current_price) ** (1 / projection_years) - 1
            else:
                irr = 0.0
            irrs[scenario.label] = irr
        
        # Probability-weighted IRR
        weighted_irr = sum(scenario.probability * irrs.get(scenario.label, 0.0) for scenario in scenarios)
        
        # Investment recommendation
        if weighted_irr > 0.12:
            recommendation = "Attractive Investment"
            rec_color = "#00ff88"
        elif weighted_irr >= 0.06:
            recommendation = "Neutral / Hold"
            rec_color = "#e0c97f"
        else:
            recommendation = "Overvalued / Avoid"
            rec_color = "#ff4444"
        
        return {
            'irrs': irrs,
            'weighted_irr': weighted_irr,
            'recommendation': recommendation,
            'recommendation_color': rec_color
        }
    

    def _calculate_reverse_dcf(self, assumptions: DCFAssumptions, base_ev: float) -> Dict[str, Any]:
        """FEATURE 1: Calculate market-implied expectations (Reverse DCF)."""
        
        # Target enterprise value from current market price
        target_equity_value = assumptions.current_price * assumptions.shares_outstanding
        target_ev = target_equity_value + assumptions.debt_value - assumptions.cash
        
        # Solve for implied revenue growth
        try:
            from scipy.optimize import fsolve
            
            def dcf_equation(growth_rate):
                test_assumptions = self._create_scenario_assumptions(
                    assumptions, 
                    growth_multiplier=growth_rate / assumptions.revenue_growth_rates[0] if assumptions.revenue_growth_rates[0] > 0 else 1.0,
                    margin_delta=0,
                    wacc_delta=0
                )
                test_ev, _ = self._quick_dcf(test_assumptions)
                return test_ev - target_ev
            
            initial_guess = assumptions.revenue_growth_rates[0]
            implied_growth = fsolve(dcf_equation, initial_guess)[0]
            implied_growth = max(-0.20, min(0.50, implied_growth))
        except Exception as e:
            implied_growth = assumptions.revenue_growth_rates[0]
        
        # Solve for implied EBIT margin
        try:
            def ebit_equation(ebit_margin):
                test_assumptions = self._create_scenario_assumptions(
                    assumptions,
                    growth_multiplier=1.0,
                    margin_delta=ebit_margin - assumptions.ebit_margin,
                    wacc_delta=0
                )
                test_ev, _ = self._quick_dcf(test_assumptions)
                return test_ev - target_ev

            initial_margin_guess = assumptions.ebit_margin
            implied_ebit = fsolve(ebit_equation, initial_margin_guess)[0]
            implied_ebit = max(0.01, min(0.80, implied_ebit))
        except Exception as e:
            implied_ebit = assumptions.ebit_margin

        # Solve for implied FCF growth
        try:
            last_fcf = assumptions.base_revenue * assumptions.ebit_margin * (1 - assumptions.tax_rate)
            if last_fcf > 0 and target_ev > 0:
                wacc_val = self._calculate_wacc(assumptions)
                if isinstance(wacc_val, (tuple, list)):
                    wacc_val = float(wacc_val[0])
                else:
                    wacc_val = float(wacc_val)
                implied_fcf_growth = wacc_val - (last_fcf / target_ev) if target_ev > 0 else 0.03
                implied_fcf_growth = max(-0.10, min(0.30, implied_fcf_growth))
            else:
                implied_fcf_growth = 0.03
        except Exception:
            implied_fcf_growth = 0.03

        # Determine market expectation label
        growth_diff = implied_growth - assumptions.revenue_growth_rates[0]
        if growth_diff > 0.03:
            expectation_label = "Market expects HIGHER growth than model"
            expectation_color = "#00ff88"
        elif growth_diff < -0.03:
            expectation_label = "Market expects LOWER growth than model"
            expectation_color = "#ff4444"
        else:
            expectation_label = "Market expectations ALIGNED with model"
            expectation_color = "#e0c97f"

        return {
            'implied_growth': implied_growth,
            'implied_ebit': implied_ebit,
            'implied_fcf_growth': implied_fcf_growth,
            'expectation_label': expectation_label,
            'expectation_color': expectation_color,
        }


    def _fetch_historical_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch 10-year historical financials using REAL statement data only.

        Data-integrity contract: revenue / EBIT / FCF figures are never
        synthesized from price data or industry averages. When real statements
        are unavailable, an explicitly-labeled UNAVAILABLE structure is
        returned so the DCF's historical anchoring is never grounded in
        fabricated numbers.
        """
        try:
            # Primary: robust multi-source engine (real statement data only)
            engine = get_historical_engine()
            result = engine.get_historical_financials(ticker, years=10)

            if result and not result['data'].empty:
                print(f"[OK] Historical data for {ticker} fetched from: {result.get('source', 'Unknown')}")
                return result

            # Fallback: direct yfinance real statements (no synthetic margins)
            print(f"Attempting yfinance fallback for {ticker}...")
            import yfinance as yf
            stock = yf.Ticker(ticker)

            financials = stock.financials
            if financials is not None and not financials.empty:
                rows = []
                revenue_series = None
                for label in ('Total Revenue', 'Revenue'):
                    if label in financials.index:
                        revenue_series = financials.loc[label]
                        break

                ebit_series = None
                for label in ('EBIT', 'Operating Income'):
                    if label in financials.index:
                        ebit_series = financials.loc[label]
                        break

                if revenue_series is not None:
                    for date_col in revenue_series.index[:10]:
                        try:
                            year = str(date_col.year) if hasattr(date_col, 'year') else str(date_col)[:4]
                            revenue = float(revenue_series[date_col]) / 1e6
                            if revenue <= 0:
                                continue
                            ebit_margin = None
                            if ebit_series is not None and date_col in ebit_series.index:
                                ebit_val = float(ebit_series[date_col]) / 1e6
                                ebit_margin = (ebit_val / revenue * 100) if revenue > 0 else None
                            rows.append({
                                'Year': year,
                                'Revenue': revenue,
                                'EBIT Margin %': ebit_margin,
                            })
                        except Exception:
                            continue

                if rows and len(rows) >= 2:
                    rows.sort(key=lambda x: x['Year'], reverse=True)
                    df = pd.DataFrame(rows)
                    df_sorted = df.sort_values('Year')
                    df_sorted['Revenue Growth %'] = df_sorted['Revenue'].pct_change() * 100
                    df = df_sorted.sort_values('Year', ascending=False)
                    # Only real margins are included; no FCF fabrication.
                    stats = {
                        'avg_revenue_growth': df['Revenue Growth %'].mean()
                        if df['Revenue Growth %'].notna().any() else 0.0,
                        'avg_ebit_margin': df['EBIT Margin %'].mean()
                        if df['EBIT Margin %'].notna().any() else 0.0,
                        'avg_fcf_margin': 0.0,
                        'std_revenue_growth': df['Revenue Growth %'].std()
                        if df['Revenue Growth %'].notna().any() else 0.0,
                        'has_real_fcf': False,
                    }
                    print(f"[OK] Historical data for {ticker} constructed from yfinance fundamentals")
                    return {'data': df, 'stats': stats, 'source': 'yfinance fundamentals',
                            'data_quality': 'REAL_PARTIAL'}
        except Exception as e:
            print(f"Historical fetch error for {ticker}: {e}")

        # No real statements available — explicit unavailable state. Synthetic
        # revenue/margin figures are NEVER generated for real data paths.
        print(f"[WARN] No real financial statements available for {ticker}; marking historical data unavailable.")
        return {
            'data': pd.DataFrame(columns=['Year', 'Revenue', 'Revenue Growth %', 'EBIT Margin %', 'FCF Margin %']),
            'stats': {'avg_revenue_growth': 0.0, 'avg_ebit_margin': 0.0, 'avg_fcf_margin': 0.0,
                      'std_revenue_growth': 0.0, 'has_real_fcf': False},
            'source': 'None (insufficient data)',
            'data_quality': 'UNAVAILABLE',
        }

    

    def _generate_catalysts(self, assumptions: DCFAssumptions) -> List[CatalystEvent]:
        """Generate comprehensive catalyst events based on ticker and market analysis."""
        catalysts = []
        today = datetime.now()
        
        # Try to get real earnings date and events
        try:
            import yfinance as yf
            stock = yf.Ticker(assumptions.ticker)
            calendar = stock.calendar
            
            # Earnings date
            if calendar is not None and 'Earnings Date' in calendar:
                earnings_dates = calendar['Earnings Date']
                if earnings_dates is not None and len(earnings_dates) > 0:
                    next_earnings = earnings_dates[0]
                    if pd.notna(next_earnings):
                        catalysts.append(CatalystEvent(
                            name="Quarterly Earnings Release",
                            date=pd.to_datetime(next_earnings).strftime("%Y-%m-%d"),
                            impact="High",
                            direction="Positive" if assumptions.revenue_growth_rates[0] > 0.08 else "Neutral",
                            description=f"Q{((today.month-1)//3)+1} earnings expected. Revenue growth of {assumptions.revenue_growth_rates[0]:.1%} and EBIT margin of {assumptions.ebit_margin:.1%} will drive sentiment."
                        ))
        except Exception:
            pass
        
        # Add default earnings if not found
        if not any(c.name == "Quarterly Earnings Release" for c in catalysts):
            catalysts.append(CatalystEvent(
                name="Quarterly Earnings Release",
                date=(today + timedelta(days=30)).strftime("%Y-%m-%d"),
                impact="High",
                direction="Positive" if assumptions.revenue_growth_rates[0] > 0.08 else "Neutral",
                description=f"Next quarterly report. Expected revenue growth {assumptions.revenue_growth_rates[0]:.1%} and EBIT margin {assumptions.ebit_margin:.1%}."
            ))
        
        # Sector-specific catalysts
        try:
            stock = yf.Ticker(assumptions.ticker)
            info = stock.info
            sector = info.get('sector', '')
            
            if 'Technology' in sector:
                catalysts.append(CatalystEvent(
                    name="Product Innovation Cycle",
                    date=(today + timedelta(days=90)).strftime("%Y-%m-%d"),
                    impact="High",
                    direction="Positive",
                    description="Technology sector: New product launches and AI integration could drive revenue acceleration and margin expansion."
                ))
                catalysts.append(CatalystEvent(
                    name="Cloud/AI Revenue Recognition",
                    date=(today + timedelta(days=180)).strftime("%Y-%m-%d"),
                    impact="Medium",
                    direction="Positive",
                    description="Accelerating cloud/AI adoption should drive higher-margin revenue mix and operating leverage."
                ))
            
            elif 'Financial' in sector:
                catalysts.append(CatalystEvent(
                    name="Federal Reserve Policy Decision",
                    date=(today + timedelta(days=45)).strftime("%Y-%m-%d"),
                    impact="High",
                    direction="Neutral",
                    description="Interest rate changes directly impact net interest margins and loan growth for financial sector."
                ))
                catalysts.append(CatalystEvent(
                    name="Credit Quality Review",
                    date=(today + timedelta(days=90)).strftime("%Y-%m-%d"),
                    impact="Medium",
                    direction="Neutral",
                    description="Loan loss provisions and credit quality metrics will determine earnings sustainability."
                ))
            
            elif 'Energy' in sector or 'Basic Materials' in sector:
                catalysts.append(CatalystEvent(
                    name="Commodity Price Movement",
                    date=(today + timedelta(days=60)).strftime("%Y-%m-%d"),
                    impact="High",
                    direction="Positive" if assumptions.ebit_margin > 0.15 else "Negative",
                    description="Oil/commodity prices directly impact revenue and margins. Current margin structure suggests sensitivity to price changes."
                ))
                catalysts.append(CatalystEvent(
                    name="Global Demand Outlook",
                    date=(today + timedelta(days=120)).strftime("%Y-%m-%d"),
                    impact="Medium",
                    direction="Neutral",
                    description="China economic growth and global industrial production will drive demand fundamentals."
                ))
            
            elif 'Healthcare' in sector:
                catalysts.append(CatalystEvent(
                    name="Drug Pipeline/FDA Approvals",
                    date=(today + timedelta(days=120)).strftime("%Y-%m-%d"),
                    impact="High",
                    direction="Positive",
                    description="New drug approvals or clinical trial results could significantly expand addressable market."
                ))
                catalysts.append(CatalystEvent(
                    name="Medicare Reimbursement Rates",
                    date=(today + timedelta(days=180)).strftime("%Y-%m-%d"),
                    impact="Medium",
                    direction="Neutral",
                    description="Government reimbursement policy changes affect revenue per procedure/prescription."
                ))
            
            elif 'Consumer' in sector:
                catalysts.append(CatalystEvent(
                    name="Consumer Spending Trends",
                    date=(today + timedelta(days=60)).strftime("%Y-%m-%d"),
                    impact="High",
                    direction="Positive" if assumptions.revenue_growth_rates[0] > 0.05 else "Negative",
                    description="Retail sales data and consumer confidence will drive near-term comparable store sales."
                ))
                catalysts.append(CatalystEvent(
                    name="Holiday Season Performance",
                    date=(today + timedelta(days=150)).strftime("%Y-%m-%d"),
                    impact="High",
                    direction="Positive",
                    description="Q4 holiday sales typically represent 30-40% of annual revenue for consumer companies."
                ))
            
        except Exception:
            pass
        
        # Add universal macro catalysts
        catalysts.append(CatalystEvent(
            name="Federal Reserve FOMC Meeting",
            date=(today + timedelta(days=45)).strftime("%Y-%m-%d"),
            impact="Medium",
            direction="Neutral",
            description=f"Interest rate changes affect WACC (currently {assumptions.risk_free_rate:.2%}). Rate cuts would reduce discount rate and increase valuation."
        ))
        
        # Market structure catalysts
        if assumptions.shares_outstanding > 0:
            market_cap = assumptions.current_price * assumptions.shares_outstanding
            if market_cap > 500:  # >$500M
                catalysts.append(CatalystEvent(
                    name="Index Rebalancing/Inclusion",
                    date=(today + timedelta(days=90)).strftime("%Y-%m-%d"),
                    impact="Medium",
                    direction="Positive",
                    description="Potential index inclusion (S&P 500/Russell) could drive $500M+ of passive inflows."
                ))
        
        # Valuation-based catalyst
        wacc = self._calculate_wacc(assumptions)
        if wacc > 0.12:
            catalysts.append(CatalystEvent(
                name="Capital Structure Optimization",
                date=(today + timedelta(days=120)).strftime("%Y-%m-%d"),
                impact="Medium",
                direction="Positive",
                description=f"High WACC of {wacc:.2%} suggests opportunity to refinance debt or optimize capital structure, potentially adding 5-10% to valuation."
            ))
        
        # Growth inflection catalyst
        if len(assumptions.revenue_growth_rates) > 1:
            growth_acceleration = assumptions.revenue_growth_rates[0] - 0.05  # vs market avg
            if growth_acceleration > 0.03:
                catalysts.append(CatalystEvent(
                    name="Growth Inflection Recognition",
                    date=(today + timedelta(days=180)).strftime("%Y-%m-%d"),
                    impact="High",
                    direction="Positive",
                    description=f"Above-market growth rate of {assumptions.revenue_growth_rates[0]:.1%} vs market ~5% should drive multiple expansion over 6-12 months."
                ))
        
        # Sort by date
        catalysts.sort(key=lambda x: x.date)
        
        return catalysts[:8]  # Return top 8 most relevant
    


    def generate_excel(self, model_data: Dict[str, Any]) -> bytes:
        """
        Generate a comprehensive Excel model from a DCFResult object.
        model_data should contain {'full_result': DCFResult}
        """
        import io
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        result = model_data.get('full_result')
        if not result:
            return b""
            
        wb = Workbook()
        ws = wb.active
        ws.title = "DCF Valuation"
        
        # Styles
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="161b22", end_color="161b22", fill_type="solid")
        title_font = Font(bold=True, size=14)
        
        # Write Title
        ws.cell(row=1, column=1, value=f"{result.ticker} - Institutional DCF Model").font = title_font
        
        # Write KPIs
        ws.cell(row=3, column=1, value="Enterprise Value ($M)")
        ws.cell(row=3, column=2, value=result.enterprise_value)
        ws.cell(row=4, column=1, value="Equity Value ($M)")
        ws.cell(row=4, column=2, value=result.equity_value)
        ws.cell(row=5, column=1, value="Fair Value Per Share")
        ws.cell(row=5, column=2, value=result.fair_value_per_share)
        
        # Write Line Items
        start_row = 8
        if not result.line_items.empty:
            df = result.line_items
            
            # Headers
            ws.cell(row=start_row, column=1, value="Metric").font = header_font
            ws.cell(row=start_row, column=1).fill = header_fill
            for col_idx, col_name in enumerate(df.columns, 2):
                cell = ws.cell(row=start_row, column=col_idx, value=str(col_name))
                cell.font = header_font
                cell.fill = header_fill
                
            # Data
            for row_idx, (idx_name, row_data) in enumerate(df.iterrows(), start_row + 1):
                ws.cell(row=row_idx, column=1, value=str(idx_name))
                for col_idx, val in enumerate(row_data, 2):
                    ws.cell(row=row_idx, column=col_idx, value=val)
                    
        # Write Assumptions
        start_row += len(result.line_items) + 4 if not result.line_items.empty else 4
        ws.cell(row=start_row, column=1, value="Key Assumptions").font = title_font
        
        assumptions = result.assumptions
        assump_data = [
            ("WACC", result.wacc),
            ("Cost of Equity", result.cost_of_equity),
            ("Terminal Growth", assumptions.terminal_growth_rate),
            ("Risk Free Rate", assumptions.risk_free_rate),
            ("Equity Risk Premium", assumptions.equity_risk_premium),
            ("Beta", assumptions.beta),
            ("Tax Rate", assumptions.tax_rate)
        ]
        
        for idx, (label, val) in enumerate(assump_data, start_row + 2):
            ws.cell(row=idx, column=1, value=label)
            ws.cell(row=idx, column=2, value=val)
            
        # Save to bytes
        output = io.BytesIO()
        wb.save(output)
        return output.getvalue()


    def _calculate_wacc(self, assumptions: DCFAssumptions) -> float:
        """Calculate Weighted Average Cost of Capital - returns single float."""
        try:
            cost_of_equity = assumptions.risk_free_rate + assumptions.beta * assumptions.equity_risk_premium
            cost_of_debt_after_tax = assumptions.cost_of_debt * (1 - assumptions.tax_rate)
            total_capital = assumptions.equity_value_market + assumptions.debt_value
            
            if total_capital == 0 or total_capital is None:
                return float(cost_of_equity)
            
            weight_equity = assumptions.equity_value_market / total_capital
            weight_debt = assumptions.debt_value / total_capital
            wacc_value = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt_after_tax)
            
            if isinstance(wacc_value, (tuple, list)):
                wacc_value = wacc_value[0]
            
            return float(max(0.01, min(0.30, wacc_value)))
            
        except Exception as e:
            print(f"WACC calculation error: {e}")
            return 0.10
    

    def _calculate_cost_of_equity(self, assumptions: DCFAssumptions) -> float:
        """Calculate cost of equity using CAPM - returns single float."""
        try:
            cost_of_equity = assumptions.risk_free_rate + assumptions.beta * assumptions.equity_risk_premium
            if isinstance(cost_of_equity, (tuple, list)):
                cost_of_equity = cost_of_equity[0]
            return float(max(0.01, min(0.30, cost_of_equity)))
        except Exception as e:
            print(f"Cost of equity calculation error: {e}")
            return 0.12
    

    def _build_line_items_df(self, revenues: List[float], ebits: List[float], 
                           nopats: List[float], fcfs: List[float],
                           discount_factors: List[float], pv_fcfs: List[float],
                           terminal_value: float, pv_terminal_value: float) -> pd.DataFrame:
        """Build the 20-line DCF table."""
        try:
            years = [f"Year {i+1}" for i in range(len(revenues))]
            rows = []
            rows.append(["Revenue"] + revenues)
            rows.append(["EBIT"] + ebits)
            rows.append(["NOPAT"] + nopats)
            rows.append(["FCF"] + fcfs)
            rows.append(["Discount Factor"] + discount_factors)
            rows.append(["PV(FCF)"] + pv_fcfs)
            rows.append(["Terminal Value"] + [0] * (len(revenues) - 1) + [terminal_value])
            rows.append(["PV(Terminal)"] + [0] * (len(revenues) - 1) + [pv_terminal_value])
            df = pd.DataFrame(rows, columns=["Line Item"] + years)
            return df
        except Exception as e:
            return pd.DataFrame({
                "Line Item": ["Revenue", "EBIT", "NOPAT", "FCF"],
                "Year 1": [0, 0, 0, 0]
            })
    

    def _run_monte_carlo(self, assumptions: DCFAssumptions, iterations: int = 10000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for valuation distribution."""
        try:
            results = []
            for _ in range(iterations):
                rev_growth = np.random.normal(assumptions.revenue_growth_rates[0], assumptions.revenue_growth_rates[0] * 0.3)
                ebit_margin = np.random.normal(assumptions.ebit_margin, assumptions.ebit_margin * 0.2)
                wacc_sim = np.random.normal(self._calculate_wacc(assumptions), 0.02)
                terminal_growth = np.random.normal(assumptions.terminal_growth_rate, 0.01)
                
                rev_growth = max(-0.10, min(0.50, rev_growth))
                ebit_margin = max(0.05, min(0.60, ebit_margin))
                wacc_sim = max(0.03, min(0.25, wacc_sim))
                terminal_growth = max(0.005, min(0.05, terminal_growth))
                
                revenue = assumptions.base_revenue
                fcfs_sim = []
                for year in range(assumptions.projection_years):
                    revenue *= (1 + rev_growth)
                    ebit = revenue * ebit_margin
                    nopat = ebit * (1 - assumptions.tax_rate)
                    da = revenue * assumptions.da_pct_revenue
                    capex = revenue * assumptions.capex_pct_revenue
                    nwc = revenue * assumptions.nwc_change_pct_revenue
                    fcf = nopat + da - capex - nwc
                    fcfs_sim.append(fcf)
                
                pv_fcfs = sum(fcf / (1 + wacc_sim) ** i for i, fcf in enumerate(fcfs_sim, 1))
                terminal_fcf = fcfs_sim[-1] * (1 + terminal_growth)
                terminal_val = terminal_fcf / (wacc_sim - terminal_growth) if wacc_sim > terminal_growth else fcfs_sim[-1] * 20
                pv_terminal = terminal_val / (1 + wacc_sim) ** assumptions.projection_years
                
                ev = pv_fcfs + pv_terminal
                equity_val = ev - (assumptions.debt_value - assumptions.cash)
                fv_per_share = equity_val / assumptions.shares_outstanding if assumptions.shares_outstanding > 0 else 0
                results.append(fv_per_share)
            
            return {
                'distribution': results,
                'mean': np.mean(results),
                'median': np.median(results),
                'std': np.std(results)
            }
        except Exception as e:
            return {
                'distribution': [assumptions.current_price] * 100,
                'mean': assumptions.current_price,
                'median': assumptions.current_price,
                'std': 0
            }
    


    def _generate_trade_signal(self, fair_value: float, current_price: float, upside_prob: float, scenarios: List[ScenarioResult]) -> TradeSignal:
        upside_pct = (fair_value / current_price) - 1 if current_price > 0 else 0
        
        # Final Verdict logic based on required 25% Margin of Safety (MoS)
        signal = "HOLD"
        # 25% MoS implies price is at least 25% below fair value (upside >= 33.3%)
        if upside_pct >= 0.50:
            signal = "STRONG BUY"
        elif upside_pct >= 0.33:
            signal = "BUY"
        elif upside_pct <= -0.15:
            signal = "STRONG SELL"
        elif upside_pct <= -0.10:
            signal = "SELL"
            
        risk = "MEDIUM"
        if scenarios:
            bull = next((s for s in scenarios if s.label == 'Bull'), None)
            bear = next((s for s in scenarios if s.label == 'Bear'), None)
            if bull and bear:
                spread = (bull.fair_value - bear.fair_value) / current_price
                if spread > 0.5: risk = "HIGH"
                elif spread < 0.2: risk = "LOW"
                
        # Professional rationale for hardware-reliant companies
        rationale = (
            "A 70% weight on the Base case reflects the predictable operational rhythms and execution requirements typical of hardware-reliant enterprises, "
            "where supply chain stability and production scaling follow linear paths. The Bull case is capped at 10% to account for the 'Fat-Tail Risk' "
            "of disruptive market shifts, while the 20% Bear case maintains conservative sensitivity to macroeconomic headwinds. "
            "Using the Median from a 10,000-iteration Monte Carlo simulation further supports this conservative Fair Value by filtering out extreme, "
            "low-probability outliers that can skew an arithmetic mean."
        )
                
        return TradeSignal(
            ticker="",
            fair_value=float(fair_value),
            market_price=float(current_price),
            upside_pct=float(upside_pct),
            signal=signal,
            confidence_pct=float(upside_prob),
            risk_level=risk,
            risk_adjusted_return=float(upside_pct * upside_prob),
            position_size_pct=min(0.05, max(0.0, upside_pct * upside_prob * 0.1)),
            rationale=rationale
        )


    def _compare_forecast_to_historical(self, assumptions: DCFAssumptions, historical: Dict[str, Any]) -> Dict[str, Any]:
        if not historical or 'stats' not in historical:
            return {}
            
        stats = historical['stats']
        hist_growth = stats.get('avg_revenue_growth', 0.0) / 100.0  # Assuming it comes back as percentage e.g. 5.5 = 5.5%
        hist_margin = stats.get('avg_ebit_margin', 0.0) / 100.0
        
        # main.py expects: metric_name -> dict with forecast, historical, deviation, deviation_pct
        
        f_growth = assumptions.revenue_growth_rates[0] * 100
        h_growth = hist_growth * 100
        
        f_margin = assumptions.ebit_margin * 100
        h_margin = hist_margin * 100
        
        return {
            'revenue_growth': {
                'forecast': float(f_growth),
                'historical': float(h_growth),
                'deviation': float(f_growth - h_growth),
                'deviation_pct': float((f_growth - h_growth) / abs(h_growth) * 100 if h_growth != 0 else 0)
            },
            'ebit_margin': {
                'forecast': float(f_margin),
                'historical': float(h_margin),
                'deviation': float(f_margin - h_margin),
                'deviation_pct': float((f_margin - h_margin) / abs(h_margin) * 100 if h_margin != 0 else 0)
            }
        }


    def _run_stress_tests(self, assumptions: DCFAssumptions) -> Dict[str, float]:
        base_wacc = self._calculate_wacc(assumptions)
        return {
            'wacc_plus_200bps': base_wacc + 0.02,
            'margin_minus_500bps': max(0.01, assumptions.ebit_margin - 0.05),
            'growth_minus_500bps': assumptions.revenue_growth_rates[0] - 0.05
        }


    def _compute_regime_adjusted_values(self, result: DCFResult) -> Dict[str, float]:
        return {
            'risk_on_value': result.fair_value_per_share * 1.1,
            'risk_off_value': result.fair_value_per_share * 0.9,
            'high_inflation_value': result.fair_value_per_share * 0.85
        }


    def _compute_diagnostic_flags(self, result: DCFResult, assumptions: DCFAssumptions) -> List[str]:
        flags = []
        if result.pv_terminal_value / result.enterprise_value > 0.8:
            flags.append("High Terminal Value Dependency (>80% of EV)")
        if assumptions.terminal_growth_rate >= result.wacc:
            flags.append("Terminal Growth Rate >= WACC (Invalid Gordon Growth)")
        return flags


    def _build_visualization_payload(self, result: DCFResult) -> Dict[str, Any]:
        return {
            'waterfall': {'base': result.enterprise_value, 'net_debt': result.net_debt, 'equity': result.equity_value},
            'monte_carlo_hist': result.mc_distribution[:100] if result.mc_distribution else []
        }


    def _build_probabilistic_forecast(
        self,
        assumptions: DCFAssumptions,
        mc_distribution: List[float],
    ) -> Dict[str, Any]:
        """Institutional probabilistic forecast summary from Monte Carlo distribution."""
        if not mc_distribution:
            return {
                "horizon_years": assumptions.projection_years,
                "expected_price": assumptions.current_price,
                "prob_up_10pct": 0.0,
                "prob_down_10pct": 0.0,
                "distribution_moments": {"mean": assumptions.current_price, "std": 0.0, "skew_proxy": 0.0},
            }

        arr = np.array(mc_distribution)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        median = float(np.median(arr))
        p10_up = float(np.mean(arr >= assumptions.current_price * 1.10))
        p10_down = float(np.mean(arr <= assumptions.current_price * 0.90))
        skew_proxy = float((mean - median) / std) if std > 0 else 0.0

        return {
            "horizon_years": assumptions.projection_years,
            "expected_price": mean,
            "prob_up_10pct": p10_up,
            "prob_down_10pct": p10_down,
            "distribution_moments": {"mean": mean, "std": std, "skew_proxy": skew_proxy},
        }



    def _build_sensitivity_table(self, assumptions: DCFAssumptions) -> pd.DataFrame:
        """
        Build a sensitivity table around WACC and Terminal Growth Rate.
        """
        base_wacc = self._calculate_wacc(assumptions)
        base_tg = assumptions.terminal_growth_rate
        
        wacc_range = [base_wacc - 0.02, base_wacc - 0.01, base_wacc, base_wacc + 0.01, base_wacc + 0.02]
        tg_range = [base_tg - 0.01, base_tg - 0.005, base_tg, base_tg + 0.005, base_tg + 0.01]
        
        table = []
        for w in wacc_range:
            row = []
            for tg in tg_range:
                # Fast approximation of Fair Value for this cell
                tg_clipped = min(tg, w - 0.01) # Gordon Model cap
                # We need a mini-DCF to get FCFs
                fcfs = []
                rev = assumptions.base_revenue
                for _ in range(assumptions.projection_years):
                    rev *= (1 + assumptions.revenue_growth_rates[0])
                    nopat = rev * assumptions.ebit_margin * (1 - assumptions.tax_rate)
                    reinvest = rev * (assumptions.capex_pct_revenue + assumptions.nwc_change_pct_revenue - assumptions.da_pct_revenue)
                    fcfs.append(nopat - reinvest)
                
                # Discount
                pv_fcf = sum(f / ((1 + w) ** (i+1)) for i, f in enumerate(fcfs))
                tv = fcfs[-1] * (1 + tg_clipped) / (w - tg_clipped)
                pv_tv = tv / ((1 + w) ** assumptions.projection_years)
                ev = pv_fcf + pv_tv
                fv = (ev + assumptions.cash - assumptions.debt_value) / max(assumptions.shares_outstanding, 1.0)
                row.append(fv)
            table.append(row)
            
        df = pd.DataFrame(table, index=[f"{w*100:.1f}%" for w in wacc_range], columns=[f"{t*100:.1f}%" for t in tg_range])
        return df


    def _normalize_scenario_probabilities(self, scenarios: List[ScenarioResult]) -> List[ScenarioResult]:
        """Ensure scenario probabilities sum to 1.0 (non-destructive normalization)."""
        if not scenarios:
            return scenarios
        total = sum(max(0.0, s.probability) for s in scenarios)
        if total <= 0:
            w = 1.0 / len(scenarios)
            for s in scenarios:
                s.probability = w
            return scenarios
        for s in scenarios:
            s.probability = max(0.0, s.probability) / total
        return scenarios

    # ──────────────────────────────────────────────────────────────────────────
    # INSTITUTIONAL UPGRADE — terminal value diagnostics (Section 15)
    # ──────────────────────────────────────────────────────────────────────────
    def _terminal_diagnostics(self, result: DCFResult,
                              assumptions: DCFAssumptions) -> Dict[str, Any]:
        ev = result.enterprise_value or 0
        pv_tv = result.pv_terminal_value or 0
        tv_pct_ev = (pv_tv / ev) if ev else 0.0
        last_fcf = 0.0
        if result.line_items is not None and not result.line_items.empty:
            try:
                fcf_rows = [c for c in result.line_items.columns if "FCF" in str(c).upper()]
                if fcf_rows:
                    vals = pd.to_numeric(result.line_items[fcf_rows[0]], errors="coerce").dropna()
                    if len(vals):
                        last_fcf = float(vals.iloc[-1])
            except Exception:
                pass
        # Implied exit multiple at the terminal year (EV / terminal EBITDA proxy)
        implied_exit_multiple = None
        if last_fcf > 0 and ev > 0:
            # TV = EV * TV%/FCF -> implied FCF multiple at terminal
            implied_exit_multiple = (ev / last_fcf) if last_fcf else None
        diag = {
            "pv_terminal_value": pv_tv,
            "tv_as_pct_of_ev": tv_pct_ev,
            "implied_terminal_fcf_multiple": implied_exit_multiple,
            "wacc": result.wacc,
            "terminal_growth": assumptions.terminal_growth_rate,
            "wacc_minus_g": max(result.wacc - assumptions.terminal_growth_rate, 0.0),
            "flags": [],
        }
        if tv_pct_ev > 0.75:
            diag["flags"].append(
                f"Terminal value is {tv_pct_ev:.0%} of EV — the valuation is dominated by "
                "the terminal value; verify the long-run growth and WACC assumptions carefully."
            )
        if assumptions.terminal_growth_rate >= result.wacc:
            diag["flags"].append("Terminal growth >= WACC — Gordon growth formula is undefined.")
        return diag

    # ──────────────────────────────────────────────────────────────────────────
    # INSTITUTIONAL UPGRADE — key assumption attribution (Section 15/24)
    # ──────────────────────────────────────────────────────────────────────────
    def _key_assumption_attribution(self, assumptions: DCFAssumptions) -> Dict[str, Any]:
        """Identify which assumptions drive the majority of valuation by
        measuring each input's marginal impact on fair value per share."""
        base_fv = self._quick_dcf(assumptions)[1]
        if base_fv <= 0:
            return {"drivers": [], "base_fv": base_fv, "note": "Degenerate base valuation."}

        drivers = []

        def _probe(**kwargs):
            mod = DCFAssumptions(**{**assumptions.__dict__, **kwargs})
            return self._quick_dcf(mod)[1]

        # Revenue growth (year-1 growth +30% relative shift)
        g0 = assumptions.revenue_growth_rates[0]
        fv_g_up = _probe(revenue_growth_rates=[g0 * 1.5] + list(assumptions.revenue_growth_rates)[1:])
        fv_g_dn = _probe(revenue_growth_rates=[g0 * 0.5] + list(assumptions.revenue_growth_rates)[1:])
        drivers.append({
            "assumption": "Revenue growth",
            "value": f"{g0:.1%}",
            "+impact": (fv_g_up / base_fv - 1) * 100,
            "-impact": (fv_g_dn / base_fv - 1) * 100,
            "elasticity": ((fv_g_up - fv_g_dn) / base_fv) / (g0 * 1.0) if g0 else 0,
        })

        # EBIT margin ±200bps
        m = assumptions.ebit_margin
        fv_m_up = _probe(ebit_margin=min(m + 0.02, 0.90))
        fv_m_dn = _probe(ebit_margin=max(m - 0.02, 0.01))
        drivers.append({
            "assumption": "EBIT margin",
            "value": f"{m:.1%}",
            "+impact": (fv_m_up / base_fv - 1) * 100,
            "-impact": (fv_m_dn / base_fv - 1) * 100,
            "elasticity": ((fv_m_up - fv_m_dn) / base_fv) / 0.04 if m else 0,
        })

        # WACC ±100bps
        w = self._calculate_wacc(assumptions)
        cost_debt_delta = 0.01
        fv_w_up = _probe(cost_of_debt=assumptions.cost_of_debt + cost_debt_delta,
                         risk_free_rate=assumptions.risk_free_rate + cost_debt_delta)
        fv_w_dn = _probe(cost_of_debt=max(assumptions.cost_of_debt - cost_debt_delta, 0.01),
                         risk_free_rate=max(assumptions.risk_free_rate - cost_debt_delta, 0.01))
        drivers.append({
            "assumption": "WACC (discount rate)",
            "value": f"{w:.1%}",
            "+impact": (fv_w_dn / base_fv - 1) * 100,   # lower WACC → higher FV
            "-impact": (fv_w_up / base_fv - 1) * 100,
            "elasticity": ((fv_w_dn - fv_w_up) / base_fv) / (2 * cost_debt_delta) if w else 0,
        })

        # Terminal growth ±50bps
        tg = assumptions.terminal_growth_rate
        fv_tg_up = _probe(terminal_growth_rate=min(tg + 0.005, w - 0.01))
        fv_tg_dn = _probe(terminal_growth_rate=max(tg - 0.005, 0.0))
        drivers.append({
            "assumption": "Terminal growth",
            "value": f"{tg:.2%}",
            "+impact": (fv_tg_up / base_fv - 1) * 100,
            "-impact": (fv_tg_dn / base_fv - 1) * 100,
            "elasticity": ((fv_tg_up - fv_tg_dn) / base_fv) / 0.01 if tg else 0,
        })

        # Sort by absolute impact (drivers ranked by what matters most)
        drivers.sort(key=lambda d: -max(abs(d["+impact"]), abs(d["-impact"])))
        total_abs = sum(max(abs(d["+impact"]), abs(d["-impact"])) for d in drivers)
        for d in drivers:
            d["share_of_total_impact"] = (max(abs(d["+impact"]), abs(d["-impact"])) / total_abs) if total_abs else 0.0

        top = [d["assumption"] for d in drivers[:2]]
        return {
            "drivers": drivers,
            "base_fv": base_fv,
            "top_drivers": top,
            "note": (
                "Impact measured as % fair-value change from ±shock to each assumption "
                "(growth ±50% relative, margin ±200bps, WACC ±100bps, terminal growth ±50bps). "
                "Ranked by absolute impact — these are the assumptions to stress-test first."
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # INSTITUTIONAL UPGRADE — market consensus layer (Sections 6/7/28)
    # ──────────────────────────────────────────────────────────────────────────
    def _consensus_layer(self, assumptions: DCFAssumptions,
                         result: DCFResult) -> Dict[str, Any]:
        """Wire the shared market-consensus / sentiment / dislocation layer
        into the DCF. Defensive: never raises even if the shared module is
        unavailable."""
        try:
            from market_consensus_engine import (
                ConsensusEstimates,
                SentimentSnapshot,
                reverse_dcf_expectations,
                ConsensusDislocationEngine,
                get_consensus_dislocation_engine,
            )

            implied = reverse_dcf_expectations(
                model_price=float(result.fair_value_per_share or 0),
                current_price=float(assumptions.current_price or 0),
                shares=float(assumptions.shares_outstanding or 0),
                net_debt=float(assumptions.debt_value - assumptions.cash),
                base_revenue=float(assumptions.base_revenue),
                model_revenue_growth=float(assumptions.revenue_growth_rates[0] or 0),
                model_ebit_margin=float(assumptions.ebit_margin),
                wacc=float(result.wacc or 0.10),
                terminal_growth=float(assumptions.terminal_growth_rate),
                projection_years=int(assumptions.projection_years),
            )

            dislocation = get_consensus_dislocation_engine().analyze(
                implied=implied,
                model_price=float(result.fair_value_per_share or 0),
                current_price=float(assumptions.current_price or 0),
                model_upside_pct=float(
                    (result.fair_value_per_share / assumptions.current_price - 1)
                    if assumptions.current_price > 0 else 0.0
                ),
            )

            return {
                "implied_expectations": implied.to_dict(),
                "dislocation": dislocation.to_dict(),
                "provenance": {
                    "market_price": {"status": "verified", "source": "live market"},
                    "model_price": {"status": "derived", "source": "DCF model"},
                    "implied_growth": {"status": "estimated", "source": "reverse DCF identity"},
                },
            }
        except Exception as e:
            return {"error": str(e), "note": "Consensus layer unavailable."}

    # ──────────────────────────────────────────────────────────────────────────
    # INSTITUTIONAL UPGRADE — dynamic scenario notes (Section 8)
    # ──────────────────────────────────────────────────────────────────────────
    def _dynamic_scenario_notes(self, assumptions: DCFAssumptions,
                                result: DCFResult) -> str:
        """Explain how many scenarios are appropriate and why, based on the
        situation rather than a fixed Bear/Base/Bull template."""
        n = len(result.scenarios or [])
        growth = assumptions.revenue_growth_rates[0] if assumptions.revenue_growth_rates else 0.0
        margin = assumptions.ebit_margin
        parts = [
            f"{n} materially distinct scenario(s) modeled. The scenario architecture is "
            "situation-driven: for this company, growth, margin, discount rate and terminal "
            "assumptions are the material variables."
        ]
        if growth > 0.20 or margin < 0.05:
            parts.append(
                "High growth / thin-margin profile: valuation is heavily regime-dependent, "
                "so downside and upside cases are wide and probability-weighted expectations "
                "matter more than any single point."
            )
        if result.terminal_diagnostics.get("tv_as_pct_of_ev", 0) > 0.75:
            parts.append(
                "Terminal-value dominance (>75% of EV): scenario dispersion is amplified by "
                "terminal assumptions; WACC and long-run growth scenarios deserve extra weight."
            )
        return " ".join(parts)


# --- GLOBAL GETTERS ---

def _safe_float(val, default: float = 0.0) -> float:
  try:
    if val is None:
      return default
    f = float(val)
    if np.isnan(f) or np.isinf(f):
      return default
    return f
  except (TypeError, ValueError):
    return default


def _extract_ebitda_millions(ticker_obj, info: Dict[str, Any]) -> float:
  """Derive LTM EBITDA in $M from yfinance info and financial statements."""
  ebitda_raw = info.get("ebitda")
  if ebitda_raw and abs(_safe_float(ebitda_raw)) > 1e6:
    return abs(_safe_float(ebitda_raw)) / 1e6

  fin = getattr(ticker_obj, "financials", None)
  if fin is not None and not fin.empty:
    for row_name in ("EBITDA", "Normalized EBITDA"):
      if row_name in fin.index:
        val = _safe_float(fin.loc[row_name].iloc[0])
        if val:
          return abs(val) / 1e6

    ebit = None
    da = 0.0
    for ebit_name in ("EBIT", "Operating Income"):
      if ebit_name in fin.index:
        ebit = _safe_float(fin.loc[ebit_name].iloc[0])
        break
    for da_name in (
      "Reconciled Depreciation",
      "Depreciation And Amortization",
      "Depreciation",
    ):
      if da_name in fin.index:
        da = _safe_float(fin.loc[da_name].iloc[0])
        break
    if ebit is not None:
      return abs(ebit + da) / 1e6

  revenue = _safe_float(info.get("totalRevenue"))
  margin = info.get("ebitdaMargins") or info.get("operatingMargins")
  if revenue > 1e6 and margin:
    return abs(revenue * _safe_float(margin)) / 1e6
  # No fabricated fallback: report "no data" so callers clamp to a sane
  # minimum instead of inventing $500M of EBITDA for a company we know
  # nothing about.
  return 0.0


def _extract_eps(info: Dict[str, Any], ticker_obj) -> Optional[float]:
  """Derive trailing EPS from info or income statement."""
  eps = _safe_float(info.get("trailingEps"), 0)
  if eps:
    return eps

  fin = getattr(ticker_obj, "financials", None)
  shares = _safe_float(info.get("sharesOutstanding"), 0)
  if fin is not None and not fin.empty and shares > 0:
    for row_name in ("Net Income", "Net Income Common Stockholders"):
      if row_name in fin.index:
        net_income = _safe_float(fin.loc[row_name].iloc[0])
        if net_income:
          return net_income / shares
  return None


def _resolve_live_price(sym: str, ticker_obj, info: Dict[str, Any]) -> Optional[float]:
  """Resolve the best available live price from multiple sources."""
  from data_sources import get_realtime_price

  current_price, prev_close = get_realtime_price(sym)
  if current_price and current_price > 0:
    return float(current_price)
  if prev_close and prev_close > 0:
    return float(prev_close)

  for key in ("currentPrice", "regularMarketPrice", "previousClose"):
    val = _safe_float(info.get(key), 0)
    if val > 0:
      return val

  try:
    fi = ticker_obj.fast_info
    val = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
    if val and float(val) > 0:
      return float(val)
  except Exception:
    pass

  try:
    hist = ticker_obj.history(period="1mo")
    if hist is not None and not hist.empty:
      close = hist["Close"]
      if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
      close = close.dropna()
      if len(close) > 0:
        return float(close.iloc[-1])
  except Exception:
    pass

  return None


def _validate_ticker_fundamentals(
  sym: str,
  info: Dict[str, Any],
  price: Optional[float],
  shares_raw: float,
  eps: Optional[float],
  ticker_obj,
) -> None:
  """Raise when yfinance did not return usable data for the symbol."""
  has_name = bool(info.get("shortName") or info.get("longName"))
  has_cap = _safe_float(info.get("marketCap"), 0) > 1e8
  has_revenue = _safe_float(info.get("totalRevenue"), 0) > 1e6
  has_shares = shares_raw > 1e6
  has_quote = info.get("quoteType") in ("EQUITY", "ETF", "MUTUALFUND")

  has_history = False
  try:
    hist = ticker_obj.history(period="1mo")
    has_history = hist is not None and len(hist) >= 3
  except Exception:
    pass

  if has_cap or (has_revenue and has_name) or (has_quote and has_shares):
    return
  if has_history and price and price > 0 and has_shares and eps:
    return
  if has_history and price and price > 0 and has_name:
    return

  raise ValueError(
    f"No market data found for '{sym}'. "
    "Check the ticker symbol and try again."
  )


def _safe_info_fetch(ticker_obj) -> Dict[str, Any]:
    """Fetch yfinance `info` defensively.

    `Ticker.info` is a known source of arbitrary exceptions (including
    `TypeError: argument of type 'NoneType' is not iterable`) when Yahoo's
    quote endpoint returns a partial payload. Never let that escape — every
    field is individually guarded downstream anyway.
    """
    try:
        raw = getattr(ticker_obj, "info", None)
        if raw is None:
            return {}
        # Some yfinance versions return a list or scalar on failure
        if not isinstance(raw, dict):
            try:
                raw = dict(raw)
            except Exception:
                return {}
        return {k: v for k, v in raw.items() if v is not None}
    except Exception:
        return {}


def _derive_revenue_millions(info: Dict[str, Any], shares_raw: float) -> float:
    """Derive revenue in $M, correctly handling per-share fallback.

    Yahoo's `info` often omits `totalRevenue` while providing
    `revenuePerShare`. Using the per-share figure directly used to produce
    ~$100M for mega-caps (e.g., AAPL) instead of ~$390B. Multiply by shares
    outstanding so the magnitude is correct.
    """
    total_rev = _safe_float(info.get("totalRevenue"), 0)
    if total_rev > 1e6:
        return total_rev / 1e6
    rps = _safe_float(info.get("revenuePerShare"), 0)
    if rps > 0 and shares_raw > 0:
        return rps * shares_raw / 1e6
    # No last-resort fabrication: without shares, a per-share figure cannot be
    # scaled to total revenue, so report 0 ("n/a") rather than inventing a
    # magnitude. Callers clamp / show n/a.
    return 0.0


def fetch_ticker_fundamentals(ticker: str) -> Dict[str, Any]:
  """
  Fetch live ticker fundamentals for DCF / LBO / M&A auto-fill.
  Uses realtime price + yfinance info and financial statements.
  Fully defensive: never raises on partial data (each field is guarded);
  raises a clear ValueError only when NO usable market data exists.
  """
  sym = (ticker or "").strip().upper()
  if not sym:
    raise ValueError("Ticker is required")

  import yfinance as yf

  t = yf.Ticker(sym)
  info = _safe_info_fetch(t)

  price = _resolve_live_price(sym, t, info)

  shares_raw = _safe_float(info.get("sharesOutstanding"), 0)
  eps = _extract_eps(info, t)

  try:
    _validate_ticker_fundamentals(sym, info, price, shares_raw, eps, t)
  except ValueError:
    # Last-ditch: fast_info may still give price + shares even when `info`
    # came back empty (the common NoneType crash path).
    try:
      fi = t.fast_info
      p2 = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
      if p2 and float(p2) > 0 and price in (None, 0):
        price = float(p2)
      s2 = getattr(fi, "shares", None)
      if s2 and float(s2) > 0 and shares_raw <= 0:
        shares_raw = float(s2)
    except Exception:
      pass
    if (not price or price <= 0) and (not shares_raw or shares_raw <= 0):
      raise

  if not price or price <= 0:
    market_cap = _safe_float(info.get("marketCap"), 0)
    if market_cap > 0 and shares_raw > 0:
      price = market_cap / shares_raw
    else:
      raise ValueError(
        f"Could not resolve a live price for '{sym}'. "
        "Check the ticker symbol and try again."
      )

  if not shares_raw or shares_raw <= 0:
    market_cap = _safe_float(info.get("marketCap"), 0)
    if market_cap > 0 and price > 0:
      shares_raw = market_cap / price
    else:
      raise ValueError(f"Shares outstanding unavailable for '{sym}'.")

  if eps is None or eps == 0:
    fin = getattr(t, "financials", None)
    if fin is not None and not fin.empty and shares_raw > 0:
      for row_name in ("Net Income", "Net Income Common Stockholders"):
        if row_name in fin.index:
          net_income = _safe_float(fin.loc[row_name].iloc[0])
          if net_income:
            eps = net_income / shares_raw
            break
  if eps is None or eps == 0:
    raise ValueError(f"EPS unavailable for '{sym}'.")

  revenue_m = _derive_revenue_millions(info, shares_raw)

  ebitda_m = _extract_ebitda_millions(t, info)

  ebit_margin = _safe_float(info.get("operatingMargins"), 0.20)
  if ebitda_m > 0 and revenue_m > 0:
    ebitda_margin = ebitda_m / revenue_m
  else:
    ebitda_margin = _safe_float(info.get("ebitdaMargins"), 0.20)
  ebitda_margin = max(min(ebitda_margin, 0.80), 0.05)

  beta = _safe_float(info.get("beta"), 1.0)
  market_cap_m = _safe_float(info.get("marketCap")) / 1e6
  if market_cap_m <= 0:
    market_cap_m = price * shares_raw / 1e6

  debt_m = _safe_float(info.get("totalDebt")) / 1e6
  cash_m = _safe_float(info.get("totalCash")) / 1e6
  shares_m = shares_raw / 1e6

  rev_growth = _safe_float(
    info.get("revenueGrowth") or info.get("earningsGrowth"),
    0.08,
  )
  rev_growth = max(min(rev_growth, 0.60), -0.10)

  tax_rate = _safe_float(info.get("effectiveTaxRate"), 0.21)
  tax_rate = max(min(tax_rate, 0.40), 0.05)

  enterprise_value = _safe_float(info.get("enterpriseValue"))
  entry_multiple = (
    round(enterprise_value / (ebitda_m * 1e6), 1)
    if enterprise_value > 0 and ebitda_m > 0
    else 10.0
  )
  entry_multiple = float(max(min(entry_multiple, 25.0), 4.0))

  cost_of_debt = 0.08
  if info.get("interestExpense") and debt_m > 0:
    cost_of_debt = abs(_safe_float(info.get("interestExpense"))) / (debt_m * 1e6)
    cost_of_debt = max(min(cost_of_debt, 0.15), 0.04)
  elif market_cap_m > 0 and debt_m > 0:
    cost_of_debt = 0.06 + min(debt_m / market_cap_m, 1.0) * 0.04

  leverage = 5.0
  if ebitda_m > 0 and debt_m > 0:
    leverage = round(debt_m / ebitda_m, 1)
    leverage = float(max(min(leverage, 7.0), 3.0))

  return {
    "ticker": sym,
    "price": round(price, 2),
    "revenue_m": round(revenue_m, 1),
    "revenue_growth": round(rev_growth * 100, 1),
    "ebit_margin_pct": round(ebit_margin * 100, 1),
    "ebitda_m": round(ebitda_m, 1),
    "ebitda_margin": round(ebitda_margin, 3),
    "tax_rate_pct": round(tax_rate * 100, 1),
    "beta": round(beta, 2),
    "market_cap_m": round(market_cap_m, 0),
    "debt_m": round(debt_m, 0),
    "cash_m": round(cash_m, 0),
    "shares_m": round(shares_m, 1),
    "eps": round(eps, 2),
    "entry_multiple": entry_multiple,
    "exit_multiple": entry_multiple,
    "leverage_multiple": leverage,
    "interest_rate_pct": round(cost_of_debt * 100, 2),
    "enterprise_value_m": round(enterprise_value / 1e6, 0) if enterprise_value else round(
      market_cap_m + debt_m - cash_m, 0
    ),
  }


def get_dcf_engine() -> InstitutionalDCFEngine:
    """Return a singleton instance of the Institutional DCF Engine."""
    global _dcf_engine_instance
    if _dcf_engine_instance is None:
        _dcf_engine_instance = InstitutionalDCFEngine()
    return _dcf_engine_instance

def get_financial_generator():
    """Alias for backwards compatibility if needed."""
    return get_dcf_engine()