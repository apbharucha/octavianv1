"""Octavian Document Analyzer — Financial Document Intelligence Engine
Analyzes uploaded financial documents and SEC filings.
Functions as a strict, integrity-first data validation engine.

Author: APB - Octavian Team
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from quant_ensemble_model import get_quant_ensemble
from data_sources import get_stock
from octavian_theme import COLORS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOCUMENT PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FinancialDocumentParser:
    """Extracts structured financial metrics and narrative signals from text."""
    _PATTERNS = {
        "revenue": [
            r'(?i)(?:total\s+)?revenue[s]?\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?',
            r'(?i)(?:net\s+)?sales\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?',
        ],
        "cogs": [
            r'(?i)(?:cost\s+of\s+goods\s+sold|cost\s+of\s+sales|COGS|cost\s+of\s+revenue)\s*(?:was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "gross_profit": [
            r'(?i)(?:gross\s+profit|gross\s+margin\s+\$)\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "opex": [
            r'(?i)(?:operating\s+expenses?|OpEx)\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "rd": [
            r'(?i)(?:research\s+and\s+development|R&D)\s*(?:expenses?)?\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "sm": [
            r'(?i)(?:sales\s+and\s+marketing|S&M|marketing)\s*(?:expenses?)?\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "ga": [
            r'(?i)(?:general\s+and\s+administrative|G&A)\s*(?:expenses?)?\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "op_inc": [
            r'(?i)(?:operating\s+income|operating\s+profit)\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "net_inc": [
            r'(?i)(?:net\s+income|net\s+profit|net\s+earnings)\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "cfo": [
            r'(?i)(?:cash\s+(?:flow\s+)?from\s+operations|operating\s+cash\s+flow|CFO)(?:\s*\(?CFO\)?)?\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "capex": [
            r'(?i)(?:capital\s+expenditure[s]?|CapEx)\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "liabilities": [
            r'(?i)(?:total\s+)?liabilities\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?',
            r'(?i)debt\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "assets": [
            r'(?i)(?:total\s+)?assets\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?',
            r'(?i)cash\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "equity": [
            r'(?i)(?:total\s+)?(?:stockholders?\'?\s+)?equity\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "ar": [
            r'(?i)(?:accounts\s+receivable|trade\s+receivable|receivables|AR)\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "deferred_rev": [
            r'(?i)(?:deferred\s+revenue|unearned\s+revenue)\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?'
        ],
        "sbc": [
            r'(?i)(?:stock[- ]based\s+compensation|share[- ]based\s+compensation|SBC)\s*(?:expense)?\s*(?:of|was|were|:)?\s*\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?']
}

    _GROWTH_PATTERNS = [
        r'(?i)(?:revenue|sales|top[- ]?line)\s+(?:grew|growth|increased|rose|up|down)\s+(?:by\s+)?([+\-]?[\d.]+)\s*%',
        r'(?i)[(]*\s*([+\-]?[\d.]+)\s*%\s*(?:year[- ]over[- ]year|YoY|y/y)\s*(?:growth|increase)?\s*[)]*',
]

    _MARGIN_ATTRS = [
        r'(?i)(?:gross|operating|net|EBITDA|EBIT)\s+margin[s]?\s*(?:of|was|were|:)?\s*([+\-]?[\d.]+)\s*%']

    _TICKER_PATTERN = r'\b([A-Z]{1,5})\b'
    def _segment_document(self, text: str) -> List[Dict[str, str]]:
        """Splits a document into segments if it contains multiple companies."""
        # Find potential major entity boundaries
        # We look for "COMPANY: [Name]"at the start of a line or after a divider
        boundaries = [m.start() for m in re.finditer(r'(?im)^\s*(?:company|firm):\s*(.*)$', text)]
        
        # If no "Company:"headers found, look for "Ticker: [TKR]"at start of line
        if not boundaries:
             boundaries = [m.start() for m in re.finditer(r'(?im)^\s*(?:ticker|symbol):\s*([A-Z0-9.\- ]{1,10})$', text)]
        
        # Also check for common section dividers or header patterns
        if not boundaries:
             boundaries = [m.start() for m in re.finditer(r'[\-=*]{5,}', text)]
        
        if not boundaries or len(boundaries) < 2:
            return [{"name": "Primary Entity", "text": text}]

        boundaries = sorted(list(set(boundaries)))
        segments = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i+1] if i+1 < len(boundaries) else len(text)
            chunk = text[start:end].strip()
            
            if len(chunk) < 50: continue # Skip tiny fragments
            
            # Identify name for this chunk
            name_match = re.search(r'(?i)(?:company|ticker|symbol|firm):\s*([A-Z0-9.\- ]{2,30})', chunk)
            entity_name = name_match.group(1).strip() if name_match else f"Segment {i+1}"            
            segments.append({"name": entity_name, "text": chunk})
            
        return segments if segments else [{"name": "Primary Entity", "text": text}]

    def _identify_tier(self, pos: int, text: str) -> int:
        """Determines the confidence tier of an extracted metric based on local context."""
        context = text[max(0, pos-100):min(len(text), pos+100)]
        
        # Tier 1: Official / Audited / Primary markers
        t1_markers = ["audited", "official", "filing", "form 10-k", "form 10-q", "consolidated", "statement of"]
        if any(m in context for m in t1_markers):
            return 1
            
        # Tier 3: Estimates / Projections / Rumors
        t3_markers = ["estimate", "forecast", "projection", "rumor", "unverified", "look-forward"]
        if any(m in context for m in t3_markers):
            return 3
            
        # Default: Standard Contextual Detection
        return 2

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Processes text, potentially splitting into multiple entities."""
        segments = self._segment_document(text)
        all_results = []
        
        for seg in segments:
            seg_text = seg["text"]
            text_lower = seg_text.lower()
            
            result: Dict[str, Any] = {
                "entity_name": seg["name"],
                "text": seg_text, # Include raw text for secondary scan
                "metrics": {}, # Maps metric -> list of (value, tier)
                "growth_rates": [],
                "explicit_margins": [],
                "all_large_numbers": [],
                "tickers_mentioned": [],
                "sentiment_cues": {"positive": 0, "negative": 0},
                "raw_length": len(seg_text)
}

            # Phase 1: Parse targeted absolute financials with Tiering
            for metric, patterns in self._PATTERNS.items():
                result["metrics"][metric] = []
                for pat in patterns:
                    for m in re.finditer(pat, text_lower):
                        try:
                            # Clean the captured string: remove $, comma, spaces
                            raw_val = m.group(1).replace(",", "").replace("$", "").strip()
                            val = float(raw_val)
                            unit = (m.group(2) or "").lower() if m.lastindex >= 2 else ""
                            if unit.startswith("b"): val *= 1_000_000_000
                            elif unit.startswith("m"): val *= 1_000_000
                            
                            tier = self._identify_tier(m.start(), text_lower)
                            result["metrics"][metric].append((val, tier))
                        except Exception:
                            pass

            # Phase 1b: Secondary Mandatory CapEx Rescan (FCF FALLBACK RULE)
            if not result["metrics"].get("capex"):
                # Looking for standalone "CapEx"or "Capital Expenditure"with numbers nearby even if not immediately adjacent
                # or in different formats.
                capex_secondary = re.findall(r'(?i)(?:capital\s+expenditure[s]?|capex)[^0-9]{0,20}\(?\s*([+\-]?\s*\$?\s*[+\-]?[\d,.]+)\s*[)]*\s*(billion|million|b|m|bn|mn)?', text_lower)
                for val_str, unit in capex_secondary:
                    try:
                        clean_val = val_str.replace(",", "").replace("$", "").strip()
                        val = float(clean_val)
                        unit = unit.lower() if unit else ""
                        if unit.startswith("b"): val *= 1_000_000_000
                        elif unit.startswith("m"): val *= 1_000_000
                        result["metrics"]["capex"].append((val, 3)) # Mark as Tier 3 since it's a fallback scan
                    except Exception: pass

            # Phase 2: Parse growth rates & explicit margins
            for pat in self._GROWTH_PATTERNS:
                for m in re.finditer(pat, text_lower):
                    try: result["growth_rates"].append(float(m.group(1)))
                    except Exception: pass

            for pat in self._MARGIN_ATTRS:
                for m in re.finditer(pat, text_lower):
                    try: result["explicit_margins"].append(float(m.group(1)))
                    except Exception: pass

            # Phase 3: Scan all generic dollar amounts
            for m in re.finditer(r'\$([\d,.]+)\s*(billion|million|B|M|bn|mn)?', text_lower):
                try:
                    val = float(m.group(1).replace(",", ""))
                    unit = (m.group(2) or "").lower() if m.lastindex >= 2 else ""
                    if unit.startswith("b"): val *= 1_000_000_000
                    elif unit.startswith("m"): val *= 1_000_000
                    result["all_large_numbers"].append(val)
                except Exception: pass

            # Phase 4: Tickers and Sentiment
            common_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "LET", "SAY", "SHE", "TOO", "USE", "NET", "EPS", "TTM", "YOY", "FY", "CEO", "CFO", "COO", "IPO", "ETF", "SEC", "GDP", "CPI", "FED", "PCE", "PPI", "PMI", "USA", "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "EBIT", "DEBT", "CASH", "COST", "LOSS", "GAIN", "SALE", "SOLD", "HIGH", "LOW", "LONG", "SHORT", "BULL", "BEAR", "COGS", "OPE"}
            raw_tickers = set(re.findall(self._TICKER_PATTERN, seg_text))
            result["tickers_mentioned"] = sorted(raw_tickers - common_words)

            pos_w = ["strong", "growth", "beat", "exceeded", "raised", "expanded", "momentum", "accelerat", "outperform", "upside", "record", "surge", "bullish", "upgrade", "optimis"]
            neg_w = ["weak", "miss", "below", "lower", "decline", "contract", "headwind", "downside", "underperform", "concern", "risk", "bearish", "downgrade", "pessimis", "recession", "bankruptcy", "default"]
            for w in pos_w: result["sentiment_cues"]["positive"] += text_lower.count(w)
            for w in neg_w: result["sentiment_cues"]["negative"] += text_lower.count(w)
            
            all_results.append(result)

        return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRIORITY LAYER: SANITY → FCF → METRICS → PARSE vs REAL → SCORE → ALLOC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _fcf_locked(cfo: float, capex: float) -> float:
    """FCF = CFO − CapEx. Never substitute CFO for FCF."""
    return float(cfo) - float(capex)


def apply_numerical_sanity_gate(vals: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """
    HARD GATE: normalize ordering, flag parse errors, discard impossible metrics.
    Revenue ≥ Operating Income ≥ Net Income (when all positive and comparable).
    Net Income must not exceed Revenue. Values >10× revenue → PARSE ERROR (metric discarded).
    """
    v = dict(vals)
    parse_errors: List[str] = []
    rev = v.get("revenue", 0) or 0

    if rev > 0:
        for k in list(v.keys()):
            if k in ("assets", "liabilities", "equity", "ar", "deferred_rev"):
                continue
            if abs(v[k]) > abs(rev) * 10 and v[k] != 0:
                parse_errors.append(
                    f"PARSE ERROR: {k.upper()} magnitude suggests unit/scale mismatch — discarded for scoring.")
                v[k] = 0.0

        if v.get("net_inc", 0) > rev * 1.05:
            parse_errors.append("PARSE ERROR: Net Income exceeded Revenue — treated as extraction error; NI cleared.")
            v["net_inc"] = 0.0
        if v.get("op_inc", 0) > rev * 1.05:
            parse_errors.append("PARSE ERROR: Operating Income exceeded Revenue — OI cleared.")
            v["op_inc"] = 0.0

    gp, oi, ni = v.get("gross_profit", 0), v.get("op_inc", 0), v.get("net_inc", 0)
    if gp > 0 and oi > gp * 1.001:
        parse_errors.append("PARSE ERROR: Operating Income > Gross Profit — OI adjusted down.")
        v["op_inc"] = min(oi, gp * 0.999)
        oi = v["op_inc"]
    if oi > 0 and ni > oi * 1.001:
        parse_errors.append("PARSE ERROR: Net Income > Operating Income — NI adjusted down.")
        v["net_inc"] = min(ni, oi * 0.999)

    if rev > 0 and gp > 0 and oi > 0 and not (rev >= oi >= ni - 1e-6):
        if oi > rev:
            v["op_inc"] = min(oi, rev * 0.99)
        if ni > v.get("op_inc", 0):
            v["net_inc"] = min(ni, v["op_inc"] * 0.99)
        parse_errors.append("PARSE ERROR: Enforced Revenue ≥ OI ≥ NI ordering for internal consistency.")

    return v, parse_errors


def classify_real_financial_risks(vals: Dict[str, float], implied_fcf: float, text: str = "") -> List[str]:
    """REAL risks (not parse) — these directly affect investment scores."""
    risks: List[str] = []
    rev = vals.get("revenue", 0) or 0
    text_lower = text.lower()
    
    # 1. FCF Evaluation & MANDATORY Conversion Warning
    if implied_fcf is not None:
        if implied_fcf < 0 and vals.get("cfo", 0) != 0:
            risks.append(f"REAL: Negative free cash flow (Recomputed: {implied_fcf:,.0f}).")
        elif rev > 0:
            fcf_margin = (implied_fcf / rev) * 100
            if fcf_margin < 5.0:
                risks.append(f"REAL: Weak FCF Conversion (Margin: {fcf_margin:.1f}%) — warning.")
        
    # 2. SBC / Shareholder Dilution (Evaluate % of revenue)
    if rev > 0 and vals.get("sbc", 0) > rev * 0.10:
        sbc_pct = (vals["sbc"] / rev) * 100
        risks.append(f"REAL: Dilution Risk (SBC load: {sbc_pct:.1f}% of revenue).")
        
    # 3. Narrative Risk Detection: Accounts Receivable up X%
    # MANDATORY: Search NOTES section (or entire block) for AR spikes
    ar_spike = re.search(r'(?i)(?:accounts\s+receivable|AR)\s+(?:up|increased|rose)\s+(\d+)%', text_lower)
    if ar_spike:
        spike_pct = ar_spike.group(1)
        risks.append(f"REAL: Revenue Quality Risk (Narrative: AR up {spike_pct}%).")
    elif rev > 0 and vals.get("ar", 0) > rev * 0.30:
        ar_pct = (vals["ar"] / rev) * 100
        risks.append(f"REAL: High AR Intensity ({ar_pct:.1f}% of revenue) — collection risk.")
        
    # 4. Margin Integrity
    if rev > 0 and vals.get("op_inc", 0) != 0:
        op_margin = (vals["op_inc"] / rev) * 100
        if op_margin < 5:
            risks.append(f"REAL: Margin Compression (Operating Margin: {op_margin:.1f}%).")
            
    # 5. Asset/Liability Stress
    if vals.get("liabilities", 0) > 0 and vals.get("equity", 0) > 0:
        lev = vals["liabilities"] / max(vals["equity"], 1e-6)
        if lev > 2.5:
            risks.append(f"REAL: High Leverage (Liab/Equity ratio: {lev:.2f}x).")
            
    return risks


def enforce_score_spreads(company_stats: Dict[str, dict]) -> Dict[str, dict]:
    """
    SCORE SPREAD ENFORCEMENT:
    If one company has higher margins, higher FCF, and fewer risks, its score     MUST be at least 10 points higher than the next peer.
    """
    if len(company_stats) < 2:
        return company_stats

    sorted_names = sorted(
        company_stats.keys(),
         key=lambda k: (
            company_stats[k].get("margin", 0),
             company_stats[k].get("fcf_margin", 0),
            -len(company_stats[k].get("real_financial_risks", []))
        ),
         reverse=True
)

    for i in range(len(sorted_names) - 1):
        top = sorted_names[i]
        bot = sorted_names[i+1]
        
        # Check if top is strictly better in all 3 pillars
        m_gap = company_stats[top]["margin"] > company_stats[bot]["margin"]
        f_gap = company_stats[top]["fcf_margin"] > company_stats[bot]["fcf_margin"]
        r_gap = len(company_stats[top]["real_financial_risks"]) < len(company_stats[bot]["real_financial_risks"])
        
        if m_gap and f_gap and r_gap:
            # Enforce at least 10 point spread
            current_gap = company_stats[top]["total_score"] - company_stats[bot]["total_score"]
            if current_gap < 10:
                adjustment = 10 - current_gap
                company_stats[top]["total_score"] = min(100, company_stats[top]["total_score"] + adjustment)
                
    return company_stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOCUMENT INTELLIGENCE ENGINE (Structured report; markdown output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DocumentIntelligenceEngine:
    """Computes advanced metrics, forces strict validations, handles self-correction."""
    def _safe_mean(self, lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    def _resolve_tiered_metric(self, metric_name: str, values_with_tiers: List[Tuple[float, int]]) -> Tuple[float, int, List[str]]:
        """Resolves a single value from tiered sources. Tier 1 > Tier 2 > Tier 3."""
        if not values_with_tiers:
            return 0.0, 0, []
        
        logs = []
        t1 = [v for v, t in values_with_tiers if t == 1]
        t2 = [v for v, t in values_with_tiers if t == 2]
        t3 = [v for v, t in values_with_tiers if t == 3]
        
        if t1:
            val = np.median(t1)
            if len(set(t1)) > 1:
                logs.append(f"[CONFLICT] Multiple Tier 1 values for {metric_name.upper()}. Using median.")
            return float(val), 1, logs
        
        if t2:
            val = np.median(t2)
            if len(set(t2)) > 1:
                logs.append(f"[CONFLICT] Multiple Tier 2 values for {metric_name.upper()}. Using median.")
            return float(val), 2, logs
            
        return float(np.median(t3)), 3, []
        
    def _normalize_scale(self, vals: Dict[str, float], tiers: Dict[str, int]) -> Tuple[Dict[str, float], List[str]]:
        """Detects unit consistency and normalizes metrics (e.g. raw to billions) based on Revenue scale."""
        adj_vals = vals.copy()
        logs = []
        rev = vals.get("revenue", 0)
        if rev == 0: return adj_vals, logs
        
        # Scaling candidates (metrics that are typically a fraction of revenue)
        candidates = ["gross_profit", "op_inc", "net_inc", "cfo", "capex", "assets", "liabilities", "equity", "sm", "rd", "ga"]
        
        for c in candidates:
            if c not in adj_vals or adj_vals[c] == 0: continue
            
            # If a value is > 0 but < 1/1000th of revenue, it's likely a unit mismatch (e.g. 3.2 instead of 3.2B)
            # We attempt to scale up if the scaled value is still <= revenue (for IS items) or justified
            if abs(adj_vals[c]) < (abs(rev) / 1000):
                # Try Billions
                scaled_b = adj_vals[c] * 1e9
                if abs(scaled_b) <= abs(rev) * 10: # Allow up to 10x for Balance Sheet items
                    adj_vals[c] = scaled_b
                    logs.append(f"SCALE CORRECTION: Normalized {c.upper()} from raw to billions based on revenue magnitude.")
                else:
                    # Try Millions
                    scaled_m = adj_vals[c] * 1e6
                    if abs(scaled_m) <= abs(rev) * 10:
                        adj_vals[c] = scaled_m
                        logs.append(f"SCALE CORRECTION: Normalized {c.upper()} from raw to millions based on revenue magnitude.")
        
        return adj_vals, logs

    def generate_html_report(self, parsed: Dict[str, Any], news_context: Dict[str, Any], market_context: Dict[str, Any]) -> tuple[str, dict]:
        metrics = parsed["metrics"]
        
        # ── 1. Resolve Tiered Data ──
        vals = {}
        tiers = {}
        soft_warnings = []
        
        core_keys = ["revenue", "cogs", "gross_profit", "opex", "op_inc", "net_inc", "cfo", "capex", "assets", "liabilities", "equity", "ar", "deferred_rev", "sbc"]
        for m_key in core_keys:
            val, tier, logs = self._resolve_tiered_metric(m_key, metrics.get(m_key, []))
            vals[m_key] = val
            tiers[m_key] = tier
            
        # ── 1x. Numerical Scale Validation (STEP 0) ──
        vals, scale_logs = self._normalize_scale(vals, tiers)
        soft_warnings.extend(scale_logs)
        # ── 1b. Infer missing values from margins ──
        if vals["revenue"] > 0:
            # Map margins: gross, operating, net
            margin_map = {m.lower(): v for m, v in zip(["gross", "operating", "net"], [0.0]*3)}
            for m_text in parsed.get("explicit_margins", []):
                # This is a bit simplistic, usually explicit_margins is just a list of floats
                # We need to know WHICH margin it is.
                pass                 
            # Re-scanning specifically for Gross/Op Margin to infer Profit/Inc
            text_low = parsed.get("text", "").lower()
            gm_match = re.search(r'(?i)gross\s+margin[s]?\s*(?:of|was|were|:)?\s*([+\-]?[\d.]+)\s*%', text_low)
            if gm_match and vals["gross_profit"] == 0:
                vals["gross_profit"] = vals["revenue"] * (float(gm_match.group(1)) / 100)
                tiers["gross_profit"] = 2 # Inferred tier
                
            om_match = re.search(r'(?i)operating\s+margin[s]?\s*(?:of|was|were|:)?\s*([+\-]?[\d.]+)\s*%', text_low)
            if om_match and vals["op_inc"] == 0:
                vals["op_inc"] = vals["revenue"] * (float(om_match.group(1)) / 100)
                tiers["op_inc"] = 2
                
            nm_match = re.search(r'(?i)net\s+margin[s]?\s*(?:of|was|were|:)?\s*([+\-]?[\d.]+)\s*%', text_low)
            if nm_match and vals["net_inc"] == 0:
                vals["net_inc"] = vals["revenue"] * (float(nm_match.group(1)) / 100)
                tiers["net_inc"] = 2
                
        # ── 1c. Infer Balance Sheet Identity ──
        if vals["assets"] == 0 and (vals["liabilities"] > 0 and vals["equity"] > 0):
            vals["assets"] = vals["liabilities"] + vals["equity"]
            tiers["assets"] = 2
        elif vals["liabilities"] == 0 and (vals["assets"] > 0 and vals["equity"] > 0):
            vals["liabilities"] = vals["assets"] - vals["equity"]
            tiers["liabilities"] = 2
        elif vals["equity"] == 0 and (vals["assets"] > 0 and vals["liabilities"] > 0):
            vals["equity"] = vals["assets"] - vals["liabilities"]
            tiers["equity"] = 2

        # ── 2. Numerical sanity gate (Layer 1–2: extraction + sanity before metrics) ──
        vals, parse_errors = apply_numerical_sanity_gate(vals)
        structural_parse: List[str] = []

        rev = vals["revenue"]
        if vals["assets"] > 0 and (vals["liabilities"] > 0 or vals["equity"] > 0):
            bs_diff = abs(vals["assets"] - (vals["liabilities"] + vals["equity"]))
            if bs_diff > (vals["assets"] * 0.005):
                structural_parse.append(
                    f"PARSE ERROR: Balance sheet identity mismatch (Assets vs Liab+Eq variance ${bs_diff:,.0f}).")
        if vals["revenue"] > 0 and vals["cogs"] > 0 and vals["gross_profit"] > 0:
            gp_calc = vals["revenue"] - vals["cogs"]
            if abs(vals["gross_profit"] - gp_calc) > (vals["revenue"] * 0.005):
                structural_parse.append(
                    f"PARSE ERROR: Gross profit ≠ Revenue − COGS (extraction/format issue).")

        parse_errors = parse_errors + structural_parse
        # FCF LOCK: always CFO − CapEx (never treat CFO as FCF)
        # FCF FALLBACK RULE: If CapEx is missing, FCF is UNKNOWN (not CFO)
        capex_found = tiers.get("capex", 0) > 0
        if capex_found:
            implied_fcf = _fcf_locked(vals["cfo"], vals["capex"])
        else:
            implied_fcf = None # UNKNOWN state
        real_financial_risks = classify_real_financial_risks(vals, implied_fcf, text=parsed.get("text", ""))
        if implied_fcf is None and vals.get("cfo", 0) != 0:
            real_financial_risks.append("REAL: Free Cash Flow UNKNOWN (CapEx not extracted).")
        
        # ── 3. EXTRACTION CONFIDENCE & FAIL-SAFE ──
        required_fields = ["revenue", "gross_profit", "op_inc", "net_inc", "cfo", "assets", "liabilities", "equity"]
        extracted_count = sum(1 for m in required_fields if tiers[m] > 0)
        extraction_pct = (extracted_count / len(required_fields)) * 100
        
        if extraction_pct > 80: confidence_label = "HIGH"
        elif extraction_pct >= 50: confidence_label = "MEDIUM"
        else: confidence_label = "LOW"        
        # FAIL-SAFE: If >50% of core metrics are missing: TERMINATE
        if extraction_pct < 50:
            return (
                f"**Analysis incomplete** — extraction coverage {extraction_pct:.0f}% (minimum 50% required).\n\n"
                "No scoring or allocation until core fields are present.",
                {
                    "total_score": 0,
                    "investment_score": 0,
                    "integrity_score": 0,
                    "red_flags": 99,
                    "implied_fcf": 0,
                    "core_confidence": "LOW",
                    "takeaway_label": "DATA FAILURE",
                    "extraction_pct": extraction_pct,
                    "parse_errors": [],
                    "real_financial_risks": [],
                    "has_hard_errors": True,
                },
)

        critical_missing = any(tiers[m] == 0 for m in ["revenue", "gross_profit", "cfo"])

        # ── 4. Scoring: parse errors reduce confidence / cap; real risks hit scores directly ──
        integrity_score = 100
        integrity_score -= len(parse_errors) * 8
        missing_essential = [m for m in required_fields if tiers[m] == 0]
        integrity_score -= len(missing_essential) * 5
        integrity_score = int(max(0, integrity_score))
        
        # Investment Score (0-100)
        rev = vals["revenue"]
        gp = vals["gross_profit"]
        oi = vals["op_inc"]
        ni = vals["net_inc"]
        fcf = implied_fcf
        
        rev_growth = self._safe_mean(parsed.get("growth_rates", []))
        gross_margin = (gp / rev * 100) if rev > 0 else 0.0
        op_margin = (oi / rev * 100) if rev > 0 else 0.0
        
        # Factor A: Growth (0-25) — DECISION ENFORCEMENT: High-precision multipliers
        score_growth = 10.0
        if rev > 0:
            # Multiplier 0.40125 and tiny epsilon from rev_growth to ensure differentiation
            score_growth = np.clip(15.0 + (rev_growth * 0.40125) + (abs(rev_growth) * 1e-6), 0, 25)
        if vals["deferred_rev"] > 0: score_growth = min(25.0, score_growth + 2.01)
        
        # Factor B: Profitability (0-25) — DECISION ENFORCEMENT
        score_prof = 10.0
        if rev > 0:
            # 0.10015 and 0.60015 multipliers to avoid collisions
            prof_base = (gross_margin * 0.10015) + (op_margin * 0.60015)
            score_prof = np.clip(10.0 + prof_base + (op_margin * 1e-6), 0, 25)
            
        # Factor C: Financial Health (0-25) — DECISION ENFORCEMENT
        score_health = 12.0
        if vals["cfo"] != 0:
            if implied_fcf is not None:
                fcf_margin = (implied_fcf / rev * 100) if rev > 0 else 5.0
                score_health = np.clip(15.0 + (fcf_margin * 0.50025) + (abs(fcf_margin) * 1e-6), 0, 25)
            else:
                score_health = 10.0 # UNKNOWN FCF penalty

        if critical_missing or confidence_label == "LOW":
            score_growth = min(8.0, score_growth)
            score_prof = min(8.0, score_prof)
            score_health = min(8.0, score_health)

        # Factor D: Risk (0-25) — DECISION ENFORCEMENT & REAL RISK PRIORITY
        score_risk = 25.0
        for r in real_financial_risks:
            if "Negative free cash flow" in r: score_risk -= 7.001
            if "Free Cash Flow UNKNOWN" in r: score_risk -= 5.001
            if "Stock-based compensation" in r:
                sbc_ratio = (vals.get("sbc", 0) / rev) if rev > 0 else 0
                score_risk -= (sbc_ratio * 40.00125)
            if "Accounts receivable" in r: score_risk -= 4.001
            if "leverage" in r.lower(): score_risk -= 4.001
        score_risk = max(0, min(25.0, score_risk))

        # Final summing with rounding to 1 decimal for ranking, then int for display
        raw_investment_score = score_growth + score_prof + score_health + score_risk
        investment_score = int(round(raw_investment_score))

        # PARSE ERRORS: cap composite, lower confidence — do not mirror real-risk penalties
        has_parse_errors = len(parse_errors) > 0
        if has_parse_errors:
            investment_score = min(70, investment_score)
            if confidence_label == "HIGH":
                confidence_label = "MEDIUM"
        has_hard_errors = has_parse_errors

        if critical_missing or confidence_label == "LOW":
            investment_score = min(40, investment_score)
            outlook_text_override = "INSUFFICIENT DATA — NEUTRAL / REVIEW"
        else:
            outlook_text_override = None
        
        # ── 4. Quant layer: KILL SWITCH — no price series → no quant block at all ──
        quant_signal = None
        q_bias = 0.0
        ticker = parsed["tickers_mentioned"][0] if parsed["tickers_mentioned"] else None
        quant_md = ""
        if ticker:
            try:
                df = get_stock(ticker, period="1y")
                if df is not None and not df.empty:
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    prices = close.dropna().astype(float).values
                    volumes = None
                    if "Volume"in df.columns:
                        v = df["Volume"]
                        if isinstance(v, pd.DataFrame):
                            v = v.iloc[:, 0]
                        v = v.dropna().astype(float)
                        if len(v) > 0 and v.sum() > 0:
                            volumes = v.values
                    if len(prices) >= 40:
                        quant_model = get_quant_ensemble()
                        quant_signal = quant_model.predict(prices, volumes)
            except Exception:
                quant_signal = None

        if quant_signal:
            q_bias = (quant_signal.probability - 0.5) * 100 * quant_signal.confidence
            
            # --- 13F Smart Money Flow Injection ---
            try:
                from sec_13f_engine import get_sec_13f_engine
                sec_engine = get_sec_13f_engine()
                global_flows = sec_engine.get_global_smart_money_flow()
                flow_map = global_flows.get("net_flow_map", {})
                net_flow = flow_map.get(ticker, 0)
                if net_flow > 0:
                    smart_money_signal = f"Strong Inflow (+${net_flow/1e6:.1f}M)"
                    q_bias += 2.0 # Slight bullish bias adjustment
                elif net_flow < 0:
                    smart_money_signal = f"Strong Outflow (-${abs(net_flow)/1e6:.1f}M)"
                    q_bias -= 2.0 # Slight bearish bias adjustment
                else:
                    smart_money_signal = "Neutral/Unchanged"
            except Exception:
                smart_money_signal = "Unavailable"
            quant_md = (
                "\n### 7 | Quant outlook (model-only)\n\n"
                f"| Direction | Probability | Confidence | Expected return | Smart Money (13F) |\n"
                f"| --- | --- | --- | --- | --- |\n"
                f"| {quant_signal.direction} | {quant_signal.probability:.1%} |"
                f"{quant_signal.confidence:.1%} | {quant_signal.expected_return:+.2%} | {smart_money_signal} |\n\n"
                f"Rationale: {'· '.join(quant_signal.reasoning[:3])}\n")

        # ── 5. Sentiment & Scenarios — MANDATORY QUANT DELETE
        bull_prob = 33; base_prob = 34; bear_prob = 33
        if quant_signal: # Only calculate probabilities if quant data is present
            sent_pos = parsed["sentiment_cues"]["positive"]
            sent_neg = parsed["sentiment_cues"]["negative"]
            if (sent_pos + sent_neg) > 0:
                net_sent = (sent_pos - sent_neg) / (sent_pos + sent_neg)
                if net_sent > 0.3: bull_prob += 15; bear_prob -= 10
                elif net_sent < -0.3: bear_prob += 20; bull_prob -= 15
            
            if bear_prob > 60: net_sent = 0.0 # Sentiment Decoupling
            
            bull_prob = int(max(5, min(bull_prob, 85)))
            bear_prob = int(max(5, min(bear_prob, 85)))
            base_prob = 100 - bull_prob - bear_prob

        # Full model score
        if quant_signal:
            full_score = int(np.clip(investment_score + q_bias, 0, 100))
        else:
            full_score = investment_score

        if has_parse_errors:
            full_score = min(full_score, 70)
        block_conviction = has_parse_errors or len(real_financial_risks) > 0

        c_success = COLORS["success"]
        c_danger = COLORS["danger"]
        c_gold = COLORS["gold"]
        c_neutral = COLORS["text_secondary"]
        c_border = COLORS["border"]
        c_card_bg = COLORS["navy_light"]
        c_card_bg_soft = COLORS["navy_mid"]

        if outlook_text_override:
            outlook_text = outlook_text_override
            outlook_color = c_danger
        elif block_conviction:
            if full_score >= 55:
                outlook_text = "ACCUMULATE (capped — parse/real-risk guard)"
                outlook_color = c_success
            elif full_score >= 40:
                outlook_text = "NEUTRAL / HOLD"
                outlook_color = c_gold
            else:
                outlook_text = "NEUTRAL / RISKY"
                outlook_color = c_danger
        elif full_score >= 75:
            outlook_text = "STRONG OVERWEIGHT"
            outlook_color = c_success
        elif full_score >= 55:
            outlook_text = "ACCUMULATE"
            outlook_color = c_success
        elif full_score >= 40:
            outlook_text = "NEUTRAL / HOLD"
            outlook_color = c_gold
        else:
            outlook_text = "UNDERWEIGHT / REDUCE"
            outlook_color = c_danger

        if block_conviction:
            outlook_text = outlook_text.replace("STRONG OVERWEIGHT", "NEUTRAL / HOLD")
            outlook_text = outlook_text.replace("STRONG ", "")
            if outlook_color == c_success and "ACCUMULATE"not in outlook_text:
                outlook_color = c_gold

        def fmt(v: float, t: int) -> str:
            if t == 0:
                return "N/A"
            if abs(v) >= 1e9:
                return f"${v/1e9:.2f}B"
            if abs(v) >= 1e6:
                return f"${v/1e6:.2f}M"
            return f"${v:,.0f}"
        # Score justifications (real data only)
        j_growth = (
            f"Growth score reflects extracted/implicit revenue growth signals (~{rev_growth:.1f}% from document)."
            if rev > 0
            else "Growth score limited — revenue not established from extraction.")
        j_prof = (
            f"Profitability from gross {gross_margin:.1f}% and operating {op_margin:.1f}% vs revenue."
            if rev > 0
            else "Profitability neutral — insufficient revenue line.")
        j_health = (
            f"Health driven by FCF = CFO − CapEx = {fmt(implied_fcf, tiers['cfo'])} (recomputed; negative FCF capped health)."
            if tiers.get("cfo", 0) > 0
            else "Health limited — CFO not extracted.")
        j_risk = (
            "Risk from verified operating/financing stressors: "+ "; ".join(real_financial_risks)
            if real_financial_risks
            else "No major real-risk flags beyond parse checks.")

        # ── HTML THEME BUILDER (Institutional Polish) ──
        def conf_badge(tier: int) -> str:
            cols = {1: c_success, 2: c_gold, 3: c_neutral, 0: c_danger}
            lbls = {1: "T1", 2: "T2", 3: "T3", 0: "—"}
            c = cols.get(tier, c_danger)
            l = lbls.get(tier, "—")
            return f'<span class="da-badge"style="background:{c}1a; color:{c}; border:1px solid {c}33;">{l}</span>'
        # Build Quant Section separately
        if quant_signal:
            q_color = c_success if quant_signal.direction == "BULLISH"else c_danger if quant_signal.direction == "BEARISH"else c_gold
            quant_html = f'''
            <div class="da-section">
                <div class="da-header">Quant Intelligence (Engine Output)</div>
                <div class="da-grid"style="grid-template-columns: repeat(4, 1fr);">
                    <div class="da-card"><div class="da-card-label">Direction</div><div class="da-card-val"style="color:{q_color};">{quant_signal.direction}</div></div>
                    <div class="da-card"><div class="da-card-label">Probability</div><div class="da-card-val">{quant_signal.probability:.1%}</div></div>
                    <div class="da-card"><div class="da-card-label">Confidence</div><div class="da-card-val">{quant_signal.confidence:.1%}</div></div>
                    <div class="da-card"><div class="da-card-label">Exp. Return</div><div class="da-card-val"style="color:{q_color};">{quant_signal.expected_return:+.2%}</div></div>
                </div>
                <div style="background:{c_card_bg_soft}; border:1px solid {c_border}; border-top:none; padding:12px; border-bottom-left-radius:6px; border-bottom-right-radius:6px; font-size:0.8rem; color:{c_neutral}; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                <b>RATIONALE:</b> {"• ".join(quant_signal.reasoning[:3])}
                </div>
            </div>
            '''
        else:
            quant_html = ""
        html = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=JetBrains+Mono:wght@500;800&display=swap');

.da-container {{
    font-family: 'Inter', -apple-system, sans-serif;
    color: {COLORS["text_primary"]};
    padding: 10px;
}}

.da-section {{
    margin-bottom: 35px;     animation: da-fadeIn 0.6s ease-out;
}}

@keyframes da-fadeIn {{
    from {{opacity: 0; transform: translateY(8px); }}
    to {{opacity: 1; transform: translateY(0); }}
}}

.da-header {{
    color: {c_neutral};     border-bottom: 1px solid {c_border};     padding-bottom: 10px;     margin-bottom: 18px;     font-family: 'JetBrains Mono', monospace;     font-size: 0.85rem;     font-weight: 800;     text-transform: uppercase;     letter-spacing: 1.5px;
}}

.da-grid {{
    display: grid;     grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));     gap: 15px; }}

.da-card {{
    background: linear-gradient(145deg, {c_card_bg_soft}, {c_card_bg});     border: 1px solid {c_border};     border-radius: 6px;     padding: 18px;     box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}}

.da-card::after {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0) 100%);
    pointer-events: none;
}}

.da-card-label {{
    color: {c_neutral};     font-size: 0.7rem;     font-weight: 700;     margin-bottom: 10px;     text-transform: uppercase;     letter-spacing: 0.8px; }}

.da-card-val {{
    color: {COLORS["text_primary"]};     font-size: 1.5rem;     font-weight: 800;     font-family: 'JetBrains Mono', monospace;     margin-bottom: 6px; }}

.da-badge {{
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 0.6rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-left: 6px;
    display: inline-block;
    vertical-align: middle;
}}

.da-integrity-container {{
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: 5px;
}}

.da-integrity-bar {{
    flex: 1;
    height: 4px;
    background: #21262d;
    border-radius: 2px;
    overflow: hidden;
}}

.da-integrity-fill {{
    height: 100%;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
}}

.da-stat-row {{
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.85rem;
}}

.da-stat-row:last-child {{border-bottom: none; }}

.da-error-box {{
    background: rgba(239, 83, 80, 0.08);
    border: 1px solid rgba(239, 83, 80, 0.25);
    border-left: 3px solid {c_danger};
    padding: 12px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: {COLORS["text_primary"]};
    line-height: 1.5;
    margin-bottom: 10px;
}}

.da-warning-box {{
    background: rgba(201, 168, 76, 0.08);
    border: 1px solid rgba(201, 168, 76, 0.25);
    border-left: 3px solid {c_gold};
    padding: 12px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: {COLORS["text_primary"]};
    line-height: 1.5;
    margin-bottom: 10px;
}}
</style>

<div class="da-container">
    <div class="da-section">
        <div class="da-header">1 | Financial Synopsis (Validated Signal)</div>
        <div class="da-grid">
            <div class="da-card">
                <div class="da-card-label">Revenue {conf_badge(tiers['revenue'])}</div>
                <div class="da-card-val">{fmt(vals['revenue'], tiers['revenue'])}</div>
            </div>
            <div class="da-card">
                <div class="da-card-label">Gross Margin {conf_badge(tiers['gross_profit'])}</div>
                <div class="da-card-val">{f"{gross_margin:.1f}%"if gp != 0 else 'N/A'}</div>
            </div>
            <div class="da-card">
                <div class="da-card-label">Operating Margin {conf_badge(tiers['op_inc'])}</div>
                <div class="da-card-val">{f"{op_margin:.1f}%"if oi != 0 else 'N/A'}</div>
            </div>
            <div class="da-card">
                <div class="da-card-label">Recomputed Free Cash Flow {conf_badge(tiers['cfo'])}</div>
                <div class="da-card-val"style="color:{c_success if implied_fcf is not None and implied_fcf >= 0 else c_danger if implied_fcf is not None else c_neutral};">
                    {fmt(implied_fcf, tiers['cfo']) if implied_fcf is not None else "UNKNOWN"}
                </div>
                <div style="font-size:0.7rem; color:{c_neutral}; margin-top:4px;">
                    {f"CFO ({fmt(vals['cfo'], tiers['cfo'])}) − CapEx ({fmt(vals['capex'], tiers['capex'])})"if capex_found else "MANDATORY CapEx scan failed to resolve value."}
                </div>
            </div>
        </div>
    </div>

    <div class="da-section">
        <div class="da-header">2 | Integrity & Validation Stream</div>
        <div class="da-grid"style="grid-template-columns: 1.5fr 1fr;">
            <div class="da-card"style="border-top: 2px solid {c_gold};">
                <div class="da-card-label">Data Integrity Score</div>
                <div class="da-integrity-container">
                    <div class="da-card-val"style="margin:0;">{integrity_score}<span style="font-size:0.9rem; color:{c_neutral}; margin-left:4px;">/100</span></div>
                    <div class="da-integrity-bar">
                        <div class="da-integrity-fill"style="width:{integrity_score}%; background: linear-gradient(90deg, {c_gold}, {c_success});"></div>
                    </div>
                </div>
                <div style="font-size:0.75rem; color:{c_neutral}; margin-top:10px;">Deterministic extraction mapped at <b>{confidence_label}</b> confidence ({extraction_pct:.0f}% field coverage).</div>
            </div>
            <div class="da-card">
                <div class="da-card-label">Parse Identity</div>
                <div style="margin-top:5px;">
                    {"".join(["<div class='da-error-box'>"+ p + "</div>"for p in parse_errors]) if parse_errors else f"<div style='color:{c_success}; font-size:0.8rem; font-weight:700;'>DETERMINISTIC IDENTITY CLEAR</div>"}
                </div>
            </div>
        </div>
    </div>

    <div class="da-section">
        <div class="da-header">{"3 | Probabilistic Scenarios & Risk"if quant_signal else "3 | Structural Risk Analysis"}</div>
        <div class="da-grid"style="grid-template-columns: {'1fr 1fr'if quant_signal else '1fr'};">
            {f"""
            <div class="da-card">
                <div class="da-card-label">Core Forecast Map</div>
                <div class="da-stat-row"><span style="color:{c_success}; font-weight:700;">Bull Case</span><span class="da-card-val"style="font-size:1.1rem; margin:0; color:{c_success};">{bull_prob}%</span></div>
                <div class="da-stat-row"><span style="color:{c_gold}; font-weight:700;">Base Case</span><span class="da-card-val"style="font-size:1.1rem; margin:0; color:{c_gold};">{base_prob}%</span></div>
                <div class="da-stat-row"><span style="color:{c_danger}; font-weight:700;">Bear Case</span><span class="da-card-val"style="font-size:1.1rem; margin:0; color:{c_danger};">{bear_prob}%</span></div>
            </div>
            """if quant_signal else ""}
            <div class="da-card">
                <div class="da-card-label">Structural Stressors</div>
                <div style="margin-top:5px;">
                    {"".join(["<div class='da-warning-box'>• "+ r + "</div>"for r in real_financial_risks]) if real_financial_risks else f"<div style='color:{c_neutral}; font-size:0.8rem;'>No idiosyncratic structural risks detected beyond parse variance.</div>"}
                </div>
            </div>
        </div>
    </div>

    {quant_html}

    <div class="da-section">
        <div class="da-header">4 | Ensemble Conviction Verdict</div>
        <div class="da-card"style="background: linear-gradient(135deg, {c_card_bg_soft} 0%, {c_card_bg} 100%); border-left: 5px solid {outlook_color}; box-shadow: 0 0 30px {outlook_color}1a;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div class="da-card-label"style="color:{outlook_color}; font-size:0.8rem;">Unified Multi-Agent Conviction</div>
                    <div style="font-size:2.2rem; font-weight:900; color:{outlook_color}; letter-spacing:-1px; line-height:1; margin-top:5px;">{outlook_text}</div>
                     <div style="font-size:0.8rem; color:{c_neutral}; margin-top:15px; max-width:500px; line-height:1.4;">
                        Integrated scorecard factoring fundamental baseline conviction ({investment_score}){f"and quant ensemble bias ({q_bias:+.1f} pts)."if quant_signal else "."}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="da-card-label">Integrated Score</div>
                    <div style="font-size:3.5rem; font-weight:900; color:{outlook_color}; line-height:1;">{full_score}<span style="font-size:1.2rem; color:{c_neutral}; margin-left:4px;">/100</span></div>
                </div>
            </div>
        </div>
    </div>
</div>
"""
        stats_payload = {
            "total_score": full_score,
            "investment_score": investment_score,
            "integrity_score": integrity_score,
            "red_flags": len(parse_errors) + len(real_financial_risks),
            "parse_errors": parse_errors,
            "real_financial_risks": real_financial_risks,
            "implied_fcf": implied_fcf,
            "core_confidence": confidence_label,
            "takeaway_label": outlook_text,
            "extraction_pct": extraction_pct,
            "has_hard_errors": has_parse_errors,
            "margin": op_margin,
            "fcf_margin": (implied_fcf / rev * 100) if (rev > 0 and implied_fcf is not None) else 0.0,
            "justifications": {
                "growth": j_growth,
                "profitability": j_prof,
                "financial_health": j_health,
                "risk": j_risk,
            },
}

        return html, stats_payload


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PORTFOLIO ALLOCATION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PortfolioAllocationEngine:
    """Relative capital allocation: $100M across imperfect names — never all $0 unless no entities."""
    def allocate(self, company_stats: Dict[str, dict]) -> str:
        total_capital = 100_000_000
        entities: List[Dict[str, Any]] = []

        for name, stats in company_stats.items():
            score = float(stats.get("total_score", 0))
            integrity = float(stats.get("integrity_score", 0))
            red_ct = int(stats.get("red_flags", 0))
            implied_fcf = float(stats.get("implied_fcf", 0))
            confidence = stats.get("core_confidence", "MEDIUM")
            parse_ct = len(stats.get("parse_errors", []) or [])
            real_ct = len(stats.get("real_financial_risks", []) or [])

            if integrity < 50:
                score *= 0.7
            elif integrity < 80:
                score *= 0.9

            has_parse = bool(stats.get("has_hard_errors", False))
            if score >= 70 and red_ct <= 2 and implied_fcf >= 0 and not has_parse:
                classification = "CORE POSITION"
            elif score >= 45:
                classification = "SATELLITE POSITION"
            elif score > 0:
                classification = "SPECULATIVE POSITION"
            else:
                classification = "RELATIVE LAGGARD"
            qual_factor = max(0.15, 1.0 - min(red_ct, 8) * 0.08 - parse_ct * 0.05)
            if implied_fcf < 0:
                qual_factor *= 0.65
            conf_factor = 0.55 if confidence == "LOW"else 1.0
            raw_weight = max(score, 0.01) * qual_factor * conf_factor

            entities.append({
                "name": name,
                "score": score,
                "class": classification,
                "raw_weight": raw_weight,
                "takeaway": stats.get("takeaway_label", ""),
                "integrity": integrity,
                "real_risk_ct": real_ct,
            })

        entities.sort(key=lambda x: x["score"], reverse=True)
        total_weight = sum(e["raw_weight"] for e in entities)

        n = len(entities)
        if n == 0:
            return "### Portfolio\n\nNo companies to allocate.\n"
        if total_weight <= 0:
            if n == 1:
                shares = [1.0]
            elif n == 2:
                shares = [0.6, 0.4]
            elif n >= 3:
                shares = [0.5, 0.3, 0.2] + [0.0] * (n - 3)
            else:
                shares = [1.0 / n] * n
        else:
            shares = [e["raw_weight"] / total_weight for e in entities]

        rows_html = []
        for i, e in enumerate(entities):
            cap = shares[i] * total_capital
            e["capital"] = cap
            e["share"] = shares[i] * 100
            
            # Allocation specific colors
            c_color = COLORS["success"] if e["class"] == "CORE POSITION"else COLORS["gold"] if e["class"] == "SATELLITE POSITION"else COLORS["text_secondary"]
            if e["class"] == "AVOID": c_color = COLORS["danger"]

            rows_html.append(f'''
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid {COLORS['border']}; padding:15px 0;">
                <div style="flex:1;">
                    <span style="color:{COLORS['text_secondary']}; font-family:'JetBrains Mono', monospace; font-weight:800; margin-right:15px;">#{i+1}</span>
                    <span style="color:{COLORS['text_primary']}; font-weight:800; font-size:1.1rem; letter-spacing:-0.2px;">{e['name']}</span>
                    <div style="font-size:0.75rem; color:{c_color}; font-weight:900; margin-top:4px; text-transform:uppercase; letter-spacing:1px;">{e['class']} (Score: {e['score']:.1f})</div>
                </div>
                <div style="flex:1; text-align:right;">
                    <div style="color:{COLORS['text_primary']}; font-weight:900; font-size:1.5rem; font-family:'JetBrains Mono', monospace;">${cap:,.0f}</div>
                    <div style="font-size:0.8rem; color:{COLORS['text_secondary']}; margin-top:2px;">{e['share']:.1f}% Allocation</div>
                </div>
            </div>
            ''')

        top = entities[0]
        html = f"""<div class="da-container"style="margin-top:20px;">
    <div class="da-section">
        <div class="da-header">System-Wide Strategy: $100M Quantitative Deployment</div>
        <div class="da-card"style="background: linear-gradient(145deg, {COLORS['navy_mid']}, {COLORS['navy_light']}); border-color:{COLORS['border']};">
            <div style="font-size:0.85rem; color:{COLORS['text_secondary']}; margin-bottom:20px; font-style:italic; border-left:2px solid {COLORS['text_secondary']}; padding-left:10px;">
                Relative mandate: <b>{top['name']}</b> receives the largest share among parsed names.                 Allocations are comparative — even imperfect sets receive capital weighted by adjusted quality-score.
            </div>
            {"".join(rows_html)}
            <div style="margin-top:20px; font-size:0.75rem; color:{COLORS['text_secondary']}; text-align:center; opacity:0.75;">
                *Parse-heavy names receive lower weights via structural quality factor; the book automatically sums to $100M.*
            </div>
        </div>
    </div>
</div>
"""
        return html
# EXTERNAL CONTEXT FETCHERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_news_context(tickers: List[str]) -> Dict[str, Any]:
    context: Dict[str, Any] = {"sentiment": 0, "articles": [], "tickers": tickers}
    if not tickers: return context
    try:
        from news_analysis_engine import get_news_engine
        engine = get_news_engine()
        sentiments = []
        for t in tickers[:3]:
            try:
                res = engine.analyze_symbol_news(t)
                if res and "sentiment_score"in res:
                    sentiments.append(res.get("sentiment_score", 0))
            except Exception: pass
        if sentiments:
            context["sentiment"] = float(np.mean(sentiments))
    except Exception: pass
    return context

def _fetch_market_context() -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    try:
        from market_heartbeat_system import fetch_heartbeat_data, generate_market_heartbeat
        g, u = fetch_heartbeat_data()
        if g and u:
            hb = generate_market_heartbeat(g, u)
            ctx["vol_regime"] = hb.get("vol", ("Unknown", ""))[0]
            ctx["risk_mode"] = hb.get("env_class", ("Unknown", ""))[0]
    except Exception: pass
    
    if not ctx:
        try:
            from regime import volatility_regime, risk_on_off
            ctx["vol_regime"] = volatility_regime()
            ctx["risk_mode"] = risk_on_off()
        except Exception: pass
    return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STREAMLIT UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_document_analyzer():
    """Render the Document Analyzer page."""
    st.title("Document Analyzer (Multi-Company Integrity Engine)")
    st.caption("A multi-agent validation engine tracking metric confidence, generating deterministic modeling, and allocating capital across multiple operations.")

    st.markdown("---")
    upload_col, text_col = st.columns(2)

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Document (TXT, CSV, PDF)",
            type=["txt", "csv", "pdf"],
            key="doc_analyzer_upload",
)

    with text_col:
        pasted_text = st.text_area(
            "Paste documents. (Use 'COMPANY: NAME'or '---'to split multiple entities)",
            height=150,
            placeholder="COMPANY: AAPL\n[data here]\n---\nCOMPANY: TSLA\n[data here]",
            key="doc_analyzer_paste",
)

    analyze_btn = st.button("Execute Integrity Validation & Intelligence Report", type="primary", key="doc_analyze_btn", width="stretch")

    if not analyze_btn:
        return

    # Parse Input
    raw_text = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                raw_text = "\n".join([page.extract_text() or ""for page in reader.pages])
            except ImportError:
                st.warning("PyPDF2 not installed — paste text instead.")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        else:
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

    raw_text += ("\n"+ pasted_text) if pasted_text else ""
    if not raw_text.strip():
        st.warning("No text to analyze. Upload a file or paste text.")
        return

    # Pipeline Execution
    with st.spinner("Executing Strict Data Validation and Confidence Scoring Protocols..."):
        parser = FinancialDocumentParser()
        engine = DocumentIntelligenceEngine()
        
        # Step 0: Multi-Company Segmentation
        parsed_segments = parser.parse(raw_text)
        
        all_htmls = []
        company_stats = {}
        
        for parsed in parsed_segments:
            entity_name = parsed["entity_name"]
            tickers = parsed["tickers_mentioned"]
            
            news_ctx = _fetch_news_context(tickers)
            market_ctx = _fetch_market_context()

            # Process EACH company independently (Steps 1–12)
            comp_html, stats = engine.generate_html_report(parsed, news_ctx, market_context=market_ctx)
            
            clean_name = entity_name if entity_name != "Primary Entity"else (tickers[0] if tickers else "Analysis")
            all_htmls.append((clean_name, comp_html))
            company_stats[clean_name] = stats

    # Render themed and organized reports
    st.markdown("---")
    st.subheader("Analysis Results")
    if len(all_htmls) == 1:
        st.html(all_htmls[0][1])
    else:
        for name, report_html in all_htmls:
            with st.expander(f"Company Analysis: {name}", expanded=False):
                st.html(report_html)

        if len(all_htmls) > 1:
            # Enforce Score Spread Rule (+10 point gaps)
            company_stats = enforce_score_spreads(company_stats)
            
            alloc_engine = PortfolioAllocationEngine()
            portfolio_html = alloc_engine.allocate(company_stats)
            st.subheader("Portfolio Construction")
            st.html(portfolio_html)

    st.caption(f"\n\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Computed via Octavian Data Integrity Engine (Confidence Verification ACTIVE)*")
