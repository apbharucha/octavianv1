import json
import os
from typing import List, Dict, Any
from datetime import datetime
import uuid

class ManualPortfolioSystem:
    def __init__(self, filepath="manual_portfolios.json"):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({"portfolios": {}}, f)

    def _load(self):
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {"portfolios": {}}

    def _save(self, data):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)

    def create_portfolio(self, name: str, goal: str, risk_level: str, timeframe: str = "1 Year", target_growth: float = 10.0) -> str:
        data = self._load()
        pid = str(uuid.uuid4())
        data["portfolios"][pid] = {
            "name": name,
            "goal": goal,
            "risk_level": risk_level,
            "timeframe": timeframe,
            "target_growth": target_growth,
            "positions": [],
            "history": [],
            "created_at": datetime.now().isoformat()
        }
        self._save(data)
        return pid

    def list_portfolios(self) -> Dict[str, Any]:
        ports = self._load().get("portfolios", {})
        for p in ports.values():
            p.setdefault("timeframe", "1 Year")
            p.setdefault("target_growth", 10.0)
        return ports
        
    def get_portfolio(self, pid: str) -> Dict[str, Any]:
        p = self._load().get("portfolios", {}).get(pid)
        if p:
            p.setdefault("timeframe", "1 Year")
            p.setdefault("target_growth", 10.0)
        return p

    def delete_portfolio(self, pid: str):
        data = self._load()
        if pid in data["portfolios"]:
            del data["portfolios"][pid]
            self._save(data)
            return True
        return False

    def add_position(self, pid: str, symbol: str, quantity: float, entry_price: float, asset_type: str):
        data = self._load()
        if pid in data["portfolios"]:
            pos_id = str(uuid.uuid4())
            data["portfolios"][pid]["positions"].append({
                "id": pos_id,
                "symbol": symbol.upper(),
                "quantity": quantity,
                "entry_price": entry_price,
                "asset_type": asset_type.lower(),
                "entry_time": datetime.now().isoformat()
            })
            self._save(data)
            return pos_id
        return None

    def delete_position(self, pid: str, pos_id: str):
        # Simply deletes a row without closing it (e.g. mistake entry)
        data = self._load()
        if pid in data["portfolios"]:
            port = data["portfolios"][pid]
            for i, pos in enumerate(port["positions"]):
                if pos["id"] == pos_id:
                    port["positions"].pop(i)
                    self._save(data)
                    return True
        return False

    def close_position(self, pid: str, pos_id: str, exit_price: float):
        data = self._load()
        if pid in data["portfolios"]:
            port = data["portfolios"][pid]
            for i, pos in enumerate(port["positions"]):
                if pos["id"] == pos_id:
                    port["positions"].pop(i)
                    
                    entry_time = datetime.fromisoformat(pos["entry_time"])
                    exit_time = datetime.now()
                    duration_days = max(0, (exit_time - entry_time).days)
                    pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    
                    port["history"].append({
                        "symbol": pos["symbol"],
                        "action": "CLOSE",
                        "quantity": pos["quantity"],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "duration_days": duration_days,
                        "entry_time": pos["entry_time"],
                        "exit_time": exit_time.isoformat()
                    })
                    self._save(data)
                    return True
        return False
        
_mps_instance = None
def get_manual_portfolio_system() -> ManualPortfolioSystem:
    global _mps_instance
    if _mps_instance is None:
        _mps_instance = ManualPortfolioSystem()
    return _mps_instance
