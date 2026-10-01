import os
import csv
import yaml
import threading
from datetime import datetime

class BudgetExceededError(Exception):
    pass

class Ledger:
    def __init__(self, ledger_path="runs/ledger.csv", budget_config_path="config/budget.yaml"):
        self.ledger_path = ledger_path
        self.budget_config_path = budget_config_path
        self.lock = threading.Lock()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "provider", "model", "tokens", "paid_usd", "official_usd", "latency_ms"])
                
        self.config = self._load_config()
        self._total_paid = 0.0
        self._model_spent = {}
        self._init_totals()

    def _init_totals(self):
        with self.lock:
            if os.path.exists(self.ledger_path):
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        paid = float(row.get("paid_usd", 0.0) or 0.0)
                        model = row.get("model", "unknown")
                        self._total_paid += paid
                        self._model_spent[model] = self._model_spent.get(model, 0.0) + paid

    def _load_config(self):
        if os.path.exists(self.budget_config_path):
            with open(self.budget_config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_total_spent(self):
        with self.lock:
            return self._total_paid, dict(self._model_spent)

    def check(self, provider: str, model: str, estimated_cost: float = 0.0):
        with self.lock:
            total_paid = self._total_paid
            spent_for_model = self._model_spent.get(model, 0.0)

        hard_limit = self.config.get("total_usd_hard_limit", 250.0)
        warn_limit = self.config.get("total_usd_warn_limit", 200.0)
        
        if total_paid + estimated_cost >= hard_limit:
            raise BudgetExceededError(f"总花费已达硬上限 ${hard_limit:.2f}! 当前已花费: ${total_paid:.4f}")
            
        limits = self.config.get("limits", {}).get(provider, {})
        model_limit = limits.get(model)
        if model_limit is not None:
            if spent_for_model + estimated_cost >= model_limit:
                raise BudgetExceededError(f"模型 {model} 已达花费上限 ${model_limit:.2f}! 当前花费: ${spent_for_model:.4f}")

    def add(self, provider: str, model: str, tokens: int, paid: float, official: float, latency_ms: float = 0.0):
        with self.lock:
            self._total_paid += paid
            self._model_spent[model] = self._model_spent.get(model, 0.0) + paid
            with open(self.ledger_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat() + "Z",
                    provider,
                    model,
                    tokens,
                    f"{paid:.8f}",
                    f"{official:.8f}",
                    f"{latency_ms:.1f}"
                ])

    def summary(self):
        total_paid, model_spent = self.get_total_spent()
        total_tokens = 0
        with self.lock:
            if os.path.exists(self.ledger_path):
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        total_tokens += int(row.get("tokens", 0) or 0)
        return {
            "total_paid": total_paid,
            "model_spent": model_spent,
            "total_tokens": total_tokens,
            "budget_limit": self.config.get("total_usd_hard_limit", 40.0),
            "warn_limit": self.config.get("total_usd_warn_limit", 32.0),
        }
