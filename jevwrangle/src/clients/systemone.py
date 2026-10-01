import os
import json
import time
import hashlib
import sqlite3
import threading
from typing import Dict, Any, Tuple, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from src.clients.ledger import Ledger

URL = "https://aihubmix.com/v1/systemone"

def _retryable(e):
    return (
        isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (402, 429, 500, 502, 503, 504)
        or isinstance(e, (httpx.TimeoutException, httpx.TransportError))
    )

class SystemOneClient:
    def __init__(
        self,
        model: str = "jev-1.13",
        cache_path: str = "cache/calls_v2.sqlite",
        price_in_per_mtok: float = 0.0462,
        official_in_per_mtok: float = 0.042,
        ledger: Optional[Ledger] = None,
        timeout: float = 60.0
    ):
        self.model = model
        self.price = price_in_per_mtok
        self.official = official_in_per_mtok
        self.ledger = ledger or Ledger()
        self.api_key = os.environ.get("AIHUBMIX_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("环境变量 AIHUBMIX_API_KEY 未设置！")
            
        self.http = httpx.Client(
            timeout=timeout,
            trust_env=False,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.local = threading.local()
        self._write_lock = threading.Lock()



    def ask(self, state, questions, bypass_cache=False, context=None):
        from pathlib import Path
        from src.clients.evidence import EvidenceTransport
        transport = EvidenceTransport(self.http, URL, self.ledger, self.cache_path,
            Path(self.ledger.ledger_path).parent / 'api_events.jsonl',
            os.environ.get('JEV_RUN_ID', 'unassigned'), self.price, official_input=self.official)
        return transport.request({'model': self.model, 'state': state, 'questions': questions}, bypass_cache, context)
