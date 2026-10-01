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

URL_BASE = "https://aihubmix.com/v1/chat/completions"

def _retryable(e):
    return (
        isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (402, 429, 500, 502, 503, 504)
        or isinstance(e, (httpx.TimeoutException, httpx.TransportError))
    )

class LLMClient:
    def __init__(
        self,
        model: str = "gpt-6-luna",
        cache_path: str = "cache/calls_v2.sqlite",
        prices_config: Optional[Dict[str, Any]] = None,
        ledger: Optional[Ledger] = None,
        timeout: float = 60.0,
        endpoint: Optional[str] = None
    ):
        self.model = model
        self.endpoint = endpoint or URL_BASE
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

        # 默认单价 (input, output) per Mtoken
        if prices_config and model in prices_config:
            p = prices_config[model]
            self.input_price = p.get("input", 0.1)
            self.output_price = p.get("output", 0.5)
        elif "deepseek" in model:
            self.input_price = 0.142
            self.output_price = 0.284
        else: # gpt-6-luna
            self.input_price = 0.1
            self.output_price = 0.5



    def ask(self, messages, max_tokens=500, bypass_cache=False, reasoning_effort=None, context=None):
        from pathlib import Path
        from src.clients.evidence import EvidenceTransport
        payload = {'model': self.model, 'messages': messages, 'temperature': 0.0,
                   'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}}
        if 'luna' in self.model:
            payload['reasoning_effort'] = reasoning_effort or 'low'
            payload['max_tokens'] = max(600, payload['max_tokens'])
        transport = EvidenceTransport(self.http, self.endpoint, self.ledger, self.cache_path,
            Path(self.ledger.ledger_path).parent / 'api_events.jsonl',
            os.environ.get('JEV_RUN_ID', 'unassigned'), self.input_price, self.output_price)
        response, meta = transport.request(payload, bypass_cache, context)
        if response.get('choices', [{}])[0].get('finish_reason') == 'length':
            if payload['max_tokens'] >= 1200:
                raise ValueError('Truncated response; original response and usage retained in api_events.jsonl')
            payload['max_tokens'] = 1200
            first = meta
            response, meta = transport.request(payload, bypass_cache, context)
            meta['paid_usd'] = meta.get('paid_usd', 0) + first.get('paid_usd', 0)
            meta['latency_ms'] = meta.get('latency_ms', 0) + first.get('latency_ms', 0)
            meta['prior_request_id'] = first['request_id']
            if response.get('choices', [{}])[0].get('finish_reason') == 'length':
                raise ValueError('Still truncated at 1200; both responses retained and accounted for')
        return response, meta
