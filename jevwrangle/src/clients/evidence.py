"""Order-sensitive cache and append-only evidence for every HTTP attempt.

No Authorization headers are persisted. Legacy cache remains untouched.
"""
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
import httpx

_LOCK = threading.RLock()


class EvidenceTransport:
    def __init__(self, http, endpoint, ledger, cache_path, journal_path, run_id,
                 input_price, output_price=0, official_input=None):
        self.http, self.endpoint, self.ledger = http, endpoint, ledger
        self.cache_path, self.journal_path = Path(cache_path), Path(journal_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id=run_id
        self.input_price,self.output_price=input_price,output_price
        self.official_input=input_price if official_input is None else official_input
        with self.database() as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls(key TEXT PRIMARY KEY, response TEXT, created_at TEXT)")

    @contextmanager
    def database(self):
        db=sqlite3.connect(self.cache_path,timeout=60)
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, event):
        with _LOCK, self.journal_path.open("a",encoding="utf8") as f:
            f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),run_id=self.run_id,**event),ensure_ascii=False)+"\n")
            f.flush()

    def request(self, payload, bypass_cache=False, context=None):
        # Hash the actual transmitted body; dictionary order is an experimental variable.
        body=json.dumps(payload,ensure_ascii=False,separators=(",",":"))
        key=hashlib.sha256((self.endpoint+"\n"+body).encode()).hexdigest()
        request_id=str(uuid.uuid4())
        common=dict(request_id=request_id,cache_key=key,requested_model=payload["model"],context=context or {})
        if not bypass_cache:
            with self.database() as db:
                row=db.execute("SELECT response FROM calls WHERE key=?",(key,)).fetchone()
            if row:
                response=json.loads(row[0])
                self.record(dict(**common,event="cache_hit",response_model=response.get("model")))
                return response,dict(cache_hit=True,key=key,request_id=request_id,response_model=response.get("model"))
        self.record(dict(**common,event="request",endpoint=self.endpoint,payload=payload,bypass_cache=bypass_cache))
        for attempt in range(1,6):
            # Reserve a conservative upper estimate for this individual attempt.
            estimated=(max(1500,len(body))*self.input_price+payload.get("max_tokens",0)*self.output_price)/1e6
            self.ledger.check(provider="aihubmix",model=payload["model"],estimated_cost=estimated)
            started=time.perf_counter()
            try:
                r=self.http.post(self.endpoint,content=body.encode("utf8"))
            except (httpx.TimeoutException,httpx.TransportError) as e:
                not_sent=isinstance(e,(httpx.ConnectError,httpx.ConnectTimeout))
                self.record(dict(**common,event="transport_error",attempt=attempt,error_type=type(e).__name__,
                                 error_message=str(e),latency_ms=1000*(time.perf_counter()-started),
                                 billing_status="request_not_dispatched" if not_sent else "unknown"))
                if not_sent and attempt<5:
                    time.sleep(min(2**(attempt-1),30));continue
                # A timeout can already have incurred usage. Do not blindly replay it.
                raise
            latency=1000*(time.perf_counter()-started)
            try:response=r.json()
            except ValueError:response=None
            usage=(response or {}).get("usage",{}) or {} if isinstance(response,dict) else {}
            tin=usage.get("input_tokens",usage.get("prompt_tokens",0)) or 0
            tout=usage.get("completion_tokens",0) or 0
            paid=(tin*self.input_price+tout*self.output_price)/1e6
            self.record(dict(**common,event="response",attempt=attempt,http_status=r.status_code,latency_ms=latency,
                             response=response,response_text=r.text if response is None else None,
                             response_model=response.get("model") if isinstance(response,dict) else None,
                             usage=usage,estimated_paid_usd=paid,billing_status="usage_estimate" if usage else "unknown"))
            # Account for truncated responses as well as final responses.
            if usage:
                self.ledger.add(provider="aihubmix",model=payload["model"],tokens=tin+tout,paid=paid,
                                official=(tin*self.official_input+tout*self.output_price)/1e6,latency_ms=latency)
            if r.status_code in [429,500,502,503,504] and attempt<5:
                time.sleep(min(2**(attempt-1),30));continue
            r.raise_for_status()
            if not isinstance(response,dict):raise ValueError("Non-JSON API response; raw body retained")
            complete=True
            if "choices" in response:
                first=response["choices"][0] if response["choices"] else {}
                complete=bool(first.get("message",{}).get("content")) and first.get("finish_reason")!="length"
            if not bypass_cache and complete:
                with _LOCK, self.database() as db:
                    db.execute("INSERT OR IGNORE INTO calls VALUES (?,?,?)",(key,json.dumps(response,ensure_ascii=False),datetime.now(timezone.utc).isoformat()))
            return response,dict(cache_hit=False,key=key,request_id=request_id,latency_ms=latency,
                                 paid_usd=paid,response_model=response.get("model"),http_status=r.status_code,
                                 in_tokens=tin,out_tokens=tout,complete=complete)
        raise RuntimeError("Unreachable request loop")
