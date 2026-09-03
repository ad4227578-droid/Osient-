import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Response
import uvicorn

# ── Config ──────────────────────────────────────────────────────────────────
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")
PORT = int(os.environ.get("PORT", 10000))

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]
DUPLICATE_CAP = 2

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── DuckDB Single-Thread Optimized Pool ─────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duck")

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET memory_limit='250MB'")
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET http_keep_alive=true;")

    for kind, urls in REMOTE_INDEXES.items():
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW people_{kind} AS SELECT * FROM read_parquet([{lst}])")
    con.execute("SET threads = 2")
    return con

def _get_conn() -> duckdb.DuckDBPyConnection:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
            _conns.append(_new_conn())
    return _conns[tid]

# ── High-Speed Helpers ──────────────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    return (row.get("phoneNumber") or "").strip(), (row.get("aadharNumber") or "").strip()

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen, out = {}, []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            # Redact sensitive identifiers securely
            if record.get("aadharNumber"):
                record["aadharNumber"] = "[Redacted]"
            connected, c_seen = [], set()
            for f in NUMBER_FIELDS:
                val = str(record.get(f, "")).strip()
                if val and val not in c_seen:
                    c_seen.add(val)
                    connected.append({"field": f, "value": "[Redacted]" if f == "aadharNumber" else val})
            record["connected_numbers"] = connected
            out.append(record)
    return out

def _run_search(q: str, limit: int) -> dict:
    v = q.replace("'", "''")
    con = _get_conn()
    
    # Direct optimized query hitting phone index first
    sql = f"SELECT * FROM people_phone WHERE phoneNumber = '{v}' LIMIT {limit * DUPLICATE_CAP + 10}"
    rows = con.execute(sql.format(v=v)).fetchall()
    
    if not rows:
        sql = f"SELECT * FROM people_aadhar WHERE aadharNumber = '{v}' LIMIT {limit * DUPLICATE_CAP + 10}"
        rows = con.execute(sql).fetchall()

    cols = [d[0] for d in con.description] if con.description else []
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    
    return {
        "query": q,
        "count": len(results),
        "results": results
    }

# ── Ultra-Fast LRU Cache Layer (Microsecond Responses) ──────────────────────
@lru_cache(maxsize=2000)
def _cached_search_str(q: str, limit: int) -> str:
    data = _run_search(q, limit)
    return json.dumps(data, ensure_ascii=False)

def get_fast_response(q: str, limit: int) -> dict:
    return json.loads(_cached_search_str(q, limit))

# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="Ultra Fast ICMR API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
async def search(q: str = Query(...), limit: int = Query(10, ge=1, le=50)):
    q_val = q.strip()
    if not q_val:
        raise HTTPException(422, "Provide search query")
    
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, get_fast_response, q_val, limit)
    
    response_data = {"success": bool(data["count"]), **data, "total": data["count"]}
    return Response(content=json.dumps(response_data, ensure_ascii=False), media_type="application/json")

# ── Warm-up on Startup ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(pool, lambda: _run_search("0000000000", 1))
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)

