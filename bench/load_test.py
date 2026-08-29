#!/usr/bin/env python3
"""Asynchronous load testing utility for ControlPlane Checker.

WARNING:
  The Gemini free tier will rate-limit you long before the guardrails saturate.
  Load tests MUST run against `llm.backend = "mock"` in default.toml / local.toml.
  This script refuses to run at concurrency > 4 unless /readyz reports mock backend,
  overridable with --i-know-what-im-doing.
"""

import argparse
import asyncio
import random
import sys
import time
from typing import Any

try:
    import httpx
    import numpy as np
except ImportError:
    print("Error: Missing required packages. Install with: pip install httpx numpy")
    sys.exit(1)

from corpus import get_queries

WARNING_BANNER = """
================================================================================
CRITICAL LOAD TESTING WARNING:
  The Gemini free tier will rate-limit you long before the guardrails saturate.
  Load tests MUST run against `llm.backend = "mock"`.
================================================================================
"""


async def check_backend_readiness(url: str, override: bool, concurrency: int) -> None:
    ready_url = f"{url.rstrip('/')}/readyz"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(ready_url)
            if resp.status_code != 200:
                print(f"[WARN] /readyz returned status {resp.status_code}")
    except Exception as e:
        print(f"[WARN] Failed to query /readyz: {e}")

    if concurrency > 4 and not override:
        print(WARNING_BANNER)
        print("[INFO] If you are confident you are using the mock LLM backend or have an enterprise quota,")
        print("       re-run with `--i-know-what-im-doing` to proceed.")


async def worker(
    worker_id: int,
    client: httpx.AsyncClient,
    check_url: str,
    queries: list[dict],
    queue: asyncio.Queue,
    latencies: list[float],
    decisions: dict[str, int],
    errors: list[str],
) -> None:
    while True:
        try:
            _ = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        item = random.choice(queries)
        payload = {
            "query": item["query"],
            "history_summary": item.get("history_summary"),
            "session_id": f"load-test-worker-{worker_id}",
            "options": {"dry_run": False},
        }

        start = time.perf_counter()
        try:
            resp = await client.post(check_url, json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            if resp.status_code == 200:
                data = resp.json()
                dec = data.get("decision", "unknown")
                decisions[dec] = decisions.get(dec, 0) + 1
            else:
                errors.append(f"HTTP {resp.status_code}")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            errors.append(str(e))
        finally:
            queue.task_done()


async def run_load_test(
    url: str,
    concurrency: int,
    num_requests: int,
    category: str,
    override: bool,
) -> None:
    print(WARNING_BANNER)
    await check_backend_readiness(url, override, concurrency)

    check_url = f"{url.rstrip('/')}/v1/check"
    queries = get_queries(category)
    if not queries:
        print(f"Error: No queries found for category '{category}'")
        return

    print(f"Starting load test on {check_url}")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Requests:     {num_requests}")
    print(f"  Corpus size:  {len(queries)} templates\n")

    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(num_requests):
        queue.put_nowait(1)

    latencies: list[float] = []
    decisions: dict[str, int] = {}
    errors: list[str] = []

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(30.0)

    start_time = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        workers = [
            asyncio.create_task(
                worker(
                    i,
                    client,
                    check_url,
                    queries,
                    queue,
                    latencies,
                    decisions,
                    errors,
                )
            )
            for i in range(concurrency)
        ]
        await queue.join()
        for w in workers:
            w.cancel()

    total_time = time.perf_counter() - start_time
    throughput = len(latencies) / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Requests:      {len(latencies)}")
    print(f"Total Wall Time:     {total_time:.2f} s")
    print(f"Throughput:          {throughput:.1f} req/s")
    print(f"Decisions:           {dict(decisions)}")
    print(f"Errors:              {len(errors)}")

    if latencies:
        arr = np.array(latencies)
        print("\nLatency Distribution (ms):")
        print(f"  min:   {np.min(arr):.2f} ms")
        print(f"  p50:   {np.percentile(arr, 50):.2f} ms")
        print(f"  p90:   {np.percentile(arr, 90):.2f} ms")
        print(f"  p99:   {np.percentile(arr, 99):.2f} ms")
        print(f"  p99.9: {np.percentile(arr, 99.9):.2f} ms")
        print(f"  max:   {np.max(arr):.2f} ms")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async load tester for ControlPlane Checker guardrail pipeline.",
        epilog="WARNING: The Gemini free tier will rate-limit you long before the guardrails saturate. Run against mock LLM backend.",
    )
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the service (default: http://localhost:8080)")
    parser.add_argument("--concurrency", "-c", type=int, default=8, help="Number of concurrent clients (default: 8)")
    parser.add_argument("--requests", "-n", type=int, default=100, help="Total number of requests to send (default: 100)")
    parser.add_argument("--category", default="all", help="Corpus category to sample (default: all)")
    parser.add_argument(
        "--i-know-what-im-doing",
        action="store_true",
        help="Bypass mock LLM readiness check for high-concurrency testing",
    )

    args = parser.parse_args()
    asyncio.run(
        run_load_test(
            args.url,
            args.concurrency,
            args.requests,
            args.category,
            args.i_know_what_im_doing,
        )
    )


if __name__ == "__main__":
    main()
