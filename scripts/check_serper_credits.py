#!/usr/bin/env python3
"""Query the remaining credits for the configured Serper API key pool."""

import argparse
import concurrent.futures
import datetime
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ACCOUNT_URL = "https://google.serper.dev/account"
SEARCH_URL = "https://google.serper.dev/search"
STRESS_TEST_REQUESTS = 64
API_KEYS = [
    "4e2e77db8d642257c05ea2d5d81a510cd00e77f9",
    "3939dba245b5329886c619b4b4f75c0859f3c693",
    "c72a82b47693b8d897be07114b722a12b027d916",
    "2ce4fae0bbd7e810c6371830aa8f3b93114e6805",
    "aaba06b8ac3aa79153511b8de4d6ac616db6f08a",
    "84c9a0a3d08ad2dcb0625120b432e32d89753999",
    "1913552d007e952d4101cee03f37cbde1b8fe2b2",
    "0f1d6c7751bddb95cf7b127dd418fb3451c4749a",
]


def query_account(api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        ACCOUNT_URL,
        headers={"X-API-KEY": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {error.code}: {detail or error.reason}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        raise RuntimeError(f"request failed: {reason}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Serper returned an invalid JSON response") from error

    if "balance" not in payload:
        raise RuntimeError(f"response does not contain 'balance': {payload}")
    return payload


def stress_query(request_id: int, query: str, api_key: str, timeout: float) -> dict:
    body = json.dumps({"q": query}).encode("utf-8")
    request = urllib.request.Request(
        SEARCH_URL,
        data=body,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    result = {
        "request_id": request_id,
        "query": query,
        "key": key_label(api_key, show_keys=False),
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            result["http_status"] = response.status
        try:
            result["response"] = json.loads(response_body)
            result["success"] = 200 <= result["http_status"] < 300
        except json.JSONDecodeError:
            result.update(success=False, response=response_body, error="response is not valid JSON")
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        result.update(success=False, http_status=error.code, error=f"HTTP {error.code}: {error.reason}")
        try:
            result["response"] = json.loads(response_body)
        except json.JSONDecodeError:
            result["response"] = response_body
    except (urllib.error.URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        result.update(success=False, http_status=None, response=None, error=f"request failed: {reason}")
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def key_label(api_key: str, show_keys: bool) -> str:
    return api_key if show_keys else f"{api_key[:8]}...{api_key[-4:]}"


def query_credit_snapshot(timeout: float) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
        futures = [executor.submit(query_account, api_key, timeout) for api_key in API_KEYS]

    snapshot = []
    for api_key, future in zip(API_KEYS, futures, strict=True):
        item = {"key": key_label(api_key, show_keys=False)}
        try:
            account = future.result()
            item.update(balance=account["balance"], rate_limit=account.get("rateLimit"))
        except RuntimeError as error:
            item["error"] = str(error)
        snapshot.append(item)
    return snapshot


def print_credit_snapshot(title: str, snapshot: list[dict]) -> None:
    print(title)
    print(f"{'API key':<24} {'Credits':>12} {'Rate limit':>12}")
    print("-" * 50)
    for item in snapshot:
        if "error" in item:
            print(f"{item['key']:<24} ERROR: {item['error']}")
        else:
            rate_limit = item["rate_limit"] if item["rate_limit"] is not None else "-"
            print(f"{item['key']:<24} {item['balance']:>12} {rate_limit:>12}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-keys", action="store_true", help="print complete API keys")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help=f"run {STRESS_TEST_REQUESTS} concurrent searches across the API key pool",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="timeout per request in seconds (default: 10)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def run_stress_test(timeout: float) -> int:
    queries = [f"Serper parallel API test unique query {request_id:02d}" for request_id in range(1, 65)]
    started_at = datetime.datetime.now(datetime.timezone.utc)
    credits_before = query_credit_snapshot(timeout)
    print_credit_snapshot("Credits before stress test", credits_before)

    with concurrent.futures.ThreadPoolExecutor(max_workers=STRESS_TEST_REQUESTS) as executor:
        futures = [
            executor.submit(stress_query, request_id, query, API_KEYS[(request_id - 1) % len(API_KEYS)], timeout)
            for request_id, query in enumerate(queries, start=1)
        ]
        results = [future.result() for future in futures]
    credits_after = query_credit_snapshot(timeout)
    print_credit_snapshot("Credits after stress test", credits_after)
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    succeeded = sum(result["success"] for result in results)
    per_key = []
    for api_key in API_KEYS:
        label = key_label(api_key, show_keys=False)
        key_results = [result for result in results if result["key"] == label]
        key_succeeded = sum(result["success"] for result in key_results)
        per_key.append(
            {
                "key": label,
                "requests": len(key_results),
                "succeeded": key_succeeded,
                "failed": len(key_results) - key_succeeded,
                "success_rate": key_succeeded / len(key_results),
            }
        )

    report = {
        "summary": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round((finished_at - started_at).total_seconds(), 3),
            "requests": len(results),
            "succeeded": succeeded,
            "failed": len(results) - succeeded,
            "success_rate": succeeded / len(results),
            "keys": len(API_KEYS),
            "requests_per_key": len(results) // len(API_KEYS),
        },
        "credits_before": credits_before,
        "credits_after": credits_after,
        "per_key": per_key,
        "requests": results,
    }
    logs_dir = Path(__file__).resolve().parents[1] / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = logs_dir / f"serper_pool_stress_test_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Requests: {len(results)}; succeeded: {succeeded}; failed: {len(results) - succeeded}")
    print(f"Success rate: {report['summary']['success_rate']:.2%}")
    for item in per_key:
        print(f"  {item['key']}: {item['succeeded']}/{item['requests']} ({item['success_rate']:.2%})")
    before_by_key = {item["key"]: item for item in credits_before}
    print("Credit usage:")
    for item in credits_after:
        before = before_by_key[item["key"]]
        if "balance" in before and "balance" in item:
            print(f"  {item['key']}: {before['balance']} -> {item['balance']} (used {before['balance'] - item['balance']})")
        else:
            print(f"  {item['key']}: unavailable")
    print(f"Report: {output_path}")
    return 0 if succeeded == len(results) else 1


def main() -> int:
    args = parse_args()
    if args.stress_test:
        return run_stress_test(args.timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
        futures = [executor.submit(query_account, api_key, args.timeout) for api_key in API_KEYS]

    results = []
    failed = False
    for api_key, future in zip(API_KEYS, futures, strict=True):
        item = {"key": key_label(api_key, args.show_keys)}
        try:
            account = future.result()
            item.update(balance=account["balance"], rate_limit=account.get("rateLimit"))
        except RuntimeError as error:
            item["error"] = str(error)
            failed = True
        results.append(item)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'API key':<24} {'Credits':>12} {'Rate limit':>12}")
        print("-" * 50)
        for item in results:
            if "error" in item:
                print(f"{item['key']:<24} ERROR: {item['error']}")
            else:
                rate_limit = item["rate_limit"] if item["rate_limit"] is not None else "-"
                print(f"{item['key']:<24} {item['balance']:>12} {rate_limit:>12}")
        balances = [item["balance"] for item in results if "balance" in item]
        print("-" * 50)
        print(f"Total credits: {sum(balances)} ({len(balances)}/{len(results)} keys succeeded)")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
