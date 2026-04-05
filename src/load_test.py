"""
P053 — Load Test Script
========================
Simulates production load against the FastAPI endpoint.
Tests single + batch endpoints at various concurrency levels.

Results saved to src/artifacts/load_test_results.json and plotted.

Usage:
    # Start the server first:
    cd src && uvicorn serve:app --port 8000
    # Then run:
    python src/load_test.py --url http://localhost:8000 --duration 60
"""

import argparse
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "httpx"])
    import httpx


def generate_sample_die() -> dict:
    """Generate a random wafer die measurement for testing."""
    return {
        "test_temp_c": float(np.random.uniform(25, 105)),
        "cell_leakage_fa": float(np.random.lognormal(4.6, 0.35)),
        "retention_time_ms": float(np.random.lognormal(5.5, 0.2)),
        "row_hammer_threshold": float(np.random.normal(500000, 50000)),
        "disturb_margin_mv": float(np.random.normal(80, 10)),
        "adjacent_row_activations": float(np.random.poisson(3)),
        "rh_susceptibility": float(np.random.exponential(0.3)),
        "bit_error_rate": float(np.random.exponential(1e-6)),
        "correctable_errors_per_1m": float(np.random.poisson(2)),
        "ecc_syndrome_entropy": float(np.random.exponential(0.5)),
        "uncorrectable_in_extended": float(np.random.poisson(0.1)),
        "trcd_ns": float(np.random.normal(13.5, 0.3)),
        "trp_ns": float(np.random.normal(13.5, 0.3)),
        "tras_ns": float(np.random.normal(32, 1)),
        "rw_latency_ns": float(np.random.lognormal(3.2, 0.15)),
        "idd4_active_ma": float(np.random.normal(450, 20)),
        "idd2p_standby_ma": float(np.random.normal(25, 5)),
        "idd5_refresh_ma": float(np.random.normal(180, 15)),
        "gate_oxide_thickness_a": float(np.random.normal(42, 1)),
        "channel_length_nm": float(np.random.normal(14, 0.3)),
        "vt_shift_mv": float(np.random.normal(0, 15)),
        "block_erase_count": float(np.random.poisson(50)),
        "tester_id": f"T{np.random.randint(1, 11):03d}",
        "probe_card_id": f"PC{np.random.randint(1, 6):03d}",
        "chamber_id": f"CH{np.random.randint(1, 5):03d}",
        "recipe_version": f"R{np.random.choice(['1.0', '1.1', '2.0', '2.1'])}",
        "die_x": float(np.random.randint(0, 200)),
        "die_y": float(np.random.randint(0, 200)),
        "edge_distance": float(np.random.uniform(0, 100)),
    }


def run_single_request(client: httpx.Client, url: str) -> dict:
    """Run a single prediction request and measure latency."""
    payload = generate_sample_die()
    t0 = time.perf_counter()
    response = client.post(f"{url}/predict", json=payload, timeout=30.0)
    latency = time.perf_counter() - t0

    return {
        "status": response.status_code,
        "latency_ms": round(latency * 1000, 2),
        "label": response.json().get("label") if response.status_code == 200 else None,
    }


def run_batch_request(client: httpx.Client, url: str, batch_size: int = 100) -> dict:
    """Run a batch prediction request."""
    payload = {"dies": [generate_sample_die() for _ in range(batch_size)]}
    t0 = time.perf_counter()
    response = client.post(f"{url}/predict/batch", json=payload, timeout=60.0)
    latency = time.perf_counter() - t0

    return {
        "status": response.status_code,
        "latency_ms": round(latency * 1000, 2),
        "batch_size": batch_size,
        "throughput": round(batch_size / latency, 1) if latency > 0 else 0,
    }


def load_test(url: str, concurrency_levels: list[int], duration_s: int = 30) -> dict:
    """Run load test at multiple concurrency levels."""
    results = {
        "url": url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "single_endpoint": [],
        "batch_endpoint": [],
    }

    # Single endpoint test
    for concurrency in concurrency_levels:
        print(f"\n--- Single /predict @ {concurrency} concurrent ---")
        latencies = []
        errors = 0
        t_start = time.time()

        with httpx.Client() as client:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = []
                while time.time() - t_start < duration_s:
                    # Submit batch of requests
                    for _ in range(concurrency):
                        futures.append(pool.submit(run_single_request, client, url))

                    # Collect completed
                    for f in as_completed(futures):
                        try:
                            r = f.result()
                            if r["status"] == 200:
                                latencies.append(r["latency_ms"])
                            else:
                                errors += 1
                        except Exception:
                            errors += 1
                    futures = []

        if latencies:
            result = {
                "concurrency": concurrency,
                "total_requests": len(latencies) + errors,
                "errors": errors,
                "error_rate_pct": round(100 * errors / (len(latencies) + errors), 2),
                "throughput_rps": round(len(latencies) / duration_s, 1),
                "latency_p50_ms": round(statistics.median(latencies), 2),
                "latency_p95_ms": round(np.percentile(latencies, 95), 2),
                "latency_p99_ms": round(np.percentile(latencies, 99), 2),
                "latency_mean_ms": round(statistics.mean(latencies), 2),
            }
            results["single_endpoint"].append(result)
            print(f"  Requests: {result['total_requests']}, "
                  f"RPS: {result['throughput_rps']}, "
                  f"P50: {result['latency_p50_ms']}ms, "
                  f"P95: {result['latency_p95_ms']}ms, "
                  f"P99: {result['latency_p99_ms']}ms")

    # Batch endpoint test
    for batch_size in [10, 100, 500, 1024]:
        print(f"\n--- Batch /predict/batch @ batch_size={batch_size} ---")
        latencies = []
        with httpx.Client() as client:
            for _ in range(5):  # 5 repetitions
                r = run_batch_request(client, url, batch_size)
                if r["status"] == 200:
                    latencies.append(r["latency_ms"])

        if latencies:
            result = {
                "batch_size": batch_size,
                "latency_mean_ms": round(statistics.mean(latencies), 2),
                "throughput_samples_per_s": round(
                    batch_size / (statistics.mean(latencies) / 1000), 1
                ),
            }
            results["batch_endpoint"].append(result)
            print(f"  Mean: {result['latency_mean_ms']}ms, "
                  f"Throughput: {result['throughput_samples_per_s']} samples/s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P053 Load Test")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=30, help="Duration per concurrency level (s)")
    parser.add_argument("--output", default="src/artifacts/load_test_results.json")
    args = parser.parse_args()

    concurrency_levels = [1, 5, 10, 25, 50]
    results = load_test(args.url, concurrency_levels, args.duration)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")
