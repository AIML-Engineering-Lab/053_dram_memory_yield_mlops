"""
P053 — Kafka Producer for Production Yield Streaming
=====================================================
Reads Parquet day files and publishes each row as a JSON message
to the 'dram-probe-results' Kafka topic.

Simulates real-time data ingestion at configurable throughput:
    - Default: 50,000 msg/sec (one wafer lot per second)
    - Burst: 200,000 msg/sec (stress test)
    - Throttled: 5,000 msg/sec (low-throughput edge case)

Key partitioning: by tester_id (8 partitions → one per tester)
    This ensures temporal ordering per tester, which is how real
    DRAM fabs organize probe data streams.

Usage:
    python -m src.kafka_producer --day 1
    python -m src.kafka_producer --day 1 --end-day 40 --rate 50000
    python -m src.kafka_producer --day 5 --batch-size 10000
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

try:
    from confluent_kafka import Producer
    KAFKA_LIB = "confluent"
except ImportError:
    try:
        from kafka import KafkaProducer as _KafkaProducer
        KAFKA_LIB = "kafka-python"
    except ImportError:
        KAFKA_LIB = None

from src.config import DATA_DIR

PRODUCTION_DIR = DATA_DIR / "production"
TOPIC = "dram-probe-results"
BOOTSTRAP_SERVERS = "localhost:9092"


def _delivery_report(err, msg):
    """Confluent-kafka delivery callback."""
    if err is not None:
        print(f"  [WARN] Message delivery failed: {err}")


def create_producer(bootstrap_servers: str = BOOTSTRAP_SERVERS):
    """Create a Kafka producer using whichever library is available."""
    if KAFKA_LIB == "confluent":
        return Producer({
            "bootstrap.servers": bootstrap_servers,
            "linger.ms": 50,                      # Batch for throughput
            "batch.num.messages": 10000,
            "queue.buffering.max.messages": 500000,
            "compression.type": "snappy",          # Match Parquet compression
            "acks": "1",                           # Leader ack (balance speed/durability)
        })
    elif KAFKA_LIB == "kafka-python":
        return _KafkaProducer(
            bootstrap_servers=[bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=50,
            batch_size=65536,
            compression_type="snappy",
            acks=1,
        )
    else:
        raise ImportError(
            "Neither confluent-kafka nor kafka-python installed. "
            "Install with: pip install confluent-kafka"
        )


def publish_day(day: int, producer, topic: str = TOPIC,
                batch_size: int = 10_000, rate_limit: int = 0) -> dict:
    """
    Read a day's Parquet file and publish all rows to Kafka.

    Args:
        day: Day number (1-40)
        producer: Kafka producer instance
        topic: Target topic name
        batch_size: Rows to send per produce batch
        rate_limit: Max messages/sec (0 = unlimited)

    Returns:
        Summary dict with counts, timing, throughput
    """
    parquet_path = PRODUCTION_DIR / f"day_{day:02d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Day {day} Parquet not found: {parquet_path}\n"
            f"Run: python -m src.streaming_data_generator --day {day}"
        )

    df = pd.read_parquet(parquet_path)
    n_rows = len(df)
    cols = df.columns.tolist()

    t0 = time.time()
    sent = 0
    errors = 0
    batch_start = time.time()

    print(f"Day {day:>2} | Publishing {n_rows:>10,} messages to '{topic}'...", flush=True)

    for start_idx in range(0, n_rows, batch_size):
        chunk = df.iloc[start_idx:start_idx + batch_size]

        for _, row in chunk.iterrows():
            # Build message — convert numpy types to native Python
            msg = {}
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    msg[col] = None
                elif isinstance(val, (np.integer,)):
                    msg[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    msg[col] = float(val)
                else:
                    msg[col] = val

            # Partition key = tester_id (ensures per-tester ordering)
            key = str(msg.get("tester_id", "unknown")).encode("utf-8")

            try:
                if KAFKA_LIB == "confluent":
                    producer.produce(
                        topic, key=key,
                        value=json.dumps(msg).encode("utf-8"),
                        callback=_delivery_report,
                    )
                else:
                    producer.send(topic, key=key, value=msg)
                sent += 1
            except BufferError:
                # Buffer full — flush and retry
                if KAFKA_LIB == "confluent":
                    producer.flush(timeout=10)
                else:
                    producer.flush()
                if KAFKA_LIB == "confluent":
                    producer.produce(
                        topic, key=key,
                        value=json.dumps(msg).encode("utf-8"),
                        callback=_delivery_report,
                    )
                else:
                    producer.send(topic, key=key, value=msg)
                sent += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [ERROR] Row {start_idx}: {e}")

        # Flush each batch
        if KAFKA_LIB == "confluent":
            producer.poll(0)
        if sent % 100_000 == 0 and sent > 0:
            elapsed = time.time() - t0
            rate = sent / elapsed
            print(f"  …{sent:>10,} sent ({rate:,.0f} msg/s)", flush=True)

        # Rate limiting
        if rate_limit > 0:
            expected_time = sent / rate_limit
            actual_time = time.time() - t0
            if actual_time < expected_time:
                time.sleep(expected_time - actual_time)

    # Final flush
    if KAFKA_LIB == "confluent":
        producer.flush(timeout=30)
    else:
        producer.flush()

    elapsed = time.time() - t0
    throughput = sent / max(elapsed, 0.001)

    stats = {
        "day": day, "sent": sent, "errors": errors,
        "elapsed_sec": round(elapsed, 1),
        "throughput_msg_per_sec": round(throughput),
    }
    print(f"  ✓ {sent:,} messages | {elapsed:.1f}s | {throughput:,.0f} msg/s | "
          f"{errors} errors")
    return stats


def publish_range(start_day: int, end_day: int, **kwargs) -> list:
    """Publish multiple days sequentially."""
    producer = create_producer()
    all_stats = []

    for day in range(start_day, end_day + 1):
        stats = publish_day(day, producer, **kwargs)
        all_stats.append(stats)

    total_sent = sum(s["sent"] for s in all_stats)
    total_time = sum(s["elapsed_sec"] for s in all_stats)
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_sent:,} messages in {total_time:.0f}s "
          f"({total_sent/max(total_time,1):,.0f} msg/s avg)")
    print(f"{'='*60}")
    return all_stats


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish production data to Kafka")
    parser.add_argument("--day", type=int, required=True, help="Start day (1-40)")
    parser.add_argument("--end-day", type=int, default=None, help="End day (default: same as --day)")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--rate-limit", type=int, default=0, help="Max msg/sec (0=unlimited)")
    parser.add_argument("--bootstrap-servers", type=str, default=BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    end_day = args.end_day or args.day
    if end_day == args.day:
        prod = create_producer(args.bootstrap_servers)
        publish_day(args.day, prod, batch_size=args.batch_size, rate_limit=args.rate_limit)
    else:
        publish_range(args.day, end_day, batch_size=args.batch_size, rate_limit=args.rate_limit)
