"""
P053 — Kafka Consumer → Parquet Landing Zone
=============================================
Consumes from 'dram-probe-results' topic and writes micro-batches
to the Parquet landing zone for Spark ETL pickup.

Design:
    - Consumer group: 'yield-etl-consumer-group'
    - Micro-batch: 50,000 messages → 1 Parquet file
    - Output: data/landing/day_NN_batch_NNNN.parquet
    - Backpressure: pauses consumption if landing zone > 100 pending files

This is the bridge between Kafka streaming and Spark batch ETL.
In production, this would be Kafka Connect → S3, but we implement
it explicitly to demonstrate understanding of the full pipeline.

Usage:
    python -m src.kafka_consumer
    python -m src.kafka_consumer --batch-size 50000 --max-messages 5000000
"""

import json
import time
import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from confluent_kafka import Consumer, KafkaError
    KAFKA_LIB = "confluent"
except ImportError:
    try:
        from kafka import KafkaConsumer as _KafkaConsumer
        KAFKA_LIB = "kafka-python"
    except ImportError:
        KAFKA_LIB = None

from src.config import DATA_DIR

LANDING_DIR = DATA_DIR / "landing"
LANDING_DIR.mkdir(parents=True, exist_ok=True)

TOPIC = "dram-probe-results"
BOOTSTRAP_SERVERS = "localhost:9092"
GROUP_ID = "yield-etl-consumer-group"

# Graceful shutdown
_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    print("\n[INFO] Graceful shutdown requested...")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def create_consumer(bootstrap_servers: str = BOOTSTRAP_SERVERS,
                    group_id: str = GROUP_ID):
    """Create a Kafka consumer."""
    if KAFKA_LIB == "confluent":
        return Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "max.poll.interval.ms": 300000,
            "fetch.min.bytes": 1024,
            "fetch.max.wait.ms": 500,
        })
    elif KAFKA_LIB == "kafka-python":
        return _KafkaConsumer(
            TOPIC,
            bootstrap_servers=[bootstrap_servers],
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=10000,
        )
    else:
        raise ImportError(
            "Neither confluent-kafka nor kafka-python installed. "
            "Install with: pip install confluent-kafka"
        )


def flush_batch(records: list, batch_num: int) -> Path:
    """Write accumulated records as a Parquet micro-batch."""
    df = pd.DataFrame(records)

    # Detect day from data
    day = int(df["day_number"].mode().iloc[0]) if "day_number" in df.columns else 0
    out_path = LANDING_DIR / f"day_{day:02d}_batch_{batch_num:04d}.parquet"

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    return out_path


def consume_loop(consumer, batch_size: int = 50_000,
                 max_messages: int = 0, max_pending: int = 100) -> dict:
    """
    Main consumption loop.

    Args:
        consumer: Kafka consumer instance
        batch_size: Messages per micro-batch Parquet file
        max_messages: Stop after N total messages (0 = run forever)
        max_pending: Pause if landing zone has > N files

    Returns:
        Summary dict
    """
    global _shutdown

    if KAFKA_LIB == "confluent":
        consumer.subscribe([TOPIC])

    records = []
    batch_num = 0
    total_consumed = 0
    total_bytes = 0
    t0 = time.time()

    print(f"[INFO] Consuming from '{TOPIC}' | Batch size: {batch_size:,}")
    print(f"[INFO] Landing zone: {LANDING_DIR}")
    print(f"[INFO] Press Ctrl+C for graceful shutdown\n")

    while not _shutdown:
        # Backpressure: check landing zone size
        pending_files = list(LANDING_DIR.glob("*.parquet"))
        if len(pending_files) > max_pending:
            print(f"  [BACKPRESSURE] {len(pending_files)} pending files > {max_pending} limit, pausing 5s...")
            time.sleep(5)
            continue

        if KAFKA_LIB == "confluent":
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"  [ERROR] {msg.error()}")
                continue
            value = json.loads(msg.value().decode("utf-8"))
            total_bytes += len(msg.value())
        else:
            # kafka-python: fetch_messages returns an iterator
            try:
                for message in consumer:
                    value = message.value
                    total_bytes += len(json.dumps(value).encode("utf-8"))
                    records.append(value)
                    total_consumed += 1
                    if len(records) >= batch_size:
                        break
                    if _shutdown:
                        break
                else:
                    # Consumer timeout — no more messages
                    if records:
                        pass  # Flush below
                    else:
                        print("[INFO] No messages, waiting...")
                        time.sleep(1)
                        continue
            except StopIteration:
                break

        if KAFKA_LIB == "confluent":
            records.append(value)
            total_consumed += 1

        # Flush micro-batch when full
        if len(records) >= batch_size:
            batch_num += 1
            out_path = flush_batch(records, batch_num)
            elapsed = time.time() - t0
            rate = total_consumed / max(elapsed, 0.001)
            size_mb = out_path.stat().st_size / 1e6
            print(f"  Batch {batch_num:>4} | {len(records):>6,} rows → {out_path.name} "
                  f"({size_mb:.1f} MB) | Total: {total_consumed:>10,} ({rate:,.0f} msg/s)")
            records = []

        # Check max_messages
        if max_messages > 0 and total_consumed >= max_messages:
            print(f"\n[INFO] Reached max_messages limit ({max_messages:,})")
            break

    # Flush remaining records
    if records:
        batch_num += 1
        flush_batch(records, batch_num)
        print(f"  Batch {batch_num:>4} | {len(records):>6,} rows (final flush)")

    # Cleanup
    if KAFKA_LIB == "confluent":
        consumer.close()

    elapsed = time.time() - t0
    stats = {
        "total_consumed": total_consumed,
        "total_batches": batch_num,
        "total_bytes": total_bytes,
        "elapsed_sec": round(elapsed, 1),
        "throughput_msg_per_sec": round(total_consumed / max(elapsed, 0.001)),
    }

    print(f"\n{'='*60}")
    print(f"CONSUMER STATS")
    print(f"  Messages: {total_consumed:,}")
    print(f"  Batches:  {batch_num}")
    print(f"  Data:     {total_bytes/1e9:.2f} GB")
    print(f"  Time:     {elapsed:.0f}s")
    print(f"  Rate:     {stats['throughput_msg_per_sec']:,} msg/s")
    print(f"{'='*60}")
    return stats


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume Kafka messages → Parquet landing zone")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--max-messages", type=int, default=0, help="0 = run forever")
    parser.add_argument("--max-pending", type=int, default=100)
    parser.add_argument("--bootstrap-servers", type=str, default=BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    consumer = create_consumer(args.bootstrap_servers)
    consume_loop(consumer, batch_size=args.batch_size,
                 max_messages=args.max_messages, max_pending=args.max_pending)
