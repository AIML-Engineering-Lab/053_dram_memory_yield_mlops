#!/usr/bin/env python3
"""Build a static S3 artifacts inventory report for P053."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "S3_Artifacts_Inventory_Report.html"

DEFAULT_BUCKET = "p053-mlflow-artifacts"
DEFAULT_REGION = "us-west-2"
DEFAULT_INSTANCE_ID = "i-0562654a22d44346f"
DEFAULT_DB_ID = "p053-mlflow-db"
WORKFLOW_FIX_COMMIT = "8b72d8e"

DAY_RE = re.compile(r"(?:^|/)day[_-]?(\d+)(?:/|_|\.|$)", re.IGNORECASE)


def aws_json(args: list[str], region: str, default: Any) -> Any:
    command = ["aws", *args]
    if "--region" not in args:
        command.extend(["--region", region])
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return default
    try:
        return json.loads(proc.stdout or "null") or default
    except json.JSONDecodeError:
        return default


def aws_text(args: list[str], region: str) -> str:
    command = ["aws", *args]
    if "--region" not in args:
        command.extend(["--region", region])
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    return proc.stdout if proc.returncode == 0 else ""


def list_current_objects(bucket: str, region: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        args = ["s3api", "list-objects-v2", "--bucket", bucket]
        if token:
            args.extend(["--continuation-token", token])
        data = aws_json(args, region, {})
        objects.extend(data.get("Contents") or [])
        token = data.get("NextContinuationToken")
        if not token:
            break
    return objects


def list_versions(bucket: str, region: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    versions: list[dict[str, Any]] = []
    delete_markers: list[dict[str, Any]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    while True:
        args = ["s3api", "list-object-versions", "--bucket", bucket]
        if key_marker:
            args.extend(["--key-marker", key_marker])
        if version_marker:
            args.extend(["--version-id-marker", version_marker])
        data = aws_json(args, region, {})
        versions.extend(data.get("Versions") or [])
        delete_markers.extend(data.get("DeleteMarkers") or [])
        key_marker = data.get("NextKeyMarker")
        version_marker = data.get("NextVersionIdMarker")
        if not key_marker:
            break
    return versions, delete_markers


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def fmt_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:,.2f} {unit}" if unit != "B" else f"{int(size):,} B"
        size /= 1024
    return f"{size:,.2f} TB"


def fmt_date(value: Any) -> str:
    if not value:
        return "-"
    text = str(value).replace("+00:00", "Z")
    return text[:19].replace("T", " ") + " UTC"


def pct(part: float, total: float) -> str:
    return "0.0%" if total <= 0 else f"{(part / total) * 100:,.1f}%"


def top_prefix(key: str) -> str:
    return key.split("/", 1)[0] if "/" in key else "(root)"


def artifact_type(key: str) -> str:
    if key.startswith("data/production/"):
        return "Production Parquet"
    if key.startswith("drift/") or key.startswith("drift_reports/"):
        return "Drift Reports"
    if key.startswith("daily_metrics/"):
        return "Daily Summaries"
    if key.startswith("models/"):
        return "Model Artifacts"
    if key.startswith("state/"):
        return "Pipeline State"
    return "Benchmarks" if key.startswith("benchmarks/") else "Other"


def parse_day(key: str) -> int | None:
    match = DAY_RE.search(key)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def collect_groups(objects: list[dict[str, Any]], key_func) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"objects": 0, "bytes": 0, "latest": ""})
    for obj in objects:
        key = obj.get("Key", "")
        group = key_func(key)
        size = int(obj.get("Size") or 0)
        grouped[group]["objects"] += 1
        grouped[group]["bytes"] += size
        modified = str(obj.get("LastModified") or "")
        if modified > grouped[group]["latest"]:
            grouped[group]["latest"] = modified
    return [
        {"name": name, **stats}
        for name, stats in sorted(grouped.items(), key=lambda item: item[1]["bytes"], reverse=True)
    ]


def collect_day_rows(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"objects": 0, "bytes": 0, "parquet_bytes": 0, "types": set(), "latest": ""}
    )
    for obj in objects:
        key = obj.get("Key", "")
        day = parse_day(key)
        if day is None:
            continue
        size = int(obj.get("Size") or 0)
        row = days[day]
        row["objects"] += 1
        row["bytes"] += size
        row["types"].add(artifact_type(key))
        if key.startswith("data/production/") and key.endswith(".parquet"):
            row["parquet_bytes"] += size
        modified = str(obj.get("LastModified") or "")
        if modified > row["latest"]:
            row["latest"] = modified

    return [
        {
            "day": day,
            "status": "Accidental extra" if day > 40 else "Intended run",
            "objects": row["objects"],
            "bytes": row["bytes"],
            "parquet_bytes": row["parquet_bytes"],
            "types": sorted(row["types"]),
            "latest": row["latest"],
        }
        for day, row in sorted(days.items())
    ]


def collect_resource_audit(region: str, instance_id: str, db_id: str) -> dict[str, Any]:
    instances = aws_json(["ec2", "describe-instances", "--instance-ids", instance_id], region, {})
    instance_items = [item for reservation in instances.get("Reservations", []) for item in reservation.get("Instances", [])]
    volumes = aws_json(
        ["ec2", "describe-volumes", "--filters", f"Name=attachment.instance-id,Values={instance_id}"],
        region,
        {},
    ).get("Volumes", [])
    rds = aws_json(["rds", "describe-db-instances", "--db-instance-identifier", db_id], region, {}).get(
        "DBInstances", []
    )
    nat = aws_json(
        ["ec2", "describe-nat-gateways", "--filter", "Name=state,Values=available,pending"], region, {}
    ).get("NatGateways", [])
    addresses = aws_json(["ec2", "describe-addresses"], region, {}).get("Addresses", [])
    snapshots = aws_json(
        ["rds", "describe-db-snapshots", "--db-instance-identifier", db_id, "--snapshot-type", "automated"],
        region,
        {},
    ).get("DBSnapshots", [])
    ecr_repos = aws_json(["ecr", "describe-repositories"], region, {}).get("repositories", [])
    project_repos = [
        repo for repo in ecr_repos if any(token in repo.get("repositoryName", "") for token in ("053", "p053", "memory"))
    ]
    return {
        "instances": instance_items,
        "volumes": volumes,
        "rds": rds,
        "nat": nat,
        "addresses": addresses,
        "snapshots": snapshots,
        "ecr_repos": project_repos,
    }


def load_pipeline_state(bucket: str, region: str) -> dict[str, Any]:
    text = aws_text(["s3", "cp", f"s3://{bucket}/state/pipeline_state.json", "-"], region)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def table(headers: list[str], rows: list[list[str]], class_name: str = "") -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = ["<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows]
    return f"<table class=\"{class_name}\"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def bar_chart(rows: list[dict[str, Any]], title: str, value_key: str = "bytes") -> str:
    if not rows:
        return "<p>No data available.</p>"
    max_value = max(float(row.get(value_key) or 0) for row in rows) or 1
    colors = ["#0f9f8f", "#ef7b45", "#3d5afe", "#d1495b", "#2a9d8f", "#f4a261", "#4361ee"]
    parts = [f"<h3>{escape(title)}</h3><div class=\"bar-chart\">"]
    for index, row in enumerate(rows):
        value = float(row.get(value_key) or 0)
        width = max(2.0, (value / max_value) * 100)
        color = colors[index % len(colors)]
        parts.append(
            "<div class=\"bar-row\">"
            f"<div class=\"bar-label\">{escape(str(row['name']))}</div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar-fill\" style=\"width:{width:.2f}%;background:{color}\"></div>"
            "</div>"
            f"<div class=\"bar-value\">{fmt_bytes(value)}</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def day_svg(day_rows: list[dict[str, Any]]) -> str:
    rows = [row for row in day_rows if row["parquet_bytes"] > 0 or row["bytes"] > 0]
    if not rows:
        return "<p>No day-level artifacts found.</p>"
    width = 980
    height = 320
    pad_left = 58
    pad_bottom = 52
    pad_top = 26
    chart_width = width - pad_left - 24
    chart_height = height - pad_top - pad_bottom
    max_value = max(row["bytes"] for row in rows) or 1
    gap = 6
    bar_width = max(10, (chart_width - gap * (len(rows) - 1)) / len(rows))
    parts = [
        f"<svg class=\"day-chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Day wise S3 artifact size chart\">",
        "<rect width=\"100%\" height=\"100%\" rx=\"18\" fill=\"#fffaf0\"/>",
        f"<line x1=\"{pad_left}\" y1=\"{height - pad_bottom}\" x2=\"{width - 16}\" y2=\"{height - pad_bottom}\" stroke=\"#d7c8a8\"/>",
        f"<line x1=\"{pad_left}\" y1=\"{pad_top}\" x2=\"{pad_left}\" y2=\"{height - pad_bottom}\" stroke=\"#d7c8a8\"/>",
        f"<text x=\"{pad_left}\" y=\"20\" class=\"svg-title\">Day-wise current S3 artifact size</text>",
    ]
    for index, row in enumerate(rows):
        value = row["bytes"]
        bar_height = (value / max_value) * chart_height
        x = pad_left + index * (bar_width + gap)
        y = height - pad_bottom - bar_height
        color = "#ef7b45" if row["day"] > 40 else "#0f9f8f"
        label_color = "#9a3412" if row["day"] > 40 else "#0f766e"
        parts.extend(
            [
                f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_width:.1f}\" height=\"{bar_height:.1f}\" rx=\"5\" fill=\"{color}\"/>",
                f"<text x=\"{x + bar_width / 2:.1f}\" y=\"{height - 30}\" text-anchor=\"middle\" class=\"svg-label\" fill=\"{label_color}\">{row['day']}</text>",
            ]
        )
    parts.extend(
        [
            f"<text x=\"{pad_left}\" y=\"{height - 8}\" class=\"svg-note\">Teal = intended production days, orange = accidental post-Day40 artifacts retained for now.</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def card(label: str, value: str, note: str, tone: str = "teal") -> str:
    return (
        f"<div class=\"metric-card {tone}\">"
        f"<div class=\"metric-label\">{escape(label)}</div>"
        f"<div class=\"metric-value\">{escape(value)}</div>"
        f"<div class=\"metric-note\">{escape(note)}</div>"
        "</div>"
    )


def build_html(
    *,
    bucket: str,
    region: str,
    objects: list[dict[str, Any]],
    versions: list[dict[str, Any]],
    delete_markers: list[dict[str, Any]],
    state: dict[str, Any],
    resources: dict[str, Any],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_bytes = sum(int(obj.get("Size") or 0) for obj in objects)
    version_bytes = sum(int(obj.get("Size") or 0) for obj in versions)
    prefix_rows = collect_groups(objects, top_prefix)
    type_rows = collect_groups(objects, artifact_type)
    day_rows = collect_day_rows(objects)
    accidental_rows = [row for row in day_rows if row["day"] > 40]
    model_objects = [obj for obj in objects if str(obj.get("Key", "")).startswith("models/")]
    largest = sorted(objects, key=lambda obj: int(obj.get("Size") or 0), reverse=True)[:12]

    instance = (resources.get("instances") or [{}])[0]
    rds = (resources.get("rds") or [{}])[0]
    volume_total = sum(int(vol.get("Size") or 0) for vol in resources.get("volumes") or [])

    summary_cards = "".join(
        [
            card("Current S3 objects", fmt_int(len(objects)), f"In s3://{bucket}", "teal"),
            card("Current S3 size", fmt_bytes(total_bytes), "Latest object versions only", "amber"),
            card("Object versions", fmt_int(len(versions)), f"{fmt_int(len(delete_markers))} delete markers", "blue"),
            card("All-version size", fmt_bytes(version_bytes), "Includes noncurrent versions", "coral"),
            card("Day artifacts found", fmt_int(len(day_rows)), "Current day-level prefixes", "green"),
            card("Accidental extras", fmt_int(len(accidental_rows)), "Days above 40 retained for now", "orange"),
        ]
    )

    prefix_table = table(
        ["Prefix", "Objects", "Current Size", "Share", "Latest Modified"],
        [
            [
                escape(row["name"]),
                fmt_int(row["objects"]),
                fmt_bytes(row["bytes"]),
                pct(row["bytes"], total_bytes),
                fmt_date(row["latest"]),
            ]
            for row in prefix_rows
        ],
        "dense",
    )

    type_table = table(
        ["Artifact Type", "Objects", "Current Size", "Share", "Latest Modified"],
        [
            [
                escape(row["name"]),
                fmt_int(row["objects"]),
                fmt_bytes(row["bytes"]),
                pct(row["bytes"], total_bytes),
                fmt_date(row["latest"]),
            ]
            for row in type_rows
        ],
        "dense",
    )

    day_table = table(
        ["Day", "Status", "Objects", "Total Size", "Parquet Size", "Types", "Latest Modified"],
        [
            [
                f"Day {row['day']}",
                f"<span class=\"pill {'warn' if row['day'] > 40 else 'ok'}\">{escape(row['status'])}</span>",
                fmt_int(row["objects"]),
                fmt_bytes(row["bytes"]),
                fmt_bytes(row["parquet_bytes"]),
                escape(", ".join(row["types"])),
                fmt_date(row["latest"]),
            ]
            for row in day_rows
        ],
        "dense day-table",
    )

    largest_table = table(
        ["Rank", "Key", "Size", "Last Modified"],
        [
            [
                fmt_int(index + 1),
                f"<code>{escape(str(obj.get('Key', '')))}</code>",
                fmt_bytes(obj.get("Size")),
                fmt_date(obj.get("LastModified")),
            ]
            for index, obj in enumerate(largest)
        ],
        "dense key-table",
    )

    model_table = table(
        ["Model Artifact", "Size", "Last Modified"],
        [
            [
                f"<code>{escape(str(obj.get('Key', '')))}</code>",
                fmt_bytes(obj.get("Size")),
                fmt_date(obj.get("LastModified")),
            ]
            for obj in sorted(model_objects, key=lambda item: str(item.get("Key", "")))
        ],
        "dense key-table",
    )

    resource_rows = [
        ["EC2", escape(str(instance.get("InstanceId", DEFAULT_INSTANCE_ID))), escape(str((instance.get("State") or {}).get("Name", "unknown"))), escape(str(instance.get("InstanceType", "-"))), "Stopped compute means no g4dn runtime charge."],
        ["Attached EBS", f"{fmt_int(volume_total)} GiB", "billable", "gp3 volume(s)", "Kept as evidence; delete/terminate only with explicit approval."],
        ["RDS", escape(str(rds.get("DBInstanceIdentifier", DEFAULT_DB_ID))), escape(str(rds.get("DBInstanceStatus", "unknown"))), escape(str(rds.get("DBInstanceClass", "-"))), "Storage and snapshots remain billable while retained."],
        ["RDS snapshots", fmt_int(len(resources.get("snapshots") or [])), "billable", "automated", "Can be removed only during destructive cleanup."],
        ["NAT gateways", fmt_int(len(resources.get("nat") or [])), "review" if resources.get("nat") else "none active", "available/pending", "No NAT runtime cost found in this audit."],
        ["Public IPv4/EIP", fmt_int(len(resources.get("addresses") or [])), "billable if allocated/associated", "EC2 address inventory", "One associated public IPv4/EIP was retained in the July 17 audit."],
        ["ECR repos", fmt_int(len(resources.get("ecr_repos") or [])), "retained", "project repositories", "Small storage cost if images remain."],
    ]
    resource_table = table(["Area", "Identifier/Count", "State", "Class", "Cost Note"], resource_rows, "dense")

    state_table = table(
        ["Field", "Value"],
        [[escape(str(key)), f"<code>{escape(str(value))}</code>"] for key, value in state.items()],
        "dense",
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>P053 S3 Artifacts Inventory Report</title>
  <style>
    :root {{
      --paper: #fff8ea;
      --ink: #24313f;
      --muted: #677386;
      --line: #e4d4b5;
      --teal: #0f9f8f;
      --amber: #f2a541;
      --coral: #d1495b;
      --blue: #3d5afe;
      --green: #2a9d8f;
      --orange: #ef7b45;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(135deg, #fff8ea 0%, #f1f7ed 45%, #eaf7ff 100%);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 34px 22px 56px; }}
    .hero {{
      border: 1px solid rgba(36,49,63,0.12);
      background: rgba(255,255,255,0.74);
      border-radius: 18px;
      padding: 30px;
      box-shadow: 0 18px 42px rgba(36,49,63,0.12);
    }}
    .eyebrow {{ color: var(--teal); font-weight: 800; letter-spacing: .08em; text-transform: uppercase; font-size: 12px; }}
    h1 {{ margin: 8px 0 12px; font-size: clamp(32px, 4vw, 56px); line-height: 1.02; }}
    h2 {{ margin: 36px 0 14px; font-size: 28px; }}
    h3 {{ margin: 18px 0 10px; font-size: 18px; }}
    p {{ margin: 0 0 14px; }}
    code {{ background: #f5ead4; color: #7a3417; padding: 2px 5px; border-radius: 5px; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 22px 0 8px; }}
    .metric-card {{ background: #fff; border: 1px solid var(--line); border-top: 6px solid var(--teal); border-radius: 14px; padding: 16px; }}
    .metric-card.amber {{ border-top-color: var(--amber); }}
    .metric-card.blue {{ border-top-color: var(--blue); }}
    .metric-card.coral {{ border-top-color: var(--coral); }}
    .metric-card.green {{ border-top-color: var(--green); }}
    .metric-card.orange {{ border-top-color: var(--orange); }}
    .metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 800; }}
    .metric-value {{ font-size: 28px; font-weight: 900; margin-top: 3px; }}
    .metric-note {{ color: var(--muted); font-size: 13px; }}
    .callout {{ border-left: 6px solid var(--orange); background: #fff5e8; padding: 16px 18px; border-radius: 12px; margin: 18px 0; }}
    .callout.good {{ border-left-color: var(--teal); background: #ecfffb; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .panel {{ background: rgba(255,255,255,0.82); border: 1px solid var(--line); border-radius: 16px; padding: 18px; overflow: hidden; }}
    .bar-chart {{ display: grid; gap: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: 150px 1fr 94px; align-items: center; gap: 10px; }}
    .bar-label {{ font-weight: 750; font-size: 13px; overflow-wrap: anywhere; }}
    .bar-track {{ height: 14px; background: #efe2c8; border-radius: 99px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 99px; }}
    .bar-value {{ color: var(--muted); font-variant-numeric: tabular-nums; font-size: 13px; text-align: right; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 14px; overflow: hidden; box-shadow: 0 0 0 1px var(--line); }}
    th, td {{ padding: 12px 13px; border-bottom: 1px solid #eadcc4; text-align: left; vertical-align: top; }}
    th {{ background: #2f4054; color: white; font-size: 12px; letter-spacing: .04em; text-transform: uppercase; }}
    tr:nth-child(even) td {{ background: #fffaf1; }}
    .dense th, .dense td {{ padding: 9px 10px; font-size: 13px; }}
    .key-table code {{ word-break: break-all; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 99px; font-weight: 800; font-size: 12px; }}
    .pill.ok {{ background: #daf7ef; color: #087466; }}
    .pill.warn {{ background: #ffe2c6; color: #9a3412; }}
    .day-chart {{ width: 100%; height: auto; display: block; }}
    .svg-title {{ font: 800 16px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #24313f; }}
    .svg-label {{ font: 800 12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .svg-note {{ font: 700 12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #677386; }}
    .footer {{ margin-top: 34px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 860px) {{
      .summary, .grid-2 {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-value {{ text-align: left; }}
      .page {{ padding: 20px 12px 36px; }}
    }}
    @media print {{
      body {{ background: white; }}
      .page {{ max-width: none; }}
      .panel, .hero, .metric-card {{ break-inside: avoid; box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">P053 final closeout evidence</div>
      <h1>S3 Artifacts Inventory Report</h1>
      <p>This static report captures the retained evidence in <code>s3://{escape(bucket)}</code>, the post-Day40 schedule leak, model artifacts, object versions, and the remaining AWS cost posture after the project was stopped.</p>
      <p><strong>Generated:</strong> {generated_at} &nbsp; <strong>Region:</strong> {escape(region)} &nbsp; <strong>Workflow hard-stop:</strong> commit <code>{WORKFLOW_FIX_COMMIT}</code>.</p>
      <div class="summary">{summary_cards}</div>
    </section>

    <section class="callout good">
      <strong>Big-bill status:</strong> EC2 and RDS compute are stopped, NAT gateways are absent, the daily GitHub Actions cron is disabled, and expensive workflow steps are guarded by <code>should_run</code>. This stops the recurring production-run spend path. Small residual charges continue while evidence resources are retained.
    </section>

    <section class="callout">
      <strong>Important incident:</strong> Days 41-45 are real unintended artifacts. The Day 40 completion check used <code>exit 0</code> inside a single GitHub Actions step, so the job continued to AWS-starting steps. The workflow is now disabled remotely and guarded in source.
    </section>

    <section>
      <h2>Pipeline State</h2>
      {state_table}
    </section>

    <section>
      <h2>Color Summary</h2>
      <div class="grid-2">
        <div class="panel">{bar_chart(prefix_rows, 'Current size by top-level S3 prefix')}</div>
        <div class="panel">{bar_chart(type_rows, 'Current size by artifact type')}</div>
      </div>
      <div class="panel" style="margin-top:18px">{day_svg(day_rows)}</div>
    </section>

    <section>
      <h2>Prefix Statistics</h2>
      {prefix_table}
    </section>

    <section>
      <h2>Artifact Type Statistics</h2>
      {type_table}
    </section>

    <section>
      <h2>Day-wise Artifact Inventory</h2>
      <p>Rows show only day-level objects currently present in S3. Days above 40 are intentionally highlighted as accidental extras retained for evidence.</p>
      {day_table}
    </section>

    <section>
      <h2>Model Artifacts</h2>
      {model_table if model_objects else '<p>No model artifacts found in current S3 listing.</p>'}
    </section>

    <section>
      <h2>Largest Current Objects</h2>
      {largest_table}
    </section>

    <section>
      <h2>AWS Resource Cost Posture</h2>
      {resource_table}
    </section>

    <section class="footer">
      <p>Report builder: <code>scripts/build_s3_artifacts_report.py</code>. This file is safe to view offline and does not call AWS after generation.</p>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--db-id", default=DEFAULT_DB_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    objects = list_current_objects(args.bucket, args.region)
    versions, delete_markers = list_versions(args.bucket, args.region)
    state = load_pipeline_state(args.bucket, args.region)
    resources = collect_resource_audit(args.region, args.instance_id, args.db_id)

    html = build_html(
        bucket=args.bucket,
        region=args.region,
        objects=objects,
        versions=versions,
        delete_markers=delete_markers,
        state=state,
        resources=resources,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()