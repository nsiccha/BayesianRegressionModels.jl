#!/usr/bin/env python3
"""Recover the 05c3f46 catalog metadata and fetch distinct model citations."""

from __future__ import annotations

import concurrent.futures
import argparse
import csv
import hashlib
import json
import pathlib
import re
import subprocess
from datetime import datetime, timezone


parser = argparse.ArgumentParser()
parser.add_argument(
    "--examples-root", type=pathlib.Path, required=True,
    help="scripts/examples directory from BayesianRegressionModels.jl@05c3f465e7987e8d7caa7e214fedddd90415a922",
)
parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).parent)
args = parser.parse_args()
ROOT = args.examples_root.resolve()
OUT = args.out.resolve()
BODIES = OUT / "bodies"
BODIES.mkdir(parents=True, exist_ok=True)


def metadata(doc: str) -> tuple[dict[str, str], str]:
    head, _, desc = doc.partition("\n----")
    result: dict[str, str] = {}
    for line in head.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"')
    return result, desc.strip()


catalog_text = (ROOT / "all.jl").read_text()
catalog = re.findall(r"\(source=:(\w+),\s*key=:(\w+)\)", catalog_text)
examples: dict[tuple[str, str], dict[str, str]] = {}
datasets: dict[tuple[str, str], list[dict[str, str]]] = {}

for path in sorted(ROOT.glob("*.jl")):
    if path.name == "all.jl":
        continue
    text = path.read_text()
    # A docstring immediately attached to examples(::Val{:key}).
    pattern = re.compile(
        r'"""\s*\n(?P<doc>(?:(?!""").)*)\n"""\s*\n'
        r'(?:function\s+)?examples\(::Val\{:(?P<key>\w+)\}\)', re.S
    )
    for match in pattern.finditer(text):
        meta, desc = metadata(match.group("doc"))
        meta["description"] = desc
        meta["source_file"] = path.name
        examples[(path.stem, match.group("key"))] = meta

    # Dataset docstrings may attach to one-line or function-form load methods.
    load_pattern = re.compile(
        r'"""\s*\n(?P<doc>(?:(?!""").)*)\n"""\s*\n'
        r'(?:function\s+)?load\(::Val\{:(?P<key>\w+)\}\)', re.S
    )
    for match in load_pattern.finditer(text):
        meta, desc = metadata(match.group("doc"))
        meta["description"] = desc
        datasets.setdefault((path.stem, match.group("key")), []).append(meta)


rows: list[dict[str, str]] = []
for ordinal, (source, key) in enumerate(catalog, 1):
    meta = examples[(source, key)]
    dataset = meta.get("dataset", "")
    data_matches = datasets.get((source, dataset), [])
    data_urls = sorted({m.get("source", "") for m in data_matches if m.get("source")})
    desc_family = ""
    family_match = re.search(r"\bFamily:\s*([^\.\n]+)", meta.get("description", ""), re.I)
    if family_match:
        desc_family = family_match.group(1).strip()
    rows.append({
        "ordinal": str(ordinal),
        "row_key": f"{source}/{key}",
        "catalog_source": source,
        "catalog_key": key,
        "deployed": "false" if meta.get("hidden", "").lower() == "true" else "true",
        "name": meta.get("name", ""),
        "example": meta.get("example", ""),
        "dataset": dataset,
        "formula": meta.get("formula", ""),
        "family_explicit": meta.get("family", ""),
        "family_description_claim": desc_family,
        "claimed_source_url": meta.get("source", ""),
        "dataset_source_urls": " | ".join(data_urls),
        "source_file": meta["source_file"],
        "hidden_claim": meta.get("hidden", ""),
        "verified_claim": meta.get("verified", ""),
        "description": meta.get("description", "").replace("\t", " ").replace("\n", " "),
    })


with (OUT / "catalog_claims.tsv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

model_urls = {r["claimed_source_url"] for r in rows if r["claimed_source_url"].startswith("http")}
dataset_urls = {
    url
    for r in rows
    for url in r["dataset_source_urls"].split(" | ")
    if url.startswith("http")
}
all_doc_source_strings = {
    value.strip()
    for path in ROOT.glob("*.jl") if path.name != "all.jl"
    for value in re.findall(r"^source:\s*(.*?)\s*$", path.read_text(), re.M)
    if value.strip()
}
all_doc_http_urls = {url for url in all_doc_source_strings if url.startswith("http")}
unmapped_doc_sources = sorted(all_doc_source_strings - model_urls - dataset_urls)
(OUT / "unmapped_doc_sources.txt").write_text("\n".join(unmapped_doc_sources) + "\n")
urls = sorted(model_urls | dataset_urls | all_doc_http_urls)


def fetch(url: str) -> dict[str, object]:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    body = BODIES / f"{digest}.body"
    headers = BODIES / f"{digest}.headers"
    started = datetime.now(timezone.utc).isoformat()
    fmt = "\nKBMETA\t%{http_code}\t%{url_effective}\t%{content_type}\t%{size_download}\t%{num_redirects}\n"
    proc = subprocess.run([
        "curl", "-sS", "-L", "--compressed", "--connect-timeout", "12", "--max-time", "45",
        "--max-filesize", "8000000", "-A", "Mozilla/5.0 BRM-receipt-audit/1.0",
        "-D", str(headers), "-o", str(body), "-w", fmt, url,
    ], text=True, capture_output=True)
    finished = datetime.now(timezone.utc).isoformat()
    marker = ""
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith("KBMETA\t"):
            marker = line
            break
    parts = marker.split("\t") if marker else []
    status = parts[1] if len(parts) > 1 else "000"
    final_url = parts[2] if len(parts) > 2 else ""
    content_type = parts[3] if len(parts) > 3 else ""
    size = parts[4] if len(parts) > 4 else "0"
    redirects = parts[5] if len(parts) > 5 else "0"
    return {
        "url": url, "url_hash": digest, "retrieved_at": started, "finished_at": finished,
        "curl_exit": proc.returncode, "http_status": status, "final_url": final_url,
        "content_type": content_type, "size_download": size, "redirects": redirects,
        "curl_error": proc.stderr.strip().replace("\t", " ").replace("\n", " | "),
        "body_path": str(body.relative_to(OUT)) if body.exists() else "",
        "headers_path": str(headers.relative_to(OUT)) if headers.exists() else "",
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
    fetched = list(pool.map(fetch, urls))

with (OUT / "url_fetches.tsv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(fetched[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(fetched)

(OUT / "url_fetches.json").write_text(json.dumps(fetched, indent=2))
print(json.dumps({
    "catalog_rows": len(rows),
    "deployed_rows": sum(r["deployed"] == "true" for r in rows),
    "hidden_rows": [r["row_key"] for r in rows if r["deployed"] == "false"],
    "http_rows": sum(r["claimed_source_url"].startswith("http") for r in rows),
    "synthetic_rows": sum(r["claimed_source_url"] == "synthetic" for r in rows),
    "distinct_model_http_urls": len(model_urls),
    "distinct_dataset_http_urls": len(dataset_urls),
    "distinct_all_http_urls": len(urls),
    "distinct_all_doc_source_strings": len(all_doc_source_strings),
    "unmapped_doc_source_strings": unmapped_doc_sources,
    "fetched_2xx": sum(str(x["http_status"]).startswith("2") for x in fetched),
    "fetch_errors": sum(x["curl_exit"] != 0 for x in fetched),
}, indent=2))
