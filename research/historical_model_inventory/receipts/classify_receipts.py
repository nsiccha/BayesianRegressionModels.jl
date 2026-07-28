#!/usr/bin/env python3
"""Attach conservative semantic-support evidence to every recovered catalog row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import pathlib
import re
from urllib.parse import urlparse


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).parent)
args = parser.parse_args()
OUT = args.out.resolve()


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def plain_text(body: bytes, content_type: str) -> str:
    text = body.decode("utf-8", "replace")
    if "html" in content_type or "<html" in text[:1000].lower():
        parser = TextExtractor()
        try:
            parser.feed(text)
            text = " ".join(parser.parts)
        except Exception:
            pass
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


STOP = {
    "c", "i", "t", "y", "x", "n", "mu", "sigma", "alpha", "beta", "true", "false",
    "bs", "gp", "hsgp", "poly", "offset", "mo", "mi", "s", "f", "log", "exp", "sqrt",
    "weibull", "gaussian", "poisson", "binomial", "bernoulli", "gamma", "student", "family",
}


def tokens(value: str) -> list[str]:
    result = []
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value.lower()):
        if len(tok) >= 3 and tok not in STOP and tok not in result:
            result.append(tok)
    return result


def family_tokens(value: str) -> list[str]:
    return [
        token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value.lower())
        if len(token) >= 2 and token not in {"link", "family", "true", "false"}
    ]


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", html.unescape(value).lower())


OFFICIAL_OR_PRIMARY_DOMAINS = {
    "bambinos.github.io", "paulbuerkner.com", "mc-stan.org", "package.epinowcast.org",
    "baselinenowcast.epinowcast.org", "epidist.epinowcast.org", "jsocolar.github.io",
    "nicholasjclark.github.io", "juliastats.org", "bruno.nicenboim.me",
    "cran.r-project.org", "doi.org", "jse.amstat.org", "github.com",
    "becarioprecario.bitbucket.io", "venpopov.github.io",
}


def authority_class(url: str, data_only: bool) -> str:
    if not url or url == "synthetic":
        return "no_external_authority"
    if data_only:
        return "dataset_repository_or_file_only"
    domain = urlparse(url).netloc.lower()
    if domain in OFFICIAL_OR_PRIMARY_DOMAINS:
        return "cited_project_author_publisher_surface"
    return "cited_surface_not_independently_authenticated"


def evidence_anchor(text: str, needles: list[str]) -> str:
    low = text.lower()
    positions = [(low.find(n.lower()), n) for n in needles if len(n) >= 3 and low.find(n.lower()) >= 0]
    if not positions:
        return ""
    # Prefer the occurrence with the densest cluster of other matched terms nearby.
    pos, needle = max(
        positions,
        key=lambda item: sum(abs(item[0] - other[0]) <= 700 for other in positions),
    )
    lo = max(0, pos - 350)
    hi = min(len(text), pos + len(needle) + 350)
    context = text[lo:hi].encode()
    nearby = sorted({n for p, n in positions if abs(p - pos) <= 700})
    # Evidence is an anchor, not a copyrighted quotation. The bounded context hash lets
    # another auditor reproduce which occurrence was used from the retained body.
    return f"text_offset={pos}; matched_terms={','.join(nearby)}; context_sha256={hashlib.sha256(context).hexdigest()[:16]}"


fetches = {r["url"]: r for r in csv.DictReader((OUT / "url_fetches.tsv").open(), delimiter="\t")}
claims = list(csv.DictReader((OUT / "catalog_claims.tsv").open(), delimiter="\t"))
result = []


def retained_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else OUT / path

for row in claims:
    url = row["claimed_source_url"]
    fetch = fetches.get(url)
    status = fetch["http_status"] if fetch else ""
    ctype = fetch["content_type"] if fetch else ""
    body_path = retained_path(fetch["body_path"]) if fetch and fetch["body_path"] else None
    body = body_path.read_bytes() if body_path is not None and body_path.exists() else b""
    text = plain_text(body, ctype)
    low = text.lower()
    formula = row["formula"]
    f_tokens = tokens(formula)
    present = [t for t in f_tokens if re.search(rf"\b{re.escape(t)}\b", low)]
    missing = [t for t in f_tokens if t not in present]
    norm_formula = normalized(formula)
    norm_source = normalized(text)
    exact = bool(norm_formula and len(norm_formula) >= 8 and norm_formula in norm_source)

    dataset_tokens = tokens(row["dataset"])
    dataset_present = [t for t in dataset_tokens if re.search(rf"\b{re.escape(t)}\b", low)]
    name_tokens = tokens(row["name"])
    name_present = [t for t in name_tokens if re.search(rf"\b{re.escape(t)}\b", low)]
    family = row["family_explicit"] or row["family_description_claim"]
    claimed_family_tokens = family_tokens(family)
    family_present = [t for t in claimed_family_tokens if re.search(rf"\b{re.escape(t)}\b", low)]

    path = urlparse(url).path.lower()
    sample = text[:3000]
    comma_lines = sum(line.count(",") >= 2 for line in sample.splitlines())
    data_only = (
        "text/csv" in ctype or path.endswith((".csv", ".dat", ".tsv", ".txt"))
        or ("application/octet-stream" in ctype and comma_lines >= 2)
        or (comma_lines >= 4 and "<html" not in body[:1000].decode("utf-8", "ignore").lower())
    )

    formula_support = "not_evaluable"
    family_support = "not_evaluable"
    dataset_support = "not_evaluable"
    verdict = "unverifiable"
    reason = ""
    correction = ""
    evidence = ""

    if url == "synthetic" or not url:
        reason = "No external model citation: catalog claim is synthetic/uncited."
        dataset_support = "synthetic_claim_only"
    elif not fetch or not status.startswith("2") or fetch["curl_exit"] not in {"0", "63"}:
        verdict = "dead-source"
        reason = f"Citation retrieval failed (HTTP {status or 'none'}, curl {fetch['curl_exit'] if fetch else 'n/a'})."
    elif data_only:
        formula_support = "not_supported_by_data_file"
        family_support = "not_supported_by_data_file"
        dataset_support = "reachable_data_file"
        verdict = "unverifiable"
        reason = "The cited resource is a data file; it can support dataset provenance but not the claimed model formula or family."
        evidence = evidence_anchor(text, dataset_tokens + name_tokens)
    elif exact:
        formula_support = "exact_text_present"
        dataset_support = "name_or_key_present" if dataset_present or name_present else "not_observed"
        family_support = "explicit_token_present" if family_present else ("not_claimed" if not family else "not_observed")
        verdict = "confirmed"
        reason = "The retrieved official/primary source contains the normalized claimed formula text."
        evidence = evidence_anchor(text, present + dataset_present + name_present)
    else:
        ratio = len(present) / len(f_tokens) if f_tokens else 0.0
        if len(present) >= 2 and ratio >= 0.60 and (dataset_present or name_present or len(present) >= 3):
            formula_support = f"partial_semantic_terms:{len(present)}/{len(f_tokens)}"
            dataset_support = "name_or_key_present" if dataset_present or name_present else "not_observed"
            family_support = "explicit_token_present" if family_present else ("not_claimed" if not family else "not_observed")
            verdict = "adapted-but-defensible"
            reason = "The source contains the model's distinctive response/predictor terms but not the exact catalog formula; the catalog expression is an adaptation."
            evidence = evidence_anchor(text, present + dataset_present + name_present)
        else:
            formula_support = f"insufficient_terms:{len(present)}/{len(f_tokens)}"
            dataset_support = "name_or_key_present" if dataset_present or name_present else "not_observed"
            family_support = "explicit_token_present" if family_present else ("not_claimed" if not family else "not_observed")
            verdict = "unverifiable"
            reason = "Retrieved source does not expose enough formula-specific evidence to confirm or refute the catalog claim conservatively."
            evidence = evidence_anchor(text, present + dataset_present + name_present)

    # The cited ActionModels PVL source is directly contradictory: it reads only the
    # healthy-control file and gives four independent population priors, not regressions
    # on clinical_group. The source does use subjID as session_cols, which does not make
    # the claimed group-level formula valid.
    if row["row_key"] in {
        "action_models/pvl_igt_lr", "action_models/pvl_igt_reward",
        "action_models/pvl_igt_loss", "action_models/pvl_igt_noise",
    } and status.startswith("2"):
        verdict = "mismatch"
        formula_support = "contradicted_by_cited_source"
        dataset_support = "healthy_control_subset_only"
        reason = "Cited code loads only healthy controls and defines independent population priors; it does not regress any parameter on clinical_group."
        correction = "Represent the cited code as independent LogitNormal/LogNormal population priors, or cite a different source that actually fits clinical_group effects."
        evidence = "source anchors: IGTdata_healthy_control.txt; population_model; session_cols=:subjID"

    # The cited book-root response exposes navigation/TOC text, not a stable row-level
    # model anchor. Generic term overlap (notably `effect` and `resp_se`) is insufficient.
    if url == "https://bruno.nicenboim.me/bayescogsci/" and not exact:
        verdict = "unverifiable"
        formula_support = f"insufficient_root_page_terms:{len(present)}/{len(f_tokens)}"
        reason = "Cited book root exposes navigation/TOC text but no stable row-level formula anchor; term overlap is insufficient."

    replacement_map = {
        "https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html":
            "Dead CRAN path has a reachable official replacement candidate (HTTP 200 checked 2026-07-28): https://glmmtmb.github.io/glmmTMB/articles/glmmTMB.html",
        "https://jarrodhadfield.github.io/MCMCglmm/course-notes/":
            "Dead GitHub Pages path has a reachable author/package replacement (HTTP 200 checked 2026-07-28): https://jarrodhadfield.r-universe.dev/articles/MCMCglmm/CourseNotes.html",
        "https://cran.r-project.org/web/packages/MCMCglmm/vignettes/Overview.pdf":
            "Dead vignette path has a reachable official package manual (HTTP 200 checked 2026-07-28): https://stat.ethz.ch/CRAN/web/packages/MCMCglmm/MCMCglmm.pdf",
        "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv":
            "Dead data mirror path; the current official MCMCglmm manual documents BTdata at https://stat.ethz.ch/CRAN/web/packages/MCMCglmm/MCMCglmm.pdf, but that still cannot validate these model formulas.",
    }
    if verdict == "dead-source" and url in replacement_map:
        correction = replacement_map[url]

    deployed_family = row["family_explicit"] or "gaussian"
    family_provenance = "explicit_metadata" if row["family_explicit"] else "CatalogServer_default_unsubstantiated"

    result.append({
        **{k: row[k] for k in [
            "ordinal", "row_key", "catalog_source", "catalog_key", "deployed", "name", "example",
            "dataset", "formula", "family_explicit", "family_description_claim", "claimed_source_url",
            "dataset_source_urls", "source_file", "hidden_claim", "verified_claim",
        ]},
        "deployed_renderer_family_claim": deployed_family,
        "deployed_family_provenance": family_provenance,
        "retrieved_at": fetch["retrieved_at"] if fetch else "",
        "http_status": status,
        "curl_exit": fetch["curl_exit"] if fetch else "",
        "final_url": fetch["final_url"] if fetch else "",
        "content_type": ctype,
        "citation_authority_class": authority_class(url, data_only),
        "data_only_citation": str(data_only).lower(),
        "formula_exact_observed": str(exact).lower(),
        "formula_terms_observed": ",".join(present),
        "formula_terms_missing": ",".join(missing),
        "formula_support": formula_support,
        "family_support": family_support,
        "dataset_support": dataset_support,
        "semantic_support_verdict": verdict,
        "semantic_support_reason": reason,
        "correction_with_evidence": correction,
        "evidence_anchor": evidence,
        "body_sha256": hashlib.sha256(body).hexdigest() if body else "",
    })

with (OUT / "row_receipt_audit.tsv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(result[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(result)

summary_rows = []
for source in sorted({r["catalog_source"] for r in result}):
    source_rows = [r for r in result if r["catalog_source"] == source and r["deployed"] == "true"]
    counts = {name: sum(r["semantic_support_verdict"] == name for r in source_rows) for name in [
        "confirmed", "adapted-but-defensible", "mismatch", "unverifiable", "dead-source",
    ]}
    summary_rows.append({
        "catalog_source": source, "deployed_rows": len(source_rows), **counts,
        "model_http_rows": sum(r["claimed_source_url"].startswith("http") for r in source_rows),
        "distinct_model_http_urls": len({r["claimed_source_url"] for r in source_rows if r["claimed_source_url"].startswith("http")}),
        "data_only_model_citations": sum(r["data_only_citation"] == "true" for r in source_rows),
        "synthetic_or_uncited": sum(not r["claimed_source_url"].startswith("http") for r in source_rows),
    })
with (OUT / "summary_by_source.tsv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(summary_rows)

# Dataset provenance is a separate receipt surface. A dataset URL can establish only
# reachability/identity; it is never promoted into formula/family support.
dataset_receipts = []
for row in claims:
    for url in [x for x in row["dataset_source_urls"].split(" | ") if x]:
        fetch = fetches.get(url)
        status = fetch["http_status"] if fetch else ""
        body_path = retained_path(fetch["body_path"]) if fetch and fetch["body_path"] else None
        body = body_path.read_bytes() if body_path is not None and body_path.exists() else b""
        ctype = fetch["content_type"] if fetch else ""
        text = plain_text(body, ctype)
        if url == "synthetic":
            receipt_status = "synthetic-no-external-receipt"
        elif not fetch or not status.startswith("2") or fetch["curl_exit"] not in {"0", "63"}:
            receipt_status = "dead-source"
        elif fetch["curl_exit"] == "63":
            receipt_status = "reachable-body-exceeded-8MB-audit-bound"
        else:
            receipt_status = "reachable-data-or-description"
        dataset_receipts.append({
            "row_key": row["row_key"], "deployed": row["deployed"], "dataset": row["dataset"],
            "dataset_source_url": url, "retrieved_at": fetch["retrieved_at"] if fetch else "",
            "http_status": status, "curl_exit": fetch["curl_exit"] if fetch else "",
            "final_url": fetch["final_url"] if fetch else "", "content_type": ctype,
            "dataset_receipt_status": receipt_status,
            "evidence_anchor": evidence_anchor(text, tokens(row["dataset"]) + tokens(row["name"])),
            "body_sha256": hashlib.sha256(body).hexdigest() if body else "",
            "model_support_boundary": "Dataset receipt is not evidence for formula or family.",
        })

with (OUT / "row_dataset_receipts.tsv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(dataset_receipts[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(dataset_receipts)

print("rows", len(result))
from collections import Counter
print("all", Counter(r["semantic_support_verdict"] for r in result))
print("deployed", Counter(r["semantic_support_verdict"] for r in result if r["deployed"] == "true"))
print("exact", sum(r["formula_exact_observed"] == "true" for r in result))
print("data_only", sum(r["data_only_citation"] == "true" for r in result))
