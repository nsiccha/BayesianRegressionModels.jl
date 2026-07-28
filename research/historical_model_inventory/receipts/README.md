# Historical BRM catalog receipt audit

Source revision: `BayesianRegressionModels.jl@05c3f465e7987e8d7caa7e214fedddd90415a922`, recovered from `scripts/examples/*.jl` and `scripts/examples/all.jl`.

Retrieval window: 2026-07-28 21:07:08–21:07:15 UTC. Each HTTP receipt was fetched with redirects enabled, a 45-second timeout, and an 8 MB body bound. Full response metadata is keyed by the SHA-256 prefix of the requested URL. Response bodies were retained for the audit and manual second pass but are not committed; row evidence preserves their SHA-256 hashes, and the retrieval script reconstructs the `bodies/` layout.

## Census corrections

- The raw `CATALOG` contains 360 rows. `epinowcast/bnc_empirical` is the sole row whose nearest attached docstring has `hidden: true`; therefore the deployed set is 359 rows.
- The deployed 359 rows contain 344 HTTP model-citation occurrences, 15 `source: synthetic` occurrences, and 125 distinct HTTP model URLs. Distinct model-URL retrieval status: 121 HTTP 200, four HTTP 404.
- All exact `source:` strings in every example-file docstring number 207, not 208: 206 HTTP strings plus `synthetic`. Three HTTP strings are attached to loader-only datasets not reached by a catalog row's primary dataset key (`HtWtData30.csv`, `SoftmaxRegData2.csv`, and auxiliary `BTped.csv`).
- There are 113 distinct row-associated dataset URLs. They are audited separately from model support.
- Only 21 deployed rows explicitly declare `family:` metadata. `CatalogServer` silently renders the other 338 as Gaussian, yielding the observed display count of 350 Gaussian, four binomial, two Bernoulli, two Poisson, and one negative-binomial. Every row records this renderer default as `CatalogServer_default_unsubstantiated`; it is not treated as source evidence.

## First-pass deployed-row verdicts

| verdict | rows | criterion |
|---|---:|---|
| `confirmed` | 142 | The normalized claimed formula text occurs in the retrieved cited source. |
| `adapted-but-defensible` | 44 | Exact syntax differs, but the cited project/author source exposes the distinctive response/predictor/model terms and relevant dataset/name context. |
| `mismatch` | 4 | Manual contradiction: the four ActionModels PVL rows claim `clinical_group` regressions, while the cited code loads only healthy controls and defines independent population priors. |
| `unverifiable` | 152 | Conservative bucket: 84 data-file-only model citations, 15 synthetic/uncited rows, or insufficient page/PDF/root-page evidence. |
| `dead-source` | 17 | Four distinct cited URLs returned HTTP 404; repeated rows expand those failures to 17. |

The four dead URLs have replacement candidates where possible in `correction_with_evidence`: the current glmmTMB site, MCMCglmm r-universe course notes, and the current official MCMCglmm manual were each checked reachable (HTTP 200) on 2026-07-28. A replacement candidate is not automatically promoted to semantic confirmation.

All 44 deployed first-pass `adapted-but-defensible` rows then received a manual
body review. Twelve became confirmed, 10 remained adapted, 21 became mismatch,
and one became unverifiable. The assembled matrix applies those overrides, for
final totals of 154 confirmed, 10 adapted, 25 mismatch, 153 unverifiable, and
17 dead-source. The first-pass TSV remains unchanged so the audit trail is
reproducible; `manual_review.tsv` is the explicit second layer.

## Method

1. Parse the ordered `CATALOG` mechanically and attach only the immediately preceding docstring to each `examples(::Val{:key})`; this avoids accidentally reading loader metadata as model metadata.
2. Attach loader docstrings independently to the catalog row's dataset key. Preserve other source strings as unmapped loader evidence.
3. Fetch each distinct HTTP URL once. Store requested/final URL, UTC timestamp, HTTP and curl status, content type, redirects, byte count, retained body, and headers.
4. Strip HTML navigation/scripts for matching. `confirmed` requires the full normalized formula to occur. `adapted-but-defensible` requires at least 60% of distinctive formula identifiers plus dataset/name or additional formula context. Anything weaker remains `unverifiable`; reachability alone never confirms a model.
5. Evidence fields contain reproducible text offsets, matched terms, and bounded context hashes rather than copied prose. Model and dataset receipts remain orthogonal.

This is deliberately conservative. PDF landing pages not exposing formula text, root book pages without a stable model anchor, data-only URLs, and synthetic claims are not guessed into confirmation. The automated exact/term criteria are visible in the TSV and can be tightened or manually reviewed without repeating retrieval.

## Files

- `row_receipt_audit.tsv` — all 360 source rows, including deployed flag; exact claim, retrieval evidence, authority class, formula/family/dataset support, correction, and verdict.
- `manual_review.tsv` / `manual_review.md` — manual second pass over all 44
  deployed first-pass `adapted-but-defensible` rows. These verdicts override
  the token-overlap classification in the assembled matrix.
- `row_dataset_receipts.tsv` — row-expanded loader/dataset receipts with the explicit boundary that they do not validate formula/family.
- `summary_by_source.tsv` — deployed verdict counts for each of the 19 catalog sources.
- `catalog_claims.tsv` — lossless normalized extraction before retrieval/classification.
- `url_fetches.tsv` — one record for each of 206 distinct HTTP source strings.
- `unmapped_doc_sources.txt` — docstring sources not attached to a catalog row's primary model/dataset receipt.
- `bodies/` — generated on a rerun, not committed; response bodies and headers
  are keyed by URL hash prefix. `url_fetches.tsv` records their relative names.
- `audit_receipts.py` / `classify_receipts.py` — reproducible extraction, retrieval, and classification scripts.

Reproduce against an exact checkout of the historical commit with:

```sh
python3 audit_receipts.py \
  --examples-root /path/to/BayesianRegressionModels.jl/scripts/examples \
  --out /path/to/audit-output
python3 classify_receipts.py --out /path/to/audit-output
```

Before running, verify that the supplied checkout resolves to
`05c3f465e7987e8d7caa7e214fedddd90415a922`. The scripts intentionally do not
guess or fetch a revision. HTTP evidence is time-sensitive, so a rerun should
be preserved as a new retrieval receipt rather than treated as byte-identical
to this 2026-07-28 capture.
