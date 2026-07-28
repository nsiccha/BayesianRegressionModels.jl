# Manual second pass of deployed adapted verdicts

Input: `row_receipt_audit.tsv` from the same receipt capture.

Scope: all 44 deployed rows whose first-pass semantic-support verdict was `adapted-but-defensible`. The one non-deployed row with that verdict (`epinowcast/bnc_empirical`) is intentionally excluded.

Manual counts: confirmed=12; adapted-but-defensible=10; mismatch=21; unverifiable=1; dead-source=0.

The TSV preserves each exact catalog formula, cited/final URL, retrieval timestamp/status, content type, and cached body SHA-256, then records a manual evidence anchor, support rationale, and correction. Review used the bodies captured by the first-pass audit; no source was credited merely for returning HTTP 200.
