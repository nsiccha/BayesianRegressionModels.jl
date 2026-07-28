# Historical BRM site model inventory

## Outcome

This is the first empirical capability census of the removed historical BRM
catalogue. The old site did not test models: its parse/sample badges were
hard-coded placeholders. The recovered source contains 359 deployed model rows
plus one hidden non-Bayesian baseline.

The inventory keeps four questions separate for every deployed row:

1. What did the catalogue claim, exactly?
2. Does its cited source support that claim?
3. Is there a semantics-preserving translation to current BRM, a BRM kernel,
   or a StanBlocks plate?
4. How far did that exact translation run today?

Passing question 4 never repairs a failure in question 2. Likewise, an
authoritative citation does not imply current compiler support.

## Source and receipt fidelity

The 359 deployed rows contain 344 HTTP model-citation occurrences, 15
synthetic/uncited occurrences, and 125 distinct row-level HTTP URLs. Retrieval
on 2026-07-28 produced 121 HTTP 200 responses and four HTTP 404 responses. The
four dead URLs expand to 17 affected rows. Dataset receipts are a separate
surface and never validate a model formula or family.

Initial conservative row verdicts before the manual second pass were:

| verdict | rows |
|---|---:|
| confirmed by exact normalized formula text | 142 |
| adapted-but-defensible candidate | 44 |
| mismatch | 4 |
| unverifiable | 152 |
| dead-source | 17 |

The four direct mismatches are ActionModels
`pvl_igt_{lr,reward,loss,noise}`. Their cited code loads healthy controls and
declares independent population priors; it does not regress those parameters
on `clinical_group` as the catalogue claims. The matrix carries the correction
without pretending a different source was originally cited.

Exact formula occurrences are strong receipts. The 44 partial/adapted cases
received an additional manual body review rather than trusting token overlap:
12 were confirmed, 10 remained adapted-but-defensible, 21 became mismatches,
and one became unverifiable. Final deployed-row totals are therefore:

| verdict | rows |
|---|---:|
| confirmed | 154 |
| adapted-but-defensible | 10 |
| mismatch | 25 |
| unverifiable | 153 |
| dead-source | 17 |

The additional mismatches are substantive, not punctuation differences. They
include omitted or invented BRMS terms, four Epinowcast module compositions,
seven INLA dataset/formula substitutions, two mvgam portal state models, and
two rstanarm gasoline beta models. Every correction and bounded body anchor is
row-keyed in `receipts/manual_review.tsv` and expanded into the matrix.

## Family metadata defect

Only 21 deployed rows explicitly declare `family:`. The historical renderer
silently labels every absent field Gaussian, producing 350 Gaussian, four
binomial, two Bernoulli, two Poisson, and one negative-binomial card. This is a
real presentation defect, not a lowering convention.

The matrix therefore carries three independent fields:

- exact family metadata (possibly empty),
- the renderer claim and its `CatalogServer_default_unsubstantiated`
  provenance, and
- a separately inferred semantic family with its evidence class.

Exact-metadata and inferred-family translations were validated separately.
WarmupHMC's independent HTML pass found 33 formulas that visibly contradict a
Gaussian badge; that is a useful strict lower bound. Across all 338 defaulted
rows, this semantic pass infers 142 non-Gaussian families, 74 Gaussian families,
and leaves 122 unresolved. Because those 142 are inferences rather than source
metadata, the matrix preserves both numbers and provenance instead of replacing
the badge silently.

The defect is also an operational selection hazard, not only a display issue.
An independent WarmupHMC catalogue benchmark found that tag-based Gaussian
selection would incorrectly admit `cbpp` (binomial), `epilepsy` (count), and
`kidney` (censored survival), while explicitly tagged `contraception` and
`verbagg` remain correct. That independently localizes the defect to the
missing-metadata default rather than to every explicit tag.

The sole verified card, `bambi/escs`, also exposed a peer parser edge: its HTML
class is `card card-verified`, so an exact `class="card"` matcher counted 358.
A class-token matcher and the source both count 359.

## Translation to current semantics

The translation deliberately follows current BRM rather than mechanically
modernizing old brms syntax:

- `I(expr)` → `protect(expr)`;
- `scale(x)` → `zscale(x)`;
- simple source `a:b` → current raw-column `a & b`;
- simple source `a*b` → `a + b + (a & b)`;
- `S(x)`/`C(x)` → raw `x` plus an integer/categorical dtype requirement;
- source `||`/`zerocorr(...)` → current `||`;
- grouped/binomial responses → explicit `BinomialLogit`;
- every accepted likelihood is explicit, including its link and
  parameterization.

Transform interactions, chained interactions, composite group identifiers,
multi-formula models, and rich historical spline/HSGP signatures are not
papered over. They remain explicit gaps.

Route classification over the 359 rows is:

| route | rows | meaning |
|---|---:|---|
| `ordinary_brm` | 316 | ordinary current descriptor candidate |
| `brm_kernel` | 4 | faithful group-local/ragged cell requires a row-specific kernel |
| `stanblocks_plate` | 39 | structured latent/covariance/time-series cell requires a plate |

In the inferred-family variant, 111 rows have executable ordinary BRM bodies,
14 have both a known family and a route-specific design, 122 remain
family-unresolved, and 112 are explicit unsupported translations. These
categories are exhaustive and row-keyed in `model_matrix.tsv`.

The most frequent unresolved/unsupported reasons are:

- family unresolved: 122;
- multi-formula/nonlinear brms declaration: 18;
- multiple component formulas requiring a joint declaration: 11;
- nested/composite grouping needing a derived id or reviewed `gr(...; by=...)`: 10;
- multi-axis or multi-argument old HSGP: 9;
- non-mechanical term, NegativeBinomial parameterization boundary,
  censor/truncation, or non-default spline semantics: 7 each.

## Current compiler and runtime evidence

The 111 executable inferred-family rows and 15 executable exact-metadata rows
collapse to 103 unique normalized bodies/data schemas. The static gate runs
BRMI evaluation, SBBRMI lowering, `brm_descriptor`, `brm_execute(:transpile)`,
and stanc. Ninety-eight unique programs pass stanc. Each of those 98 also
instantiates under BridgeStan 2.9.0 and has finite log density and gradient at
the synthetic-data zero point; no stanc-accepted program fails the runtime
gate. Byte-identical fan-out maps those 98 programs to 106 deployed rows. These
results are capability evidence, not posterior-correctness claims.

The five static failures are one concrete current substrate seam: grouped
BinomialLogit translations fail generated-quantity tracing at
`binomial_logit_rng(::array[] tokenof, ::int, ::vector)`. They are
`mcelreath/{chimpanzees_intercept,chimpanzees_slopes,moralizing_gods}` and
`kruschke/{recall_conditions,recall_pooled}`.

An earlier probe version bound vector-valued Beta shape expressions to
intermediate names and exposed a missing `lpxf_expr`. Re-checking the native
current idiom showed that keeping those expressions inline is accepted; the
translator was corrected and all six Beta rows now pass static validation.

These are not historical-source verdicts and are not extrapolated to merely
similar rows. Full BridgeStan and sampling results are recorded in
`capability_results.tsv`, `runtime_controls.tsv`, and the per-row matrix.

## Real-data and structured-route controls

The real-data control uses the historical sleepstudy model on the official
Rdatasets `lme4/sleepstudy` data (180 rows, 18 subjects). For direct comparison
with the peer control, `Reaction` is divided by 100 and raw subject ids are
densely recoded. The scaling is a capability-test choice, not a catalogue
claim or a validated historical fit.

On BRM `784712998ea67f6429d0a3b5a3241fe9cb690e64`, StanBlocks
`329a178a7ad7877da0b58ad2c360d417ddd663f9`, and current-host WarmupHMC
`b185eedbbeeef6fb3327afb30dc995c98591af02`, that control reproduces the peer
dimension (42) and zero-point log density (`-832.3603659550055`), has a finite
gradient, and completes 100 WarmupHMC draws with zero divergences. The peer's
stronger sampling receipt—500 draws across 12 seeds with zero divergences—used
WarmupHMC `38398527e0406ad31aeaec3efe24d581a18a269e` and remains a separate
control rather than evidence inherited by catalog rows.

Two additional controls are kept deliberately orthogonal to the row evidence:

- a BRM `kernel(...) do ... end` ragged group-local model derived from
  `test/descriptor.jl`;
- a fixed-width correlated StanBlocks `plate` model derived from
  `test/plate_stress.jl`.

They pass or fail as route controls only. No historical row inherits them
without an exact row-specific implementation. Both controls pass stanc,
BridgeStan instantiation, finite density/gradient, and WarmupHMC sampling: the
kernel is dimension 15 with 50 draws and zero divergences; the plate is
dimension 21 with 50 draws and zero divergences.

## Current SbPMX semantic mount

The coherent baseline is SbPMX `origin/main@a8fd3c0`, which contains every
advertised semantic branch and is three commits beyond the recorded deployed
release `88c4fbf`. Exact paths and blob ids are in `PROVENANCE.md`.

The translation target is not old page syntax. It is the descriptor-driven
pattern exercised by current `design/SbPMX.jl` and
`compat/check-semantic-model-pipeline.jl`:

- one `semantic_app` graph;
- `@options` backed by `option_domain`/`option_records`;
- semantic cards/nodes with HTML, Markdown, Hyperview, and text peers;
- descriptor operations projected as actions/forms;
- ordinary navigation URLs, artifacts, and disclosures.

`@semantic` and `dynamic_domain` are retired. The primer-recorded
`APPDATA.built(:entry)` 13-card finite density/gradient gate, reflected in the
current semantic pipeline compatibility check, is a reference control—not
evidence for these 359 translations.

BRM's descriptor is authoritative, but the adapter from
`BRMDescriptor.operations + columns` to the HTMXObjects mount is not shipped.
Every matrix mount recommendation is therefore marked experimental. A future
adapter should consume the descriptor; it should not re-parse formula text or
create a second declaration graph.

## What remains genuinely open

- Resolve or source-correct the 122 rows whose response family cannot be
  defended from exact metadata and primary evidence.
- Implement row-specific kernel/plate translations for the 43 structured-route
  candidates; representative substrate controls are not substitutes.
- Address the BinomialLogit generated-quantity seam above before claiming the
  five currently failing ordinary translations.
- Decide semantics for unsupported joint, multivariate, categorical,
  zero-inflated/hurdle, censored, survival, and historical smooth/GP models.
- Ship the BRMDescriptor→semantic_app adapter, keeping the descriptor as the
  single authoritative graph.

No server was started or restarted for this inventory.
