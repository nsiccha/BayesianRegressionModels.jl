# Provenance ledger

## Historical catalogue

- Source revision: BayesianRegressionModels.jl
  `05c3f465e7987e8d7caa7e214fedddd90415a922` (`origin/main`).
- Recovered paths: `scripts/examples/all.jl` and every
  `scripts/examples/*.jl` method named by its ordered `CATALOG`.
- Source census: 360 rows, of which `epinowcast/bnc_empirical` alone carries
  `hidden: true`; the historical site surface therefore contains 359 models.
- HTML control: snapshot retrieved 2026-07-28 21:00:07 UTC, 1,241,917 bytes,
  SHA-256
  `51e16039b24cb806cff516481ef812fa3c081511981c46c269d3f01ca93600e0`.
  It contains 359 card articles. The sole verified card is
  `<article class="card card-verified" id="bambi-escs" ...>`; an earlier peer
  parser requiring the exact class string `card` counted 358 and omitted it.
- `scripts/CatalogServer/src/CatalogServer.jl` hard-codes both historical
  parseability and sampleability checks to false. Those badges are not prior
  empirical results. Only one source row (`bambi/escs`) claims `verified: true`.
- The renderer substitutes `gaussian` whenever the `family:` field is absent.
  Only 21 deployed rows declare that field. The resulting display distribution
  is 350 Gaussian, four binomial, two Bernoulli, two Poisson, and one
  negative-binomial; the 338 substitutions are recorded as
  `CatalogServer_default_unsubstantiated` rather than evidence.

The old catalogue was removed from the active development line by `a660429`.
No current-site stamp or current compilation result is inferred from it.

Receipt retrieval ran from 2026-07-28 21:07:08 through 21:07:15 UTC with
redirects, a 45-second timeout, and an 8 MB body bound. The row-expanded
first-pass capture has SHA-256
`1ffacae3ea8ed304c9e6a41093a49a349e213ad54d05cfc4945a79335734ffd7`;
the committed LF-normalized TSV has SHA-256
`5268eb4a16cc97afea5cf6409f44f6023a59688f6bb1a703defba9fe04d403cb`.
The untouched 206-URL retrieval ledger has SHA-256
`8a2ca250c48d2adceda9eb424d8db2e0945f29482330c1608dfffa4172827523`;
the committed copy differs only by replacing its capture-directory prefix with
relative `bodies/` paths and normalizing line endings; it has SHA-256
`9de9c1b6940bc193e1b2377575d8ae0064865250ef573a94f684ec0a15cebbf6`.
All 44 deployed partial-support verdicts were then reviewed against the retained
bodies; the override TSV has SHA-256
`657a1f8186c86436a94aff8f8ab0cf4e9adcbd84aa31d87b7b60858d9fe723cd`.

## Current execution environment

- BayesianRegressionModels.jl:
  `784712998ea67f6429d0a3b5a3241fe9cb690e64` (`ns/devibe`).
- StanBlocks.jl:
  `329a178a7ad7877da0b58ad2c360d417ddd663f9` (local active checkout selected
  by the canonical resolver).
- WarmupHMC.jl:
  `b185eedbbeeef6fb3327afb30dc995c98591af02` for this rerun. The independent
  peer sleepstudy control used
  `38398527e0406ad31aeaec3efe24d581a18a269e`; its result is preserved as a
  comparison receipt, not represented as this run's package SHA.
- Julia 1.10.11; BridgeStan 2.9.0 at its default
  `~/.bridgestan/bridgestan-2.9.0`; `KB_HOST=strato2`.

The corpus probe records generated Stan and Stan-data hashes. A result is fanned
out only when the normalized current BRM body, declared data columns, grouping
columns, selected family, generated Stan, and data schema are identical.

## SbPMX semantic HTMXO baseline

The selected design baseline is `origin/main` at
`a8fd3c02e1e2305683a56c91384701bfeba6b528`, observed from the remote on
2026-07-28. It is the newest coherent contract, not just the newest timestamp:

- `kb-design/SbPMX-inferrable` at `7fcd50cefc0fe4f4f85ad5bda2b038cd5ad957ee`
  is the merge base and is 79 commits behind main (0 ahead).
- Recorded deployed release
  `88c4fbff71228dfa9fca22504b87f016b76fa361` is three commits behind main
  (0 ahead).
- Main adds `81255eb` (code/target artifacts and disclosure), `10a819c`
  (routes return semantic nodes), and merge `a8fd3c0`.
- Other advertised semantic branches (`postprocessing`, `structure`,
  `synthetic-simulation`, and `v2`) are also contained in main.

Exact reviewed files and main-tree blob ids:

| path | blob |
|---|---|
| `design/ASKS.md` | `463d245aa4f9220611fd17092fb78c05b6bb3f06` |
| `design/SbPMX.jl` | `5889e49aa69386bfcc79515541d626017fe2a674` |
| `design/mock_runtime.jl` | `372f88a9b9f634909c161be858c54119a05bbe7c` |
| `compat/check-semantic-model-pipeline.jl` | `011d5ae50aa07118fdbdcaab186fb2fceae7ed1b` |

The contract uses `semantic_app`, semantic output nodes/cards, `@options`,
`option_domain`/`option_records`, descriptor operations, and ordinary GET
navigation. Retired `@semantic`/`dynamic_domain` vocabulary is not used.
`compat/check-semantic-model-pipeline.jl` is the existing 13-model
BRM→SBBRMI→SLIC→Stan reference gate and requires the complete card option
domain plus format-peer projections.

BRM's `brm_descriptor`/`brm_execute` graph is authoritative for executable
ordinary and kernel models. The adapter from `BRMDescriptor.operations` and
columns to an HTMXObjects `semantic_app` mount is not shipped. Consequently
every mount recommendation in `model_matrix.tsv` is explicitly experimental;
it proposes the current SbPMX semantic pattern rather than claiming an existing
authoritative application graph.

HTMXObjects semantic lock for the recorded SbPMX release:
`3a75633a5d9e50da844b1d553b037560ae1312f2`.
