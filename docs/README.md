# Documentation

The site is built from the current package with Documenter and VitePress. Model
views and AlgebraOfVega specifications execute during the build; posterior
summaries come from deliberate, cached research runs. CI never samples.

| Part | Page | Purpose |
|:--|:--|:--|
| Start | `index.md` | Install the environment, inspect a hierarchical regression and read its fit. |
| Guides | `model-syntax.md` | Declare responses, predictors, groups and reusable components. |
| Guides | `terms-and-priors.md` | Set term structure and priors. |
| Guides | `response-families.md` | Choose families and express truncation, censoring and weights. |
| Guides | `space-time-fields.md` | Choose Gaussian-process and structured spatial or temporal terms. |
| Guides | `backends.md` | Select Julia, Stan or JAX execution and inspect generated models. |
| Guides | `adaptive-centering.md` | Choose and interpret hierarchical parameterizations. |
| Examples | `warfarin.md` | Fit sequential and joint pharmacokinetic/pharmacodynamic models. |
| Examples | `wastewater.md` | Infer transmission from wastewater with a renewal model. |
| Examples | `renewal.md` | Combine infection renewal and reporting delays. |
| Examples | `seal-ipm.md` | Combine demographic observations in an integrated population model. |
| Examples | `state-space.md` | Connect continuous latent dynamics with discrete observations. |
| Examples | `monster.md` | Fit a whole-body pharmacokinetic model. |
| Examples | `jax.md` | Export a hierarchical regression and compare Julia and NumPyro fits. |
| Examples | `svi.md` | Fit streaming and batch variational approximations. |
| Examples | `filtering.md` | Compare sequential posterior updates and full-data sampling. |
| Examples | `spatial-fields.md` | Reconstruct a spatial field and show uncertainty at new locations. |
| Reference | `api.md` | Find public functions and types by task. |
| Reference | `terms-reference.md` | Look up formula terms and their arguments. |

Research history, fit settings and reproduction commands live in `research/` or
`bench/`, linked from each example's Reproduce footer. Fit directories contain
small TSV summaries and a receipt; raw draws and rendered figures stay outside
git. The build reads missing results as errors.

The figure runtime is copied from the pinned AlgebraOfVega package to the build
output. No compiled JavaScript, Vega runtime or rendered plot is committed.
Light and dark desktop renders are part of documentation review.

The candidate uses the existing `BRMDocsComparisons` table and figure hooks until
the shared documentation package is public. The transitional quantity schema is
`quantity, unit, description, median, q025, q975, rhat, ess_bulk, ess_tail`, with
one fit per directory. It is compatible with the planned DocsFits summaries.

## Landing notes

The rework removes the feature atlas, slide deck and live gallery, with their
compiled deck files. Their subjects are taught in the task guides and worked
examples. Rendered plot files and serialized posterior draws are removed from
the documentation and research trees; scripts retain the regeneration path.

`research/irregular_read_gmrf` is removed because its acceptance experiments
required a private uploaded dataset and were not reproducible from public or
synthetic inputs. The historical material remains in the private archive; this
public repository no longer carries its pointer or dependent scripts.
Research scripts use repository-relative output directories in place of local
worktree and scratch paths. Public receipts describe the machine generically.
