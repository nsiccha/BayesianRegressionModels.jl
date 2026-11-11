# Documentation

The site is built from the current package with Documenter and VitePress. Model
views and AlgebraOfVega specifications execute during the build; posterior
summaries come from deliberate, cached research runs. CI never samples.
The hosted build runs Julia 1.11: every official 1.10.x miscompiles the prior
constructor binding of the RK backend (JuliaLang/julia#52644), which the fleet
hosts avoid with a patched build. The docs project's public pins live in its
`[docs_pins]` table, read by the workflow, because Pkg 1.11 honours `[sources]`.

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
| Examples | `state-space.md` | Connect continuous latent dynamics with discrete observations. |
| Examples | `monster.md` | Fit a whole-body pharmacokinetic model and inspect convergence and predictive checks. |
| Examples | `jax.md` | Export a hierarchical regression and compare Julia and NumPyro fits. |
| Examples | `variational-inference.md` | Fit streaming and batch variational approximations. |
| Examples | `filtering.md` | Compare sequential posterior updates and full-data sampling. |
| Examples | `spatial-fields.md` | Map a space-time field from station measurements. |
| Reference | `api.md` | Find public functions and types by task. |
| Reference | `terms-reference.md` | Look up formula terms and their arguments. |

Research history, fit settings and reproduction commands live in `research/` or
`bench/`, linked from each example's Reproduce footer. Fit directories contain
small TSV summaries and a receipt; raw draws and rendered figures stay outside
git. The build reads missing results as errors.

The figure runtime is copied from the pinned AlgebraOfVega package to the build
output. No compiled JavaScript, Vega runtime or rendered plot is committed.
Light and dark desktop renders are part of documentation review.

The table and figure helpers live in `docs/backend_comparisons.jl`. Quantity
tables use `quantity, unit, description, median, q025, q975, rhat, ess_bulk,
ess_tail`, with one fit per directory.

Every model comparison has the same five tabs: BRM authoring, StanBlocks model,
RKPPL model, Stan source and JAX module. The comparison dialog initially selects
BRM, StanBlocks and RKPPL. The build emits each backend view from the displayed
model; unavailable backends keep their tab and show the emission diagnostic.

## Landing notes

The Warfarin page draws its five figures at build time from the committed
fit tables (`research/warfarin/figures.jl`). The Monster page also includes
its cached fit, convergence diagnostics and posterior predictive figure
(`research/monster/figures.jl`). Its run kept three of four chains, with no
divergences; eight population parameters have R-hat above 1.1. The page states
that limitation alongside the predictive checks.

The grey-seal example is held out pending a density-dependence correction and
a successful fit. Its provenance is settled: the author wrote the implementation
of the published grey-seal IPM. Correcting the research files' source note remains
part of restoring the page. It is absent from this site's navigation and build.

The rework removes the feature atlas, slide deck and live gallery, with their
compiled deck files and the deck renderer that depended on the retired atlas.
Their subjects are taught in the task guides and worked
examples. Rendered plot files and serialized posterior draws are removed from
the documentation and research trees; scripts retain the regeneration path.

`research/irregular_read_gmrf` is removed because its acceptance experiments
required a private uploaded dataset and were not reproducible from public or
synthetic inputs. The historical material remains in the private archive; this
public repository no longer carries its pointer or dependent scripts.
Research scripts use repository-relative output directories in place of local
worktree and scratch paths. Public receipts describe the machine generically.

Historical receipts retain their numeric results and source hashes; machine-local
paths are normalized and raw render logs are omitted from the public tree.

Large generated inventory tables and compressed draw tables are also untracked;
small summary tables remain for fitted examples and research reference.
Obsolete case-study page generators and the `research/automatic_total_docs`
render fixture are removed, so research commands cannot recreate retired site
pages or copy rendered PNGs back into the documentation.

The paired-field and station-series test fixtures retain the author's own
model structures and synthetic data at the package owner's direction.
