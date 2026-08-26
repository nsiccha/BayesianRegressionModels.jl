# Wastewater-based Rt inference — EpiSewer / `ww-inference-model` port

A faithful port of the semi-mechanistic renewal model behind
[EpiSewer](https://github.com/adrian-lison/EpiSewer) (R / Julia) and the CDC
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model), built on
BRM's StanBlocks backend.

Authorized by decisions `2026-08-26T10-40-17-350-1kyw2vk` and
`2026-08-26T11-06-02-112-12s1pu8`. The deliverable is the **model** (it
transpiles, `stanc`-checks, and yields a finite BridgeStan density/gradient) —
not a fit; sampling was explicitly out of scope.

## The model (`renewal.jl`)

Three EpiSewer modules, all in one `@slic` model broadcast over sites by `plate`:

| Module | Form | Where |
|---|---|---|
| **Infection** | renewal `I(t) = Rt(t)·Σ_s g(s)·I(t-s)`, `Rt` a log random walk | `ww_renewal` `@deffun` + plate cell |
| **Shedding** | load `L(t) = Σ_s sh(s)·I(t-s+1)` (fecal shedding convolution) | `ww_shed` `@deffun` |
| **Measurement** | `log C(t) ~ Normal(log L(t) + log_scale, sigma_obs)`; below-LOD censoring variant | plate cell obs |

- `wastewater_model([data])` — core, lognormal concentration likelihood.
- `wastewater_censored_model([data])` — left-censors non-detects below `lloq`
  via `censored(normal, …; lower = lloq)`.
- `wastewater_fixture(; nsites, nt)` — a minimal multi-site dataset.

Sites share `sigma_rw` / `sigma_obs` / `log_scale` / `logR0`; each site carries
its own random-walk `Rt` and seeding.

### Why the shape is what it is

The mechanistic core is a **carried-state sequential scan** (`I[t]` depends on
`I[t-1 … t-ng]`) — which `plate` deliberately does **not** express (its cells are
independent). So the two convolutions live in `@deffun` Stan functions and are
called loop-free from the `@slic`/plate body, exactly as SbPMX's `ode_rk45` PMX
cells do. This is the key structural fact: the observation/hierarchy **shell** is
ordinary BRM (multilevel + a family + censoring), and only the renewal/shedding
**recurrence** needs the `@deffun` seam.

The generation-interval PMF `g` and shedding PMF `sh` are passed as **data**
(captured into the plate cell), so they are re-bindable without editing the
model — no baked-in constants.

## Verification

`test/wastewater_model.jl` gates both variants on transpile + `stanc` + finite
BridgeStan density/gradient (`julia --project=test test/wastewater_model.jl`),
plus a data re-bind. Measured green on strato2: 10/10 assertions; core and
censored each dim=91.

## Known BRM limitation surfaced here

The `@brm` **kernel** formula surface cannot yet thread a shared, non-per-subject
data vector (a generation-interval / delay PMF) into its cell — neither by
lexical capture nor as a data kwarg (measured: `Could not find g in model,
builtin, Main`). Plate-level capture at the `@slic` layer *does* work (used
here), which is why this port is written at the `@slic` layer — the faithful
EpiSewer reference is hand-written Stan at that same layer anyway. Lifting that
into the `@brm` kernel surface would be a BRM ergonomics enhancement (tracked
separately); it is not needed for the model to be correct.

## Faithful-but-simplified

Kept minimal for a first port; each is a straightforward extension, not a
blocker:

- **Flow normalization** — `log_scale` is one scalar; EpiSewer divides load by
  measured flow (a per-timepoint covariate → a ragged per-site plate slice).
- **Rt prior** — a Gaussian RW; EpiSewer also offers smoothing-spline / AR
  variants (a different cell prior).
- **Overdispersion** — lognormal measurement noise; a Gamma/observation-level
  dispersion is a family swap.
- **Multiple subpopulations per site** (CDC `ww-inference-model`) — a nested
  grouping layer.
