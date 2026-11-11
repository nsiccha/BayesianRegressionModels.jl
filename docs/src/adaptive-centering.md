# Adaptive centering

Adapt the coordinates of hierarchical Gaussian effects during warmup while
preserving the model's posterior. Different groups, margins and HSGP
frequencies can benefit from different amounts of centering.

## Coordinates and the posterior

A noncentered scalar effect is `b = tau*z`, with `z ~ Normal(0,1)`.
A centered effect samples `b ~ Normal(0,tau)` directly. Partial centering
uses `u = tau^c*z`, so `b = tau^(1-c)*u`. The endpoints `c=0` and `c=1`
are NCP and CP. A coordinate change includes its Jacobian; it does not
change the prior or add information.

```@example centering_identity
using Distributions
tau, z = 0.3, 1.2
controls = [0., 0.5, 1.]
logdensities = [logpdf(Normal(0, tau^c), tau^c*z) + c*log(tau) for c in controls]
@assert all(v -> isapprox(v, logpdf(Normal(), z)), logdensities)
collect(zip(controls, logdensities))
```

Weakly informed groups often suit NCP, while dense measurements can favor
CP. Correlation, uncertain scales and unequal group sizes make a single
choice less useful. For correlated blocks the partial coordinates use
conditional means and scales in a defined ordering, not independent
rescaling of each marginal.

## An ordinary hierarchy

Eight schools has eight estimated effects, one population location and one
between-school scale. The known measurement SDs differ across schools.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:CenteringGuide), raw"""
model = (@brm begin
    theta ~ 1 + (1 | schools | school)
    y ~ Normal(theta, sigma)

    effect(theta, Intercept) ~ Normal(0, 5)
    sd(:, schools) ~ Cauchy(0, 5)
end)((; school=collect(1:8), y=[28., 8., -3., 7., -1., 1., 18., 12.],
    sigma=[15., 10., 16., 11., 9., 11., 10., 18.]))
""", :model; title="Eight schools in conventional coordinates", require_stan=true)
```

`SBBRMI` defaults to conventional noncentered random effects.
`centered_groups=(:school,)` chooses a fixed centered Stan representation.
RK emits noncentered effects; its adaptive wrapper changes the sampler's
frame at runtime. Exact-total and S2Z coordinates are separate preparation
choices described in the [terms reference](terms-reference.md).

## The fitting workflow

For Stan, prepare `SBBRMI(model; total_groups=())`, instantiate its density
with `stan_instantiate`, then call
`adaptive_centering_problem(sb, problem, ad_backend)`. For RK, prepare
`RKBRMI(model)` and `rk_logdensity_problem(rk; ad_backend)`, then pass
`rk`, that problem and the same backend to `adaptive_centering_problem`.
WarmupHMC samples the wrapped problem. Its completion-based workflow runs
independent chains; a fit receipt must report any omitted chains.

Inspect `adaptive_centering_blocks` and `adaptive_centering_controls` to
relate controls to model quantities. The wrapper exposes a grid from zero
to one per supported cell. A position loss measures scale variation in
candidate frames; a gradient loss measures position–gradient dependence.
A pilot-based choice requires a fresh fit, while online adaptation updates
during warmup. Compare their full gradient costs, including pilots.

Returned `posterior_position` draws are in the original model frame.
Checkpoint positions may be in the sampler's adapted frame. Keep their
frame and control metadata when resuming or auditing a fit; do not feed
checkpoint coordinates directly into ordinary prediction.

## HSGP frequencies

The same idea applies to HSGP weights: each basis function has a spectral
scale determined by its frequency and the GP hyperparameters. Low and high
frequencies need not select the same centering.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:CenteringSurfaceGuide), raw"""
model = (@brm begin
    mu ~ 1 + hsgp(time; k=8, domain=(-1.5, 1.5))
    y ~ Normal(mu, sigma)

    effect(mu, Intercept) ~ Normal(0, 1)
    sd(mu, hsgp(time)) ~ Exponential(1)
    sigma ~ Exponential(0.3)
end)((; time=collect(range(-1, 1; length=9)),
    y=[-0.4, -0.5, -0.1, 0.2, 0.6, 0.8, 0.5, 0.2, 0.1]))
""", :model; title="One control per HSGP weight", require_stan=true)
```

Stan's adaptive wrapper supports ordinary Gaussian and R2D2 blocks, supported
HSGP weights, CAR and S2Z cells. Grouped HSGP weights can use correlated
levels and group-specific scales; their conditional ordering is fixed by
the fitted levels. `ranef_order=:learned` applies to supported ordinary
correlated blocks. Bounds that depend on other sampled parameters are
outside the HSGP adaptive-scale surface. Extremely small spectral scales
use a smooth numerical floor; controls observed only on that floor stay
noncentered.

RK's wrapper currently covers ordinary Gaussian effects with sampled scales.
Bounded or Student-t blocks, known level covariance and stratified group
blocks have narrower support. Unsupported field cells pass through in
their original frame. Check the prepared block inventory before assuming
that every latent quantity adapts. SVI's learned centering optimizes its
variational objective; it is a separate procedure, though its controls can
initialize supported MCMC adaptation.

## What a comparison can establish

The recorded eight-schools study compares fixed coordinates with position
and gradient adaptation. The figure uses minimum scientific-quantity bulk
ESS per total gradient, relative to NCP; pilot gradients are included.
These are historical single-chain runs with 10,000 retained draws per arm,
not current runtime benchmarks or evidence of multichain convergence.

```@eval
isdefined(Main, :BRMGuideFigures) || Base.include(Main, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "docs", "guide_figures.jl"))
Main.BRMDocsComparisons.figure(Main.BRMGuideFigures.centering_efficiency();
    caption="Recorded eight-schools efficiency relative to fixed NCP. The denominator includes warmup and any pilot. Each arm retained one chain of 10,000 draws; the research receipt records versions, seeds and diagnostics.")
```

```@eval
Main.BRMDocsComparisons.results_table("research/centering_refresh/results/eight/comparison.tsv";
    columns=[:method => "Method", :min_bulk_ess => "Minimum bulk ESS",
        :divergences => "Divergences", :total_gradients => "Total gradients"])
```

Adaptation does not guarantee greater efficiency. Check divergences, bulk
and tail ESS, R-hat across independent chains, and agreement in scientific
quantities. Compare the same posterior and count all fitting work. A short
pilot that misses a region can select poor coordinates for a later fit.

## Reproduce

The [research index](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/adaptive_centering)
links the motorcycle, eight-schools, radon, pupil, pupil-scale, air and RBesT
protocols, source provenance and scripts. Their figures and draws are
regenerated outside the repository; the page reads small recorded summaries.
For adaptive fitting in a full model, continue to the
[Warfarin example](warfarin.md).
