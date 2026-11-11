# Terms and priors

Choose a predictor term, attach a prior to the quantity it creates, and inspect
the resulting declaration before fitting. A term may allocate a coefficient,
a group-effect block or a latent function; these have different prior addresses.

## Prior targets

The three forms below name an authored parameter, a formula coefficient, or a
quantity owned by a term or group block. Distribution arguments use Julia's
conventions: `Exponential` and `Gamma` take a scale, not Stan's rate.

| Target | Prior statement | Quantity |
| --- | --- | --- |
| Authored parameter | `sigma ~ Exponential(1)` | A positive scalar |
| Formula coefficient | `effect(mu, x) ~ Normal(0, 0.3)` | The coefficient multiplying `x` |
| Term or group parameter | `sd(mu, subject) ~ Exponential(0.5)` | The subject-effect SD |

Population coefficients default to independent standard normals. A smooth's
amplitude belongs to the smooth, so use `sd(mu, hsgp(x))`; it has no additional
population multiplier. `offset(x)` fixes the multiplier at one and allocates
no coefficient. The [terms reference](terms-reference.md) lists defaults,
prior families and backend coverage by term.

This model overrides the population slope, the group SD and the HSGP amplitude.
It retains the HSGP's approximation-validity bound on its length scale.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:PriorGuide), raw"""
model = (@brm begin
    mu ~ 1 + x + hsgp(t; k=5) + (1 | subject)
    y ~ Normal(mu, sigma)

    sigma ~ Exponential(0.5)
    effect(:, x) ~ Normal(0, 1)
    effect(mu, x) ~ Normal(0, 0.3)
    sd(mu, subject) ~ Exponential(0.5)
    sd(mu, hsgp(t)) ~ Exponential(0.7)
end)((; x=[-1., 0., 1., -1., 0., 1.],
    t=[0., 0.5, 1., 1.5, 2., 2.5], subject=["a", "a", "a", "b", "b", "b"],
    y=[-0.4, 0.2, 0.6, -0.1, 0.3, 0.8]))
""", :model; title="Coefficient, group and smooth priors", require_stan=true)
```

```@example prior_lookup
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:PriorGuide).model
(coefficient_priors=effect_priors(model), term_priors=term_priors(model))
```

A colon supplies a default for every matching predictor or coefficient; the
more specific address wins. Equal-specificity ties and unmatched addresses
raise errors during lowering. `optional(address) ~ prior` is for a reusable
prior block whose target may be absent in some members of a model family.

A linked formula such as `log(sigma) ~ 1 + x` takes addresses under `sigma`,
with coefficient priors on the log scale. A categorical column uses its column
name, `effect(mu, site)`. Cell means also accept `site_lvl_k` for the level in
fitted-order position `k`. For a shared `|p|` group block, `sd(:, p)` reaches
all margins and `cor(:, p)` sets its LKJ prior; `sd(mu, p)` selects the margins
belonging to `mu`. The LKJ dimension must match the block width.

Term addresses omit numeric and keyword arguments: address `me(x, 0.3)` as
`me(x)`. A tensor smooth also needs its penalty component, for example
`sd(mu, t2(x, z), rr)`. Standardized innovations retain their standard-normal
law; changing their scale would duplicate the term's amplitude.

## Bounds and vectors

`restricted(D; lower, upper)` evaluates the original density on the bounded
support. `truncated(D; lower, upper)` also divides by the probability mass
inside the bounds. They give the same posterior when that mass is constant;
with sampled hyperparameters, their joint laws differ.

The vector statement here declares one independent positive scale per
observation.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:VectorPriorGuide), raw"""
model = (@brm begin
    @plate for i in eachindex(y)
        y[i] ~ Normal(location, scales[i])
    end
    location ~ Normal(0, 1)
    scales[1:length(y)] .~ truncated.(Normal.(0.0, 0.5), 0.0, Inf)
end)((; y=[-0.2, 0.1, 0.7]))
""", :model; title="A declared vector of positive parameters", require_stan=true)
```

```@example prior_mass
using Distributions
mass_inside(mu) = ccdf(Normal(mu, 1), 0)
locations = [-1., 0., 1.]
mass = mass_inside.(locations)
@assert issorted(mass)
collect(zip(locations, mass))
```

The mass changes with the Normal location. Dropping a truncation normalizer
when that location is sampled would change its prior.

| Declaration | Use |
| --- | --- |
| `x[1:n] .~ Normal.(0, scales)` | Independent elements; scalar arguments broadcast, data vectors apply elementwise |
| `x ~ MvNormal(mean, covariance)` | A joint Gaussian vector; `Diagonal(v)` contains variances |
| `share ~ Dirichlet(K, a)` | A simplex with concentration `a` per component |
| `share ~ Dirichlet(alpha)` | A simplex with a data-valued concentration vector |

The length of a declared vector comes from an integer or data expression.
An undeclared `x ~ Normal(0, scales)` with vector data has omitted-response
semantics; it is not a vector-parameter declaration. `mo(c)` returns a monotonic
effect on observation rows, so it is not a substitute for a standalone simplex.

## Structured shrinkage

`effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)` allocates a predictor's
variance between its non-intercept coefficients and one random-intercept
residual. `sd(:, p) ~ r2d2(reference_scale=sigma)` instead allocates a shared
group block's variance; `include=(:population, :contrasts)` extends that budget
to its predictors' coefficients. The reference documents exclusions and
reference-scale choices. These constructions replace independent priors;
they are not extra priors on top of them.

`beta ~ Horseshoe(...)` declares a coefficient used through an assignment such
as `mu = beta * x`. `effect(mu, x) ~ Horseshoe(...)` addresses an existing
population coefficient. Inspect the backend's supported form before sharing
a shrinkage scale across coefficients.

For a fitted application of group and scale priors, continue to the
[Warfarin example](warfarin.md).
