# Response families

Specify the sampling distribution beside each response and use formulas for
any of its parameters. Links are explicit: `log(sigma) ~ 1 + x` declares a
positive scale, while `sigma ~ 1 + x` declares an unconstrained predictor.

## Support and parameters

Choose support first, then a parameterization whose priors you can interpret.
The [terms reference](terms-reference.md#Response-families) lists every mapped
constructor and the supported compositions.

| Response | Family examples | Parameters | Common explicit link |
| --- | --- | --- | --- |
| Real | `Normal(mu, sigma)`, `LocationScale(mu, sigma, TDist(nu))` | Location, scale; degrees of freedom | Identity for `mu`, log for scale |
| Positive | `Gamma(shape, scale)`, `InverseGaussian(mu, lambda)` | Shape/scale or mean/concentration | Log |
| Probability | `Beta(a, b)` | Two positive shapes | Log on shapes, or transform mean and precision |
| Binary/trials | `BernoulliLogit(eta)`, `BinomialLogit(n, eta)` | Log odds; trials are data | Identity on the log odds |
| Count | `Poisson(lambda)`, `NegativeBinomial2(mu, phi)` | Rate or mean/precision | Log |
| Unordered category | `CategoricalLogit(eta2, eta3, ...)` | Log odds against the first level | Identity on each log odds |
| Ordered category | `Ordinal(Cumulative(), ProbitLink(), eta)` | Predictor and inferred thresholds | Link supplied by the tag |
| Direction | `CircularVonMises(mu, kappa)` | Direction and concentration | Identity for direction, log for concentration |
| Joint real vector | `MvNormalCholesky([mu1, mu2], L)` | Mean vector, covariance factor | Identity for means |

Julia's `Exponential` and `Gamma` arguments are scales. BRM converts them to
Stan's rate convention when emitting Stan. `NormalCanon(eta, lambda)` uses
precision, `NegativeBinomial2(mu, phi)` has variance `mu + mu^2 / phi`, and
the scale in a Student-t model is not its standard deviation.

This model uses one covariate in a real-valued measurement and a positive
waiting time. Each response has its own distributional predictors.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:ContinuousFamilies), raw"""
model = (@brm begin
    mu ~ 1 + x
    log(sigma) ~ 1
    log(wait_mean) ~ 1 + x
    y ~ Normal(mu, sigma)
    waiting ~ InverseGaussian(wait_mean, concentration)

    effect(mu, :) ~ Normal(0, 1)
    effect(sigma, Intercept) ~ Normal(-1, 0.5)
    effect(wait_mean, :) ~ Normal(0, 0.5)
    concentration ~ Exponential(1)
end)((; x=[-1., 0., 1.], y=[-0.2, 0.4, 0.8], waiting=[0.5, 0.9, 1.8]))
""", :model; title="Real and positive responses", require_stan=true)
```

## Counts and extra zeros

`HurdlePoisson(lambda, p_zero)` assigns probability `p_zero` to zero and uses
a zero-truncated Poisson for positive counts. With
`ZeroInflatedPoisson(lambda, zi)`, the ordinary Poisson component can also
produce zeros. These probabilities therefore have different meanings.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:CountFamilies), raw"""
model = (@brm begin
    log(lambda) ~ 1 + x
    logit(p_zero) ~ 1
    eta ~ 1 + x
    count ~ HurdlePoisson(lambda, p_zero)
    successes ~ BinomialLogit(trials, eta)

    effect(lambda, :) ~ Normal(0, 0.5)
    effect(p_zero, Intercept) ~ Normal(0, 1)
    effect(eta, :) ~ Normal(0, 1)
end)((; x=[-1., 0., 1., 2.], count=[0, 1, 0, 4],
    trials=[10, 10, 12, 12], successes=[2, 5, 8, 10]))
""", :model; title="Hurdle counts and trial outcomes", require_stan=true)
```

## Categories and thresholds

A nested formula gives `CategoricalLogit` one predictor per non-reference
level. Plain vectors use sorted unique levels; a categorical vector uses its
declared level order. The first level has logit zero. Fitted levels remain
fixed during replay.

An ordinal model supplies its location through thresholds, so its common
predictor has no intercept. `Cumulative()` has ordered thresholds;
`StoppingRatio()` has separate stage intercepts. Both accept `LogitLink()`,
`ProbitLink()` or `CloglogLink()`.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:CategoryFamilies), raw"""
model = (@brm begin
    choice ~ CategoricalLogit(@brm(1 + x))
    eta ~ 0 + x
    rating ~ Ordinal(Cumulative(), ProbitLink(), eta)

    effect(eta, x) ~ Normal(0, 1)
end)((; x=[-1., -0.5, 0., 0.5, 1., 1.5],
    choice=["a", "b", "c", "a", "c", "b"], rating=[1, 1, 2, 2, 3, 3]))
""", :model; title="Unordered and ordered categories", require_stan=true)
```

## Observation mechanisms

Truncation conditions on being inside a sampling interval. Censoring records
a limit in place of an unobserved value. An interval-censored response is the
lower endpoint, and the likelihood is the probability of the interval
`(response, upper]`. These are different observations of the same base law.

| Observation | Declaration | Predictive quantity |
| --- | --- | --- |
| Selected positive values | `y ~ truncated(Normal(mu, sigma); lower=0)` | Truncated values |
| Detection limit | `y ~ censored(Normal(mu, sigma); lower=lloq)` | Values with the same clamping |
| Known interval | `y ~ interval_censored(Normal(mu, sigma); upper=hi)` | Uncoarsened values from the base law |
| Partly missing continuous values | `mi(y) ~ Normal(mu, sigma)` | A latent value for each missing row |
| Known relative precision | `y ~ weighted(Normal(mu, sigma), aweights(w))` | Normal SD `sigma / sqrt(w)` |
| Replicated or powered density | `y ~ weighted(Normal(mu, sigma), fweights(w))` | Base-law prediction; density multiplied by the weight |

The reference identifies supported base families. Response bounds may be
data columns. `interval_censored(x; upper=lloq)` inside a predictor has a
separate meaning: it imputes a censored covariate.

## Mixtures and correlated outcomes

`MixtureModel([Normal(mu1, sigma1), Normal(mu2, sigma2)], share)` marginalizes
component membership. Weights may be fixed, a declared Dirichlet simplex,
or a valid row-dependent expression. The reference describes allowed
component families. Stan can return membership probabilities as well as
predictive draws; RK currently supplies the density, gradient and pointwise
likelihood for this surface, but not mixture prediction.

For residual correlation, declare
`L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)` and observe
`[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`. This is a joint Gaussian
observation per complete row. Non-Gaussian residual copulas and partially
missing response vectors are outside that surface.

Continue to the [Warfarin example](warfarin.md) for responses with different
observation scales and a shared mechanistic model.
