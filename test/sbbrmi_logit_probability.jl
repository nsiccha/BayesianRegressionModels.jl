# test/sbbrmi_logit_probability.jl — SBBRMI reads a probability the model gives
# as the logistic of a predictor on the logit scale (snag
# sbbrmi-logit-lin-2c4202f8).
#
#     julia --startup-file=no --project=test test/sbbrmi_logit_probability.jl
#
# Such a `p` has three spellings: the `p` of `logit(p) ~ ...`, an inline
# `logistic(x)`, and an assignment `p = logistic(x)`. `Bernoulli(p)` and
# `Binomial(n, p)` lower to Stan's `bernoulli_logit` / `binomial_logit` of `x`,
# a `BetaBinomial2` mean gives its second shape as `inv_logit(-x) .* precision`,
# and a written `1 - p` lowers to `inv_logit(-x)`. On the probability scale
# these programs were -Inf or NaN from x = 37, where `1 - inv_logit(x)` rounds
# to zero, and at x = -745, where `inv_logit(x)` underflows. Densities and
# gradients are compared with 512-bit references.
#
# Shape-parameter laws (`beta_binomial`, `beta`) stay exact through |x| = 700;
# from |x| ≈ 745 a shape itself underflows to zero (`inv_logit(-745)` is the
# smallest subnormal), so those points are not checked.

using Test
using BayesianRegressionModels
using Distributions
using BridgeStan
using LogDensityProblems
using SpecialFunctions: loggamma
using StanBlocks

const BRM = BayesianRegressionModels

const PHI = 20.0
const KAPPA = 5.0
const DATA = (; y=[3, 30, 0, 57, 1], n=[10, 60, 12, 57, 400], PHI_=PHI,
    z=[0, 1, 1, 0, 1], prop=[0.2, 0.7, 0.4, 0.6, 0.3], KAPPA_=KAPPA)

# Exact log densities of the one coordinate x (the linear predictor), up to an
# x-free constant, under the Normal(0, 1000) prior every model below puts on it.
setprecision(BigFloat, 512)
prior(x) = -x^2 / (2 * big(1000.0)^2)
logsig(x) = -log1p(exp(-x))
rising(a, k) = sum(log(a + i) for i in 0:k-1; init=big(0.0))
function exact_betabinomial(x::BigFloat)
    a = PHI / (1 + exp(-x)); b = PHI / (1 + exp(x))
    prior(x) + sum(rising(a, y) + rising(b, n - y) - rising(big(PHI), n)
        for (y, n) in zip(DATA.y, DATA.n))
end
exact_binomial(x::BigFloat) = prior(x) +
    sum(y * logsig(x) + (n - y) * logsig(-x) for (y, n) in zip(DATA.y, DATA.n))
exact_bernoulli(x::BigFloat) = prior(x) +
    sum(z == 1 ? logsig(x) : logsig(-x) for z in DATA.z)
function exact_beta(x::BigFloat)
    k = big(KAPPA)
    a = k / (1 + exp(-x)); b = k / (1 + exp(x))
    prior(x) + sum((a - 1) * log(big(v)) + (b - 1) * log1p(-big(v)) -
        (loggamma(a) + loggamma(b) - loggamma(k)) for v in DATA.prop)
end
exact_gradient(f, x) = (h = big(1e-30); (f(x + h) - f(x - h)) / (2h))

const MODERATE = (2.0, -2.0, 10.0, -10.0, 30.0, -30.0, 36.0, 37.0, -37.0,
    40.0, -40.0, 100.0, -100.0, 700.0, -700.0)
const EXTREME = (745.0, -745.0, 800.0, -800.0)

model_block(code) = first(split(last(split(code, "model {")), "generated quantities {"))

# Each case: name, builder, exact density, points, model-block lines that must
# appear, and text that must not.
const CASES = [
    ("linked BetaBinomial2", (@brm begin
        logit(p) ~ 1
        effect(p, Intercept) ~ Normal(0, 1000)
        y ~ BetaBinomial2(n, p, PHI_)
    end), exact_betabinomial, MODERATE,
        ["y ~ beta_binomial(n, (p .* PHI_), (inv_logit((-logit_p)) .* PHI_));"], ["(1 - p)"]),
    ("inline BetaBinomial2", (@brm begin
        eta ~ 1
        effect(eta, Intercept) ~ Normal(0, 1000)
        y ~ BetaBinomial2(n, logistic(eta), PHI_)
    end), exact_betabinomial, MODERATE,
        ["y ~ beta_binomial(n, (inv_logit(eta) .* PHI_), (inv_logit((-eta)) .* PHI_));"],
        ["(1 - inv_logit"]),
    ("linked Binomial", (@brm begin
        logit(p) ~ 1
        effect(p, Intercept) ~ Normal(0, 1000)
        y ~ Binomial(n, p)
    end), exact_binomial, (MODERATE..., EXTREME...),
        ["y ~ binomial_logit(n, logit_p);"], ["binomial(n, p)"]),
    ("inline Binomial", (@brm begin
        eta ~ 1
        effect(eta, Intercept) ~ Normal(0, 1000)
        y ~ Binomial(n, logistic(eta))
    end), exact_binomial, (MODERATE..., EXTREME...),
        ["y ~ binomial_logit(n, eta);"], ["inv_logit(eta)"]),
    ("linked Bernoulli", (@brm begin
        logit(p) ~ 1
        effect(p, Intercept) ~ Normal(0, 1000)
        z ~ Bernoulli(p)
    end), exact_bernoulli, (MODERATE..., EXTREME...),
        ["z ~ bernoulli_logit(logit_p);"], ["bernoulli(p)"]),
    ("written 1 - mu in Beta shapes", (@brm begin
        logit(mu) ~ 1
        effect(mu, Intercept) ~ Normal(0, 1000)
        prop ~ Beta(mu * KAPPA_, (1 - mu) * KAPPA_)
    end), exact_beta, MODERATE,
        ["prop ~ beta((mu .* KAPPA_), (inv_logit((-logit_mu)) .* KAPPA_));"], ["(1 - mu)"]),
    ("cell BetaBinomial2", (@brm begin
        eta ~ Normal(0, 1000)
        @plate for i in eachindex(y)
            y[i] ~ BetaBinomial2(n[i], logistic(eta), 20.0)
        end
    end), exact_betabinomial, MODERATE, ["inv_logit((-eta))"], ["(1 - inv_logit"]),
    ("cell Bernoulli", (@brm begin
        eta ~ Normal(0, 1000)
        @plate for i in eachindex(z)
            z[i] ~ Bernoulli(logistic(eta))
        end
    end), exact_bernoulli, (MODERATE..., EXTREME...),
        ["~ bernoulli_logit(eta);"], ["bernoulli(inv_logit"]),
    # Its predictive draw is StanBlocks' scalar `binomial_logit_rng` (0.2.71).
    ("cell Binomial", (@brm begin
        eta ~ Normal(0, 1000)
        @plate for i in eachindex(y)
            y[i] ~ Binomial(n[i], logistic(eta))
        end
    end), exact_binomial, (MODERATE..., EXTREME...),
        ["~ binomial_logit(", "binomial_logit_rng("], ["inv_logit(eta)", "binomial("]),
]

# `present` is read in the whole program (a cell's shapes are transformed
# parameters), `absent` in the model block (the functions block holds
# StanBlocks' `binomial_logit_rng`, which draws `binomial_rng(N, inv_logit(eta))`).
@testset "SBBRMI logit-scale probability" begin
@testset "$name" for (name, builder, exact, points, present, absent) in CASES
    sb = SBBRMI(builder(DATA); mod=@__MODULE__)
    code = stan_code(sb)
    model = model_block(code)
    for line in present
        @test occursin(line, code)
    end
    for text in absent
        @test !occursin(text, model)
    end
    problem = StanBlocks.stan_instantiate(sb.model)
    @test LogDensityProblems.dimension(problem) == 1
    lp0 = first(LogDensityProblems.logdensity_and_gradient(problem, [0.0]))
    exact0 = exact(big(0.0))
    for x in points
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, [x])
        reference = Float64(exact(big(x)) - exact0)
        reference_gradient = Float64(exact_gradient(exact, big(x)))
        @test isapprox(lp - lp0, reference; rtol=1e-12, atol=1e-9)
        @test isapprox(only(gradient), reference_gradient; rtol=1e-10)
    end
end
end

@testset "the other spellings reach the same Stan forms" begin
    assigned = @brm begin
        eta ~ 1
        effect(eta, Intercept) ~ Normal(0, 1000)
        p = logistic(eta)
        y ~ BetaBinomial2(n, p, PHI_)
        z ~ Bernoulli(p)
    end
    code = stan_code(SBBRMI(assigned(DATA); mod=@__MODULE__))
    model = model_block(code)
    @test occursin("y ~ beta_binomial(n, (p .* PHI_), (inv_logit((-eta)) .* PHI_));", model)
    @test occursin("z ~ bernoulli_logit(eta);", model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # A scalar logit: the observed responses' predictive draws still compile.
    scalar = @brm begin
        logit_z ~ Normal(0.0, 1.0)
        y ~ Binomial(n, logistic(logit_z))
        z ~ Bernoulli(logistic(logit_z))
    end
    code = stan_code(SBBRMI(scalar(DATA); mod=@__MODULE__))
    model = model_block(code)
    @test occursin("y ~ binomial_logit(n, logit_z);", model)
    @test occursin("z ~ bernoulli_logit(logit_z);", model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # A written `1 - logistic(x)` in a cell, and its broadcast at model scope.
    written = @brm begin
        eta ~ Normal(0, 1000)
        @plate for i in eachindex(prop)
            prop[i] ~ Beta(logistic(eta) * 5.0, (1 - logistic(eta)) * 5.0)
        end
    end
    code = stan_code(SBBRMI(written(DATA); mod=@__MODULE__))
    @test occursin("(inv_logit((-eta)) * 5.0)", code)
    @test !occursin("(1 - inv_logit", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    broadcast = @brm begin
        logit(mu) ~ 1
        q = 1 .- mu
        prop ~ Beta(mu * KAPPA_, q * KAPPA_)
    end
    code = stan_code(SBBRMI(broadcast(DATA); mod=@__MODULE__))
    @test occursin("q = inv_logit((-logit_mu));", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # A cell's `BinomialLogit` draws through the same scalar `binomial_logit_rng`.
    cell_logit = @brm begin
        eta ~ Normal(0, 1000)
        @plate for i in eachindex(y)
            y[i] ~ BinomialLogit(n[i], eta)
        end
    end
    code = stan_code(SBBRMI(cell_logit(DATA); mod=@__MODULE__))
    @test occursin("~ binomial_logit(", model_block(code))
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # An omitted cell binomial is only drawn, now through `binomial_logit_rng`.
    # With `y` omitted the program is the prior one: `eta` is drawn in generated
    # quantities too, so nothing is sampled, and at the prior's wide `eta` each
    # draw is all failures or all successes.
    omitted_cell = @brm begin
        eta ~ Normal(0, 1000)
        @plate for i in eachindex(n)
            y[i] ~ Binomial(n[i], logistic(eta))
        end
    end
    sb = SBBRMI(omitted_cell(Base.structdiff(DATA, NamedTuple{(:y,)})); mod=@__MODULE__)
    code = stan_code(sb)
    @test occursin("binomial_logit_rng(", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    problem = StanBlocks.stan_instantiate(sb.model)
    @test LogDensityProblems.dimension(problem) == 0
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    rng = BridgeStan.StanRNG(problem.model, 3)
    draws = reduce(hcat, (BridgeStan.param_constrain(problem.model, Float64[];
        include_tp=true, include_gq=true, rng) for _ in 1:200))
    eta = draws[findfirst(==("eta"), names), :]
    for (i, trials) in enumerate(DATA.n)
        y = draws[findfirst(==("y_cell_y.$i"), names), :]
        @test all(v -> isinteger(v) && 0 <= v <= trials, y)
        @test all(y[eta .> 40] .== trials)
        @test all(y[eta .< -40] .== 0)
    end
end

@testset "probability-scale forms stay where the logit forms do not apply" begin
    # A probability that is not a logistic keeps its family.
    identity_p = @brm begin
        p ~ Beta(2, 2)
        y ~ Binomial(n, p)
        w ~ Beta(2, 2)
        prop ~ Beta(w * KAPPA_, (1 - w) * KAPPA_)
    end
    code = stan_code(SBBRMI(identity_p(DATA); mod=@__MODULE__))
    @test occursin("y ~ binomial(n, p);", code)
    @test occursin("(1 - w)", code)
    # A cell's binomial of a probability that is not a logistic keeps its family.
    cell_binomial = @brm begin
        p ~ Beta(2, 2)
        @plate for i in eachindex(y)
            y[i] ~ Binomial(n[i], p)
        end
    end
    code = stan_code(SBBRMI(cell_binomial(DATA); mod=@__MODULE__))
    @test !occursin("binomial_logit", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "a logit-linked binomial keeps its pointwise, predictive and weighted paths" begin
    linked = @brm begin
        logit(p) ~ 1
        y ~ Binomial(n, p)
    end
    descriptor = brm_descriptor(linked, DATA; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)
    @test occursin("y ~ binomial_logit(n, logit_p);", code)
    @test occursin("p = inv_logit(logit_p);", code)
    @test brm_output(descriptor, :p; role=:linear_predictor).name === :p
    problem = brm_execute(descriptor, :instantiate)
    q = [0.4]
    pointwise = brm_execute(descriptor, :pointwise_loglik; problem, draws=q, seed=5)
    @test pointwise.y_likelihood ≈ logpdf.(Binomial.(DATA.n, logistic(only(q))), DATA.y)
    predictive = brm_execute(descriptor, :predict; problem, draws=q, seed=5)
    @test length(predictive.y_gen) == length(DATA.y)
    @test all(0 .<= predictive.y_gen .<= DATA.n)

    weighted_model = @brm begin
        logit(p) ~ 1
        z ~ weighted(Bernoulli(p), fweights(freq))
    end
    code = stan_code(SBBRMI(weighted_model(merge(DATA, (; freq=[1, 2, 1, 3, 1])));
        mod=@__MODULE__))
    @test occursin("bernoulli_logit", model_block(code))
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end
