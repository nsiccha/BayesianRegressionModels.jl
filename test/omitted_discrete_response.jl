# Omitted responses with a discrete law, `BetaBinomial2` among them.
#
#   julia --startup-file=no --project=test test/omitted_discrete_response.jl
#
# Stan has no discrete parameters, so an omitted response with a discrete law
# emits its bound twin's likelihood statement and StanBlocks forward-simulates
# it, its arguments spelled as there (`logistic` -> `inv_logit`, dotted
# products). `BetaBinomial2`'s outcome structure is its trials argument, so it
# lowers with its response omitted and in a `@plate` cell
# (snag prior-draws-refu-aeb3b1e0).

using Test
using BayesianRegressionModels, StanBlocks
using BridgeStan, LogDensityProblems
using Distributions
using LogExpFunctions: logistic
using Random: MersenneTwister
using Statistics: mean

const BRM = BayesianRegressionModels

omit(data, keys::Symbol...) = Base.structdiff(data, NamedTuple{keys})
stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
# Emitted Stan without whitespace, so a call reads the same however it wraps.
flat(sb) = replace(BRM.stan_code(sb), r"\s+" => "")
operation_names(d) = Set(op.name for op in d.operations)

# `n` constrained draws (generated quantities included) at the zero
# unconstrained point, with their names.
function gq_draws(sb, seed, n)
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    rng = BridgeStan.StanRNG(problem.model, seed)
    u = zeros(LogDensityProblems.dimension(problem))
    draws = reduce(hcat, (BridgeStan.param_constrain(problem.model, u;
        include_tp=true, include_gq=true, rng) for _ in 1:n))
    names, draws
end
column(names, draws, name) = draws[findfirst(==(name), names), :]

# The reported shape: a beta-binomial count and a binomial count, each with
# its trials as a data column and its mean through `logistic`.
const counts = @brm begin
    logit_q ~ Normal(0.0, 1.0)
    phi_q ~ LogNormal(1.6, 1.0)
    logit_z ~ Normal(0.0, 1.0)
    k ~ BetaBinomial2(n_k, logistic(logit_q), phi_q)
    m ~ Binomial(n_m, logistic(logit_z))
end
const count_data = (; n_k=[3, 5, 10, 0], n_m=[2, 4, 1, 6],
    k=[1, 2, 6, 0], m=[1, 3, 0, 2])
const bb2_law = "n_k,(inv_logit(logit_q).*phi_q),((1-inv_logit(logit_q)).*phi_q))"
const binomial_law = "n_m,inv_logit(logit_z))"

@testset "omitted BetaBinomial2 and Binomial with data trials draw the prior predictive" begin
    sb = @test_logs (:warn, r"unconditioned \(prior\) program") SBBRMI(
        counts(omit(count_data, :k, :m)); mod=@__MODULE__)
    bound = SBBRMI(counts(count_data); mod=@__MODULE__)
    @test stanc_ok(sb)
    code = flat(sb)
    @test occursin("parameters{}", code)
    # The bound program's predictive draw and the omitted response read one law.
    @test occursin("beta_binomial_int_rng(n_k_n," * bb2_law, code)
    @test occursin(bb2_law, flat(bound))
    @test occursin("binomial_int_rng(n_m_n," * binomial_law, code)
    @test occursin(binomial_law, flat(bound))
    @test !occursin("logistic(", code)
    @test Set(sb.model.observations) == Set((:k, :m))
    for target in (:k, :m)
        declaration = only(d for d in generative_plan(sb).declarations
                           if d.target === target)
        @test declaration.role === :observation
        @test isnothing(declaration.data_source)
    end

    descriptor = brm_descriptor(sb)
    @test :fit ∉ operation_names(descriptor)
    @test :instantiate in operation_names(descriptor)
    @test brm_output(descriptor, :k; role=:posterior_predictive).name === :k_gen
    @test brm_output(descriptor, :m; role=:posterior_predictive).name === :m_gen
    @test Set(brm_operation(descriptor, :predict).outputs) == Set((:k_gen, :m_gen))

    names, draws = gq_draws(sb, 11, 400)
    for (i, trials) in enumerate(count_data.n_k)
        k = column(names, draws, "k_gen.$i")
        @test all(isinteger, k) && all(0 .<= k .<= trials)
    end
    @test all(==(0), column(names, draws, "k_gen.4"))
    for (i, trials) in enumerate(count_data.n_m)
        m = column(names, draws, "m_gen.$i")
        @test all(isinteger, m) && all(0 .<= m .<= trials)
    end
end

@testset "an omitted BetaBinomial2 draws its mean/precision pmf" begin
    literal = @brm begin
        k ~ BetaBinomial2(n, 0.3, 4.0)
    end
    sb = @test_logs (:warn, r"unconditioned") SBBRMI(
        literal((; n=[10, 0])); mod=@__MODULE__)
    names, draws = gq_draws(sb, 7, 8000)
    k = column(names, draws, "k_gen.1")
    empirical = [mean(k .== v) for v in 0:10]
    # Mean 0.3 and precision 4 are Beta(1.2, 2.8) success probabilities.
    @test maximum(abs.(empirical .- pdf.(BetaBinomial(10, 1.2, 2.8), 0:10))) < 0.02
    @test maximum(abs.(empirical .- pdf.(BRM.BetaBinomial2(10, 0.3, 4.0), 0:10))) < 0.02
    # The overdispersion shows: a binomial with the same mean is far off.
    @test maximum(abs.(empirical .- pdf.(Binomial(10, 0.3), 0:10))) > 0.1
    @test all(==(0), column(names, draws, "k_gen.2"))
end

@testset "a fitted Binomial beside an omitted BetaBinomial2" begin
    mixed = @test_logs (:warn, r"`k` bind\(s\) no data column") SBBRMI(
        counts(omit(count_data, :k)); mod=@__MODULE__)
    @test stanc_ok(mixed)
    binomial_only = @brm begin
        logit_z ~ Normal(0.0, 1.0)
        m ~ Binomial(n_m, logistic(logit_z))
    end
    reference = SBBRMI(binomial_only((; n_m=count_data.n_m, m=count_data.m));
                       mod=@__MODULE__)
    p = StanBlocks.stan_instantiate(mixed.model)
    q = StanBlocks.stan_instantiate(reference.model)
    # The omitted count's mean and precision read only it, so they are drawn
    # beside the fit and the density is the binomial's alone.
    @test LogDensityProblems.dimension(p) == LogDensityProblems.dimension(q) == 1
    for u in ([-0.7], [0.0], [1.3])
        @test LogDensityProblems.logdensity(p, u) ≈ LogDensityProblems.logdensity(q, u)
    end
    descriptor = brm_descriptor(mixed)
    @test :fit in operation_names(descriptor)
    @test brm_output(descriptor, :k; role=:posterior_predictive).name === :k_gen
    @test brm_operation(descriptor, :pointwise_loglik).outputs == (:m_likelihood,)
end

const rates = (; x=[-1.0, 0.0, 1.0, 2.0], w=[1.0, 2.0, 0.5, 3.0],
    n=[3, 5, 10, 0], lo=[0, 0, 1, 1], hi=[5, 9, 20, 4], c=[1, 2, 3, 2])

@testset "omitted discrete laws draw from the bound twin's law" begin
    product = @brm begin
        eta ~ 1 + x
        c ~ Poisson(exp(eta) * w)
    end
    logistic_product = @brm begin
        phi ~ LogNormal(1.0, 0.5)
        eta ~ 1 + x
        c ~ NegativeBinomial2(logistic(eta) * w, phi)
    end
    scaled = @brm begin
        a ~ Normal(0.0, 1.0)
        c ~ Poisson(0.25 * logistic(a) * exp(x) * w)
    end
    bounded = @brm begin
        phi ~ LogNormal(1.0, 0.5)
        eta ~ 1 + x
        c ~ truncated(NegativeBinomial2(exp(eta), phi); lower=lo, upper=hi)
    end
    # Each law as the bound program's `c_gen` draws it; the omitted program
    # sizes it by another column (`w_n`, `lo_n`) than `c_n`.
    cases = (
        (product, r"poisson_int_rng\((c|w)_n,\(exp\(eta\)\.\*w\)\)"),
        (logistic_product,
         r"neg_binomial_2_int_rng\((c|w)_n,\(inv_logit\(eta\)\.\*w\),phi\)"),
        (scaled, r"poisson_int_rng\((c|w)_n,\(\(\(0\.25\.\*inv_logit\(a\)\)\.\*exp\(x\)\)\.\*w\)\)"),
        (bounded,
         r"conditioning_int_neg_binomial_2_rng\((c|lo)_n,lo,hi,exp\(eta\),phi\)"),
    )
    for (builder, law) in cases
        omitted = @test_logs (:warn, r"unconditioned") SBBRMI(
            builder(omit(rates, :c)); mod=@__MODULE__)
        bound = SBBRMI(builder(rates); mod=@__MODULE__)
        @test stanc_ok(omitted)
        @test occursin(law, flat(omitted))
        @test occursin(law, flat(bound))
    end
    # Draws stay in each row's window.
    bounded_draws = @brm begin
        phi ~ LogNormal(1.0, 0.5)
        c ~ truncated(NegativeBinomial2(2.0, phi); lower=lo, upper=hi)
    end
    omitted = SBBRMI(bounded_draws(omit(rates, :c)); mod=@__MODULE__)
    @test stanc_ok(omitted)
    names, draws = gq_draws(omitted, 5, 200)
    for i in eachindex(rates.lo)
        c = column(names, draws, "c_gen.$i")
        @test all(rates.lo[i] .<= c .<= rates.hi[i])
    end

    # Families that validate observed values have none to validate.
    for (label, builder, law) in (
            ("BernoulliLogit", @brm(begin
                eta ~ 1 + x
                c ~ BernoulliLogit(eta)
            end), "bernoulli_logit"),
            ("BinomialLogit", @brm(begin
                eta ~ 1 + x
                c ~ BinomialLogit(n, eta)
            end), "binomial_logit"))
        omitted = @test_logs (:warn, r"unconditioned") SBBRMI(
            builder(omit(rates, :c)); mod=@__MODULE__)
        @test stanc_ok(omitted)
        @test occursin(law, flat(omitted))
    end
end

@testset "a continuous omitted response spells its callables as its bound twin" begin
    builder = @brm begin
        s ~ Exponential(1.0)
        eta ~ 1 + x
        y ~ Normal(logistic(eta), s)
    end
    omitted = @test_logs (:warn, r"unconditioned") SBBRMI(
        builder(omit(rates, :c)); mod=@__MODULE__)
    @test stanc_ok(omitted)
    @test occursin("normal_vector_rng(x_n,inv_logit(eta),s)", flat(omitted))
    @test !occursin("logistic(", flat(omitted))
end

# The same law in a `@plate` cell, observed per subject through a binding.
const subjects = (; subject=["b", "a", "c"], weight=[60.0, 75.0, 90.0],
    obs_subject=["a", "c", "b", "a", "b", "c"], n_k=[3, 5, 2, 7, 0, 4],
    k=[1, 2, 2, 3, 0, 1])
plate_counts(data) = @brm data begin
    phi_q ~ LogNormal(1.6, 1.0)
    logit_q ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(logit_q)
        k[obs[i]] ~ BetaBinomial2(n_k[obs[i]], logistic(logit_q[i]), phi_q)
    end
end
# Its Stan family written out: `beta_binomial(n, mu * phi, (1 - mu) * phi)`.
plate_counts_native(data) = @brm data begin
    phi_q ~ LogNormal(1.6, 1.0)
    logit_q ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(logit_q)
        k[obs[i]] ~ beta_binomial(n_k[obs[i]], logistic(logit_q[i]) * phi_q,
                                  (1 - logistic(logit_q[i])) * phi_q)
    end
end

@testset "BetaBinomial2 in a @plate cell, bound and omitted" begin
    bound = SBBRMI(plate_counts(subjects); mod=@__MODULE__)
    native = SBBRMI(plate_counts_native(subjects); mod=@__MODULE__)
    @test stanc_ok(bound)
    p = StanBlocks.stan_instantiate(bound.model)
    q = StanBlocks.stan_instantiate(native.model)
    @test BridgeStan.param_unc_names(p.model) == BridgeStan.param_unc_names(q.model)
    rng = MersenneTwister(3)
    for _ in 1:3
        u = randn(rng, LogDensityProblems.dimension(p))
        @test LogDensityProblems.logdensity(p, u) ≈ LogDensityProblems.logdensity(q, u)
    end

    omitted = @test_logs (:warn, r"`k` bind\(s\) no data column") SBBRMI(
        plate_counts(omit(subjects, :k)); mod=@__MODULE__)
    @test stanc_ok(omitted)
    @test occursin("beta_binomial_int_rng(", flat(omitted))
    problem = StanBlocks.stan_instantiate(omitted.model)
    @test LogDensityProblems.dimension(problem) == 0
end
