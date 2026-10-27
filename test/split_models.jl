# test/split_models.jl — `priors_of` / `structure_of` model partition.
#
# Run: julia --project=test test/split_models.jl
#
# The two accessors split ONE BRMI into its prior fragment and its structural
# fragment (each a BRMI). They are the inverse of `Base.merge(::BRMI, ::BRMI)`:
# `merge(structure_of(b), priors_of(b))` must recompose an equivalent model.
# `show(::BRMI)` prints the model's statements as authored (decision `0eqw9xp`),
# so each fragment prints as its own source card; `d.formula` and the web BRMI
# view print the same source. A data-free `BRMBuilder` splits its source
# statements the same way, into two builders.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        subject=[1, 1, 2, 2, 3, 3],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

# A full model: a linear predictor, a SCALAR parameter declaration
# (`sigma ~ Exponential(1)` — this counts as MODEL, not a prior), a likelihood,
# and the prior-address block (`effect(...)` / `sd(...)`).
inline = @brm df begin
    log_ka ~ 1 + x + (1 | pk | subject)
    sigma ~ Exponential(1)
    y ~ Normal(log_ka, sigma)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, x) ~ Normal(0, 0.1)
    sd(:, pk) ~ Exponential(1)
end

stan(brmi) = BayesianRegressionModels.stan_code(SBBRMI(brmi; mod=@__MODULE__))
opkeys(brmi) = collect(keys(brmi.operations))

@testset "priors_of / structure_of partition the operations" begin
    priors = priors_of(inline)
    structure = structure_of(inline)
    @test priors isa BayesianRegressionModels.BRMI
    @test structure isa BayesianRegressionModels.BRMI

    # Disjoint, exhaustive cover: every op lands in exactly one fragment, and
    # each fragment keeps the original relative order of its ops.
    @test isempty(intersect(opkeys(priors), opkeys(structure)))
    @test sort(vcat(opkeys(priors), opkeys(structure))) == sort(opkeys(inline))
    @test opkeys(priors) == filter(in(Set(opkeys(priors))), opkeys(inline))
    @test opkeys(structure) == filter(in(Set(opkeys(structure))), opkeys(inline))

    # priors_of holds exactly the prior-address statements.
    @test [(p.predictor, p.coefficient, p.family, p.arguments)
           for p in effect_priors(priors)] ==
          [(p.predictor, p.coefficient, p.family, p.arguments)
           for p in effect_priors(inline)]
    @test length(ranef_effect_priors(priors)) == 1

    # structure_of holds NO prior-address statements...
    @test isempty(effect_priors(structure))
    @test isempty(ranef_effect_priors(structure))
    # ...but KEEPS the scalar `sigma ~ Exponential(1)` (declares a param, not a
    # prior). Its key is the bare `sigma`, present in structure, absent in priors.
    @test :sigma in opkeys(structure)
    @test :sigma ∉ opkeys(priors)
    @test :log_ka in opkeys(structure)   # the linear predictor stays with the model
end

@testset "print views: suppressed vs only" begin
    # structure_of prints the model WITHOUT the prior block ("suppressed").
    structure_txt = sprint(show, structure_of(inline))
    @test occursin("log_ka", structure_txt)
    @test occursin("sigma", structure_txt)
    @test !occursin("effect(", structure_txt)
    @test !occursin("sd(", structure_txt)

    # priors_of prints ONLY the prior block ("set to the only thing"). Every
    # prior statement prints the way it was written: the `sd(:, pk)` prior as
    # `sd(:, pk)`, never as its internal `effect(sd, pk)` carrier.
    priors_txt = sprint(show, priors_of(inline))
    @test occursin("effect(", priors_txt)
    @test occursin("sd(:, pk) ~ ", priors_txt)
    @test !occursin("effect(sd", priors_txt)
    @test !occursin("Normal(log_ka", priors_txt)  # no likelihood
    @test !occursin("1 + x", priors_txt)          # no linear predictor

    # The full listing interleaves both.
    full_txt = sprint(show, inline)
    @test occursin("effect(", full_txt)
    @test occursin("log_ka", full_txt)
end

@testset "split is the inverse of merge" begin
    # merge(structure, priors) recomposes an equivalent model: same Stan.
    recomposed = Base.merge(structure_of(inline), priors_of(inline))
    @test StanBlocks.stan.transpiles(SBBRMI(recomposed; mod=@__MODULE__).model)
    @test stan(recomposed) == stan(inline)

    # Non-vacuous: structure alone (priors defaulted) lowers differently.
    @test stan(structure_of(inline)) != stan(inline)

    # Idempotent / disjoint: re-splitting a fragment is a no-op, and the prior
    # fragment has no structural part (and vice versa).
    @test opkeys(priors_of(priors_of(inline))) == opkeys(priors_of(inline))
    @test opkeys(structure_of(structure_of(inline))) == opkeys(structure_of(inline))
    @test isempty(opkeys(priors_of(structure_of(inline))))
    @test isempty(opkeys(structure_of(priors_of(inline))))
end

# A data-free builder splits its source the same way, so a listing shows the
# structure and prior cards without binding data (snag show-a-model-s-s-596039d4).
@testset "a builder splits its source without data" begin
    ranef_df = merge(df, (; z=[1, 2, 3, 1, 2, 3]))
    builder = @brm begin
        log_ka ~ 1 + x + mo1(z) + (1 + x | pk | subject)
        effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
        sigma ~ Exponential(1)
        optional(effect(log_ka, weight)) ~ Normal(0, 0.1)
        sd(:, pk) ~ Exponential(1)
        y ~ Normal(log_ka, sigma)
        cor(:, pk) ~ LKJCholesky(2, 2)
        ranef(:, pk) ~ TDist(4)
        simplex(log_ka, mo1(z)) ~ Dirichlet(2, 1)
        effect(log_ka, x) .~ Normal.(0.0, 0.5)
        sd(log_ka, pk, x) = 0.3
    end
    source = deepcopy(builder.source)
    structure, priors = structure_of(builder), priors_of(builder)
    @test structure isa BRMBuilder
    @test priors isa BRMBuilder
    @test split(sprint(show, structure), '\n') == [
        "log_ka ~ 1 + x + mo1(z) + (1 + x | pk | subject)",
        "sigma ~ Exponential(1)", "y ~ Normal(log_ka, sigma)"]
    # Every prior head, the optional marker, an elementwise prior and a known
    # value, each as written (`.~` infix, as the built model prints it).
    @test split(sprint(show, priors), '\n') == [
        "effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)",
        "optional(effect(log_ka, weight)) ~ Normal(0, 0.1)",
        "sd(:, pk) ~ Exponential(1)", "cor(:, pk) ~ LKJCholesky(2, 2)",
        "ranef(:, pk) ~ TDist(4)", "simplex(log_ka, mo1(z)) ~ Dirichlet(2, 1)",
        "effect(log_ka, x) .~ Normal.(0.0, 0.5)", "sd(log_ka, pk, x) = 0.3"]
    @test builder.source == source

    # The source test and the operation test split alike: the builder's cards
    # are the built model's.
    built = builder(ranef_df)
    @test sprint(show, builder) == sprint(show, built)
    @test sprint(show, structure) == sprint(show, structure_of(built))
    @test sprint(show, priors) == sprint(show, priors_of(built))
    @test brm_columns(structure) == brm_columns(builder)
    @test stan(structure(ranef_df)) == stan(structure_of(built))
    @test stan(merge(structure, priors)(ranef_df)) == stan(built)

    # A declared vector parameter and an r2d2 decomposition: the first declares
    # a parameter (structure), the second is a prior address.
    declared = @brm begin
        phi[1:2] .~ Normal.(0.0, 0.5; lower = 0.0)
        mu ~ 1 + x
        effect(mu, :) ~ r2d2(R2 = 0.5)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test sprint(show, structure_of(declared)) ==
          "phi[1:2] .~ Normal.(0.0, 0.5; lower=0.0)\nmu ~ 1 + x\n" *
          "sigma ~ Exponential(1)\ny ~ Normal(mu, sigma)"
    @test sprint(show, priors_of(declared)) == "effect(mu, :) ~ r2d2(; R2=0.5)"
    @test sprint(show, declared) == sprint(show, declared(df))

    # A `begin … end` statement in a `brm(source; mod)` body is spliced in, as
    # the parser splices it.
    fragment = quote
        effect(mu, x) ~ Normal(0, 0.5)
        sigma ~ Exponential(1)
    end
    nested = brm(quote
        mu ~ 1 + x
        $fragment
        y ~ Normal(mu, sigma)
    end; mod=@__MODULE__)
    @test sprint(show, structure_of(nested)) ==
          "mu ~ 1 + x\nsigma ~ Exponential(1)\ny ~ Normal(mu, sigma)"
    @test sprint(show, priors_of(nested)) == "effect(mu, x) ~ Normal(0, 0.5)"
    @test sprint(show, structure_of(nested)) == sprint(show, structure_of(nested(df)))

    # A model without priors has an empty prior card.
    plain = @brm begin sigma ~ Exponential(1); mu ~ 1 + x; y ~ Normal(mu, sigma) end
    @test sprint(show, priors_of(plain)) == ""
    @test brm_columns(priors_of(plain)) == ()
    @test sprint(show, structure_of(plain)) == sprint(show, plain)
end

println("split_models.jl: all testsets passed")
