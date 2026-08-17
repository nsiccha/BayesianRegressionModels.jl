# test/merge_models.jl — `Base.merge(::BRMI, ::BRMI)` model composition.
#
# Run: julia --project=test test/merge_models.jl
#
# The headline contract: a structural model and a separate prior fragment,
# composed with `Base.merge`, must be BYTE-IDENTICAL after lowering to the same
# priors written inline — post-hoc priors take effect exactly as inline ones.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        subject=[1, 1, 2, 2, 3, 3],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

# Priors written inline (statements after the structural model so the operation
# key order matches what `merge` produces: model keys, then appended priors).
inline = @brm df begin
    log_ka ~ 1 + x + (1 | pk | subject)
    y ~ Normal(log_ka, 0.2)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, x) ~ Normal(0, 0.1)
    sd(:, pk) ~ Exponential(1)
end

# The same model split into a structural fragment + a prior fragment.
structure = @brm df begin
    log_ka ~ 1 + x + (1 | pk | subject)
    y ~ Normal(log_ka, 0.2)
end
priors = @brm df begin
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, x) ~ Normal(0, 0.1)
    sd(:, pk) ~ Exponential(1)
end

stan(brmi) = BayesianRegressionModels.stan_code(SBBRMI(brmi; mod=@__MODULE__))

@testset "merge composes structure + priors" begin
    merged = Base.merge(structure, priors)
    @test merged isa BayesianRegressionModels.BRMI

    # Structural combine: all structure keys kept in order, prior keys appended.
    @test collect(keys(merged.operations)) ==
          vcat(collect(keys(structure.operations)),
               collect(keys(priors.operations)))

    # A prior fragment alone captures its priors with symbolic (unresolved)
    # addresses — nothing needs the structural model to be present yet.
    @test length(effect_priors(priors)) == 2
    @test length(ranef_effect_priors(priors)) == 1

    # Priors survive the merge, identical to the inline model's.
    @test [(p.predictor, p.coefficient, p.family, p.arguments)
           for p in effect_priors(merged)] ==
          [(p.predictor, p.coefficient, p.family, p.arguments)
           for p in effect_priors(inline)]
    @test [(p.class, p.id, p.predictor, p.coefficient, p.family, p.arguments)
           for p in ranef_effect_priors(merged)] ==
          [(p.class, p.id, p.predictor, p.coefficient, p.family, p.arguments)
           for p in ranef_effect_priors(inline)]

    # THE HEADLINE: post-hoc priors lower to the same Stan as inline priors.
    @test StanBlocks.stan.transpiles(SBBRMI(merged; mod=@__MODULE__).model)
    @test stan(merged) == stan(inline)

    # And they genuinely took effect: the priorless structure lowers differently
    # (default std_normal / half-normal), so the equality above is not vacuous.
    @test stan(structure) != stan(inline)
    @test occursin(string(log(1 / 8)), stan(merged))
end

@testset "post-hoc override is last-wins by address" begin
    override = @brm df begin
        effect(log_ka, Intercept) ~ Normal(0.0, 5.0)
    end
    result = Base.merge(inline, override)

    caps = effect_priors(result)
    # Still exactly two effect priors — the override replaced, did not append.
    @test length(caps) == 2
    intercept = only(p for p in caps if p.coefficient === :Intercept)
    @test intercept.arguments == (0.0, 5.0)         # the override, not log(1/8), 0.8
    @test occursin("[5.0, 0.1]'", stan(result))     # override scale reaches Stan
    @test !occursin(string(log(1 / 8)), stan(result))  # original intercept loc gone
end

@testset "variadic merge folds left" begin
    a = @brm df begin
        log_ka ~ 1 + x + (1 | pk | subject)
        y ~ Normal(log_ka, 0.2)
    end
    b = @brm df begin
        effect(log_ka, Intercept) ~ Normal(0, 1)
    end
    c = @brm df begin
        effect(log_ka, Intercept) ~ Normal(2, 3)    # overrides b
        effect(log_ka, x) ~ Normal(0, 0.1)
    end
    folded = Base.merge(a, b, c)
    caps = effect_priors(folded)
    @test length(caps) == 2
    @test only(p for p in caps if p.coefficient === :Intercept).arguments == (2, 3)
end

println("merge_models.jl: all testsets passed")
