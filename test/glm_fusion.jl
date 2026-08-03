# test/glm_fusion.jl — pure-population Gaussian likelihoods lower to Stan's
# fused `normal_id_glm_lpdf`.
#
# Run: julia --startup-file=no --project=test test/glm_fusion.jl
# Set BRM_GLM_FUSION_RUNTIME=0 to skip the BridgeStan gradient-equivalence gate.
#
# Two coupled emission changes are under test, and BOTH matter for the measured
# ~3.8x gradient speedup at N=2000, K=4 (`bench/sbbrmi_emission/`):
#
#   1. the likelihood becomes `y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma)`,
#      whose hand-written gradient never materialises `X * beta` on the tape;
#   2. the linear predictor `mu = X_mu * pop_mu` is emitted AFTER the
#      likelihood, so it lands in `generated quantities` rather than
#      `transformed parameters` — off the gradient path entirely.
#
# The fusion is guarded to the case where it is provably equivalent: a
# data-backed single Gaussian response, an identity link, a linear predictor
# built only from population terms, and that predictor read by exactly one
# likelihood and by nothing else. Everything else must keep its previous
# emission byte for byte, which is what the "declines" testset pins.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Exponential, Normal, Poisson, truncated
using Random

const BRM = BayesianRegressionModels
const FUSION_CACHE = joinpath(tempdir(), "brm-glm-fusion")
const FUSION_RUNTIME = get(ENV, "BRM_GLM_FUSION_RUNTIME", "1") != "0"

df = (; x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        z = [0.2, -0.7, 1.1, 0.4, -0.3, 0.9],
        g = [1, 1, 2, 2, 3, 3],
        yc = [0, 1, 2, 1, 3, 2],
        y = [-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
        y2 = [0.3, -0.1, 0.6, 0.2, -0.4, 0.8])

emit(brmi) = BRM.stan_code(SBBRMI(brmi; mod = @__MODULE__))

@testset "pure-population Gaussian fuses to normal_id_glm" begin
    brmi = @brm df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + z
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(brmi; mod = @__MODULE__)
    code = BRM.stan_code(sb)

    @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    @test occursin("y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma);", code)
    @test !occursin("y ~ normal(mu, sigma);", code)

    # `pop_<lp>` now carries the COEFFICIENTS, so it is a K-vector alias in
    # transformed parameters rather than an N-vector product.
    @test occursin("vector[pop_mu_n_covariates] pop_mu = pop_mu_beta_pop;", code)
    @test !occursin("pop_mu = (X_mu * pop_mu_beta_pop)", code)

    # The parameter name is the `popcoefnames` public contract and must not move.
    @test occursin("vector[pop_mu_n_covariates] pop_mu_beta_pop;", code)
    @test occursin("pop_mu_beta_pop ~ std_normal();", code)
    @test popcoefnames(brmi, :mu) == [:Intercept, :x, :z]

    # The linear predictor survives as a reported quantity — in generated
    # quantities, which is the second half of the speedup.
    gq = code[findfirst("generated quantities {", code)[1]:end]
    @test occursin("mu = (X_mu * pop_mu)", gq)
    tp = code[findfirst("transformed parameters {", code)[1]:findfirst("model {", code)[1]]
    @test !occursin(" mu = ", tp)

    # Both predictive twins are synthesised against the fused primitive.
    @test occursin("y_likelihood = normal_id_glm_lpdfs(y, X_mu, 0.0, pop_mu, sigma)", gq)
    @test occursin("y_gen = normal_id_glm_vector_rng(y_n, X_mu, 0.0, pop_mu, sigma)", gq)

    # Descriptor roles and coefficient labels ride the coefficient-returning
    # sibling submodel exactly as they rode `popefs`.
    plan = generative_plan(sb)
    pop_decl = only(d for d in plan.declarations if d.target === :pop_mu)
    @test pop_decl.family === :_popefs_coefs
    d = brm_descriptor(sb)
    byname = Dict(o.name => o for o in d.outputs)
    @test byname[:pop_mu].role === :population_effect
    @test byname[:pop_mu_beta_pop].role === :population_effect
    @test byname[:pop_mu_beta_pop].labels == [:Intercept, :x, :z]
    @test byname[:mu].role === :linear_predictor
end

@testset "intercept-only, effect priors, dpar scale and multiple responses" begin
    # A single design column is a valid `normal_id_glm` matrix.
    intercept = emit(@brm df begin
        sigma ~ Exponential(1)
        mu ~ 1
        y ~ Normal(mu, sigma)
    end)
    @test StanBlocks.stanc_check(intercept; warn_pedantic = false).ok
    @test occursin("y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma);", intercept)

    # Named coefficient priors route through the `_popefs_normal` sibling.
    priors = emit(@brm df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, 0.25)
        y ~ Normal(mu, sigma)
    end)
    @test StanBlocks.stanc_check(priors; warn_pedantic = false).ok
    @test occursin("y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma);", priors)
    @test occursin("pop_mu_beta_pop ~ normal(", priors)
    @test occursin("[1.0, 0.25]'", priors)

    # A vector scale from a distributional-parameter formula is a valid
    # `normal_id_glm` signature. Its OWN predictor is not fusible (a following
    # `sigma = exp(log_sigma)` reads it), so exactly one of the two fuses.
    dpar = emit(@brm df begin
        log(sigma) ~ 1 + z
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)
    @test StanBlocks.stanc_check(dpar; warn_pedantic = false).ok
    @test occursin("y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma);", dpar)
    @test occursin("pop_log_sigma = (X_log_sigma * pop_log_sigma_beta_pop)", dpar)

    # Each qualifying response fuses independently.
    multi = emit(@brm df begin
        s1 ~ Exponential(1)
        s2 ~ Exponential(1)
        mu ~ 1 + x
        nu ~ 1 + z
        y ~ Normal(mu, s1)
        y2 ~ Normal(nu, s2)
    end)
    @test StanBlocks.stanc_check(multi; warn_pedantic = false).ok
    @test occursin("y ~ normal_id_glm(X_mu, 0.0, pop_mu, s1);", multi)
    @test occursin("y2 ~ normal_id_glm(X_nu, 0.0, pop_nu, s2);", multi)
end

@testset "the fusion declines everything it is not provably equivalent on" begin
    # A random effect adds a second summand to the linear predictor.
    ranef = emit(@brm df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        y ~ Normal(mu, sigma)
    end)
    @test occursin("y ~ normal(mu, sigma);", ranef)
    @test occursin("pop_mu = (X_mu * pop_mu_beta_pop)", ranef)
    @test !occursin("normal_id_glm", ranef)

    # A non-identity LHS link: the emitted predictor is read by the inverse-link
    # assignment as well as by the likelihood.
    linked = emit(@brm df begin
        sigma ~ Exponential(1)
        log(mu) ~ 1 + x
        y ~ Normal(mu, sigma)
    end)
    @test !occursin("normal_id_glm", linked)

    # A predictor consumed by two likelihoods cannot move past either of them.
    shared = emit(@brm df begin
        s1 ~ Exponential(1)
        s2 ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, s1)
        y2 ~ Normal(mu, s2)
    end)
    @test !occursin("normal_id_glm", shared)

    # Non-Gaussian families lower to a different Stan name entirely.
    pois = emit(@brm df begin
        log(lam) ~ 1 + x
        yc ~ Poisson(lam)
    end)
    @test !occursin("normal_id_glm", pois)

    # Truncation wraps the density; the plain two-arg Gaussian never appears.
    trunc = emit(@brm df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ truncated(Normal(mu, sigma); lower = -4.0, upper = 4.0)
    end)
    @test !occursin("normal_id_glm", trunc)

    # An integer-valued Gaussian response declares as `array[] int`, which plain
    # `normal` accepts and `normal_id_glm_lpdf` does not. `transpiles()` cannot
    # see the difference, so this case is pinned at stanc.
    ints = @brm df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        yc ~ Normal(mu, sigma)
    end
    int_code = emit(ints)
    @test !occursin("normal_id_glm", int_code)
    @test StanBlocks.stanc_check(int_code; warn_pedantic = false).ok

    # Direct (non-population) summands such as `offset(...)` add a second term
    # to the predictor and take it out of the fusible class.
    off = emit(@brm df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + offset(z)
        y ~ Normal(mu, sigma)
    end)
    @test !occursin("normal_id_glm", off)
end

# ---- gradient equivalence against an unfused hand-written reference ---------
#
# Same parameters in the same order (`sigma`, then the K coefficients), same
# priors, same design matrix — so the two unconstrained spaces are identical
# and the gradients are directly comparable. This is the gate `transpiles()`
# cannot give: a real stanc + C++ compile through BridgeStan.
if FUSION_RUNTIME
    @testset "fused density and gradient match the unfused reference" begin
        isdir(FUSION_CACHE) || mkpath(FUSION_CACHE)
        rng = MersenneTwister(11)
        N = 200
        x = randn(rng, N); z = randn(rng, N)
        y = 0.4 .+ 0.8 .* x .- 0.3 .* z .+ 0.5 .* randn(rng, N)
        d = (; x, z, y)

        fused = SBBRMI(@brm d begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + z
            y ~ Normal(mu, sigma)
        end; mod = @__MODULE__).model
        @test occursin("normal_id_glm", StanBlocks.stan_code(fused))

        # Distinct names, read back off `d`: `@brm`'s formula scope rebinds the
        # bare column symbols to `NamedColumn`s, which are not StanBlocks data.
        rx, rz, ry = d.x, d.z, d.y
        reference = StanBlocks.@slic (; rx, rz, ry) begin
            sigma ~ exponential(1.0)
            X = hcat(rep_vector(1.0, num_elements(rx)), rx, rz)
            beta ~ std_normal(; n = dims(X)[2])
            ry ~ normal(X * beta, sigma)
        end

        problems = map((fused, reference)) do m
            code = StanBlocks.stan_code(m)
            StanBlocks.stan_instantiate(m;
                path = joinpath(FUSION_CACHE, string(hash(code)) * ".stan"))
        end
        @test LogDensityProblems.dimension(problems[1]) ==
              LogDensityProblems.dimension(problems[2]) == 4

        for seed in 1:5
            q = 0.4 .* randn(MersenneTwister(seed), 4)
            lps, grads = zip(map(p -> LogDensityProblems.logdensity_and_gradient(p, q),
                                 problems)...)
            @test all(isfinite, lps)
            # Reassociating the sum is a float64 round-off difference, nothing more.
            @test isapprox(lps[1], lps[2]; rtol = 1e-10)
            @test isapprox(grads[1], grads[2]; rtol = 1e-8, atol = 1e-10)
        end
    end
else
    @info "Skipping BridgeStan gradient-equivalence gate (BRM_GLM_FUSION_RUNTIME=0)"
end
