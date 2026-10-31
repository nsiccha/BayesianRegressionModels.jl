# NLMEEstimation's integrated families on BRM models
# (ext/BayesianRegressionModelsNLMEEstimationExt.jl): SAEM integrates chosen
# covariance families by BRM's sparse Laplace (src/nlme_laplace_sparse.jl) and
# samples the rest. Checked: `check_protocol` with each set of families
# integrated, on a Normal and a Poisson nested model (values, modes and
# covariance blocks against NLMEEstimation's own Laplace; gradients against
# central differences); Laplace-EM on a Gaussian nested panel against the
# direct maximum of the same objective; SAEM sampling the subjects with the
# units integrated; brm_ladder_fit's rung 3 (SAEM from rung 2's point).
#
# NLMEEstimation.jl is not part of test/Project.toml. Run this file in the
# NLMEEstimation scratch environment (test/README.md, "NLME estimators"):
#     julia --project=<scratch-env> test/nlme_integrated.jl
Base.find_package("NLMEEstimation") === nothing && error(
    "test/nlme_integrated.jl needs NLMEEstimation.jl in the active environment; " *
    "see the header of this file")

using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Random
using WarmupHMC
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
# Optim is not a direct test dependency; WarmupHMC loads it through Pathfinder.
const Optim = Base.root_module(Base.PkgId(
    Base.UUID("429524aa-4258-5aef-a3af-852621145aeb"), "Optim"))

# `nested_model` (nlme_fixtures.jl) with a Poisson count per row.
nested_poisson(data) = @brm data begin
    mu ~ 1 + (1 | p | subject)
    unit_shift ~ 0 + (1 | u | unit)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(mu)
        pred[i] = mu[i] .+ unit_shift[obs[i]]
        y[obs[i]] ~ Poisson(exp.(pred[i]))
    end
end

# `ns` subjects with `units` nested units of `rows` rows each, interleaved.
function nested_panel(ns; units=3, rows=3, counts=false, rng=Xoshiro(5))
    subject = ["s$i" for i in 1:ns]
    obs_subject = String[]; unit = String[]; y = counts ? Int[] : Float64[]
    for i in 1:ns
        a = 0.5randn(rng)
        for k in 1:units
            b = 0.3randn(rng)
            for _ in 1:rows
                push!(obs_subject, "s$i"); push!(unit, "u$(i)_$k")
                η = 0.4 + a + b
                push!(y, counts ? rand(rng, Poisson(exp(η))) : η + 0.4randn(rng))
            end
        end
    end
    order = randperm(rng, length(y))
    (; subject, obs_subject=obs_subject[order], unit=unit[order], y=y[order])
end

nested_nlme(build, data) = brm_nlme_model(RKBRMI(build(data)); ad_backend=NLME_AD,
    subject_group=:subject, nested_groups=(unit=:obs_subject,))

@stestset "check_protocol with each set of families integrated" begin
    rng = Xoshiro(3)
    for (build, data) in ((nested_model, NESTED_DATA),
            (nested_poisson, nested_panel(4; counts=true, rng)))
        m = nested_nlme(build, data)
        layout = NE.nlme_layout(m)
        @test layout isa NE.RaggedNLMELayout && length(layout.family_sizes) == 2
        θ = 0.2 .* randn(rng, length(m.theta)); σ = 0.2 .* randn(rng, length(m.sigma))
        ω = 0.3 .* randn(rng, NE.nomega(layout))
        H = [0.3 .* randn(rng, length(c)) for c in m.view.subject_coordinates]
        for integrated in ([2], [1], [1, 2])
            @test NE.check_protocol(m, θ, σ, ω; integrated, η=H)
        end
        # one problem per set of families, built once
        @test sort(collect(keys(m.laplace))) == [[1], [1, 2], [2]]
    end
end

@stestset "Laplace-EM reaches the maximum of the same objective" begin
    data = nested_panel(20)
    m = nested_nlme(nested_model, data)
    n = NE.nsubjects(m)
    nθ, nσ = length(m.theta), length(m.sigma)
    start = (zeros(nθ), zeros(nσ), zeros(NE.nomega(NE.nlme_layout(m))))
    H0 = [zeros(length(c)) for c in m.view.subject_coordinates]
    # The fit's objective: the Laplace marginals (exact here, the model is
    # Gaussian in its effects) and the population prior.
    function objective(x)
        θ, σ, ω = x[1:nθ], x[nθ+1:nθ+nσ], x[nθ+nσ+1:end]
        ℓ, gθ, gσ, gω = NE.laplace_conditionals_and_population_gradient(m, [1, 2],
            1:n, θ, σ, ω, H0)
        lp, pθ, pσ, pω = NE.population_logprior_and_gradient(m, θ, σ, ω)
        sum(ℓ) + lp, vcat(gθ .+ pθ, gσ .+ pσ, gω .+ pω)
    end
    result = Optim.optimize(x -> -first(objective(x)), (G, x) -> (G .= .-last(objective(x))),
        vcat(start...), Optim.LBFGS(), Optim.Options(; g_abstol=1e-9, iterations=1000))
    x̂ = Optim.minimizer(result)
    best = first(objective(x̂))
    @test norm(last(objective(x̂))) < 1e-5
    fit = saem(m, start...; integrated=[1, 2], options=SAEMOptions(iterations=400))
    @test fit.kernel == :laplace_em && fit.integrated == [1, 2]
    @test fit.objective == :MAP
    x = vcat(fit.theta, fit.sigma, fit.omega)
    @test best - first(objective(x)) < 1e-6
    @test maximum(abs, x - x̂) < 1e-3
    # The covariance step takes the Laplace-gradient hook: 2p + 1 calls, its R
    # the central difference of the objective's gradient.
    @test NE.provides_laplace_gradient(m)
    step = NE.covariance_step(m, x̂[1:nθ], x̂[nθ+1:nθ+nσ], x̂[nθ+nσ+1:end];
        method=:r, approximation=NE.AdaptiveGaussHermite(nodes=1), η0=H0)
    p = length(x̂)
    @test step.evaluations == 2p + 1
    hessian = reduce(hcat, (last(objective(x̂ .+ 1e-4 .* e)) .-
        last(objective(x̂ .- 1e-4 .* e))) ./ 2e-4 for e in eachcol(Matrix(1.0I, p, p)))
    @test step.R ≈ -Symmetric((hessian + hessian') / 2) rtol=1e-3
end

@stestset "SAEM sampling the subjects, the units integrated" begin
    data = nested_panel(20)
    m = nested_nlme(nested_model, data)
    n = NE.nsubjects(m)
    start = (zeros(length(m.theta)), zeros(length(m.sigma)),
        zeros(NE.nomega(NE.nlme_layout(m))))
    H0 = [zeros(length(c)) for c in m.view.subject_coordinates]
    marginal(fit) = sum(NE.laplace_conditionals_and_population_gradient(m, [1, 2], 1:n,
        fit.theta, fit.sigma, fit.omega, H0)[1]) +
        NE.population_logprior(m, fit.theta, fit.sigma, fit.omega)
    em = saem(m, start...; integrated=[1, 2], options=SAEMOptions(iterations=400))
    fit = saem(m, start...; integrated=[2], rng=Xoshiro(7),
        options=SAEMOptions(iterations=300))
    @test fit.integrated == [2] && fit.kernel != :laplace_em
    # within half a log-likelihood unit of the Laplace-EM maximum
    @test marginal(em) - marginal(fit) < 0.5
end

@stestset "brm_ladder_fit's rung 3: SAEM over the units from rung 2's point" begin
    data = nested_panel(20)
    rk = RKBRMI(nested_model(data))
    fit = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=4, draws=20, rng=Xoshiro(5),
        subject_group=:subject, nested_groups=(unit=:obs_subject,),
        saem_options=(SAEMOptions(iterations=100), SAEMOptions(iterations=100)))
    @test [r.rung for r in fit.rungs] == [1, 2, 3, 4]
    c = fit.ladder[3]
    # every family is Normal-read, so the plan integrates both and SAEM is
    # Laplace-EM
    @test c.kernel == :laplace_em && c.integrated == [1, 2]
    # rung 2 is the Laplace target's maximum; rung 3's MAP, taken in
    # NLMEEstimation's coordinates, sits at or below it
    @test isfinite(c.gap) && c.gap >= -1e-6
    @test fit.point == fit.rungs[3].point
    @test all(fit.draws[fit.problem.population, k] == fit.point for k in 1:20)
    phases = fit.rungs[3].phases
    @test c.phases == length(phases) == 2 && c.iterations == 200
    @test phases[1].start == vec(fit.rungs[2].draws)
    @test phases[2].start == phases[1].point
    @test phases[1].plan !== phases[2].plan && phases[2].plan === fit.plan
    @test Set(b.declaration for b in fit.problem.blocks) ==
        Set(t.declaration for t in fit.plan.terms if t.admitted)

    # Explicit groupings have no automatic plan. They still tell SAEM which
    # families to integrate, including when units are supplied on continuation.
    first = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=4, draws=2,
        integrate=(:subject, :unit), rng=Xoshiro(8))
    @test first.plan === nothing && first.ladder[3].skipped
    continued = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=4, draws=20,
        continue_from=first, subject_group=:subject,
        nested_groups=(unit=:obs_subject,), rng=Xoshiro(9),
        saem_options=SAEMOptions(iterations=200))
    @test continued.plan === nothing && !continued.automatic
    @test continued.ladder[3].integrated == [1, 2]
    @test continued.ladder[3].kernel == :laplace_em
    @test continued.point == continued.rungs[3].point
    @test all(continued.draws[continued.problem.population, k] == continued.point for k in 1:20)
    @test first.ladder[3].skipped && size(first.draws, 2) == 2
end
