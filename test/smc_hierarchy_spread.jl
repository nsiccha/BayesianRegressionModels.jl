# Tuned RK-only SMC on a two-group hierarchy against an exact grid oracle.
#
# The population intercept m and log scale l of `mu ~ 1 + (1 | g)` are weakly
# identified by two groups, so the bridge is funnel-shaped there while the group
# effects are well identified. The oracle integrates the exact marginal posterior
# of (m, l) on a grid and mixes the Gaussian group effects over it; it is never
# supplied to the sampler. Positive checks: the continuous diagonal pilot and the
# adaptive pilot with the standard-normal reference reproduce the oracle's
# spread. RKPPL 0.1.47 repairs learned references by fitting retained pilot
# moments with separate inflation (ReactiveKernels snag learned-smc-refe-cc5677a1).
# Check the reference repair directly and retain the known population-intercept
# spread deficit: the repaired reference alone does not establish coverage.
using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random, Statistics, Printf, Test

const SIGMA = 0.5
const PRIOR_M = (0.0, 1.0)      # effect(mu, Intercept) ~ Normal(0, 1)
const PRIOR_L = (-1.0, 1.0)     # sd(mu, g) ~ LogNormal(-1, 1), so log tau ~ Normal(-1, 1)

hierarchy(d) = @brm d begin
    mu ~ 1 + (1 | g)
    effect(mu, Intercept) ~ Normal(0, 1)
    sd(mu, g) ~ LogNormal(-1, 1)
    y ~ Normal(mu, 0.5)
end

function synthetic(; means=(0.3, -0.1), n=48)
    pattern = SIGMA .* cos.(2pi .* (0:n-1) ./ n)
    pattern .-= mean(pattern)                      # exact zero mean: group means are exact
    (; g=vcat(fill("a", n), fill("b", n)), y=vcat(means[1] .+ pattern, means[2] .+ pattern))
end

function oracle(d; mgrid=range(-2.5, 2.5; length=801), lgrid=range(-9.0, 3.0; length=961))
    groups = unique(d.g)
    ybar = [mean(d.y[d.g .== k]) for k in groups]
    ng = [count(==(k), d.g) for k in groups]
    logp = [begin
            tau2 = exp(2l)
            lp = logpdf(Normal(PRIOR_M...), m) + logpdf(Normal(PRIOR_L...), l)
            for k in eachindex(groups)
                v = tau2 + SIGMA^2 / ng[k]
                lp += -0.5 * log(2pi * v) - (ybar[k] - m)^2 / (2v)
            end
            lp
        end for m in mgrid, l in lgrid]
    w = exp.(logp .- maximum(logp)); w ./= sum(w)
    pm = vec(sum(w; dims=2)); pl = vec(sum(w; dims=1))
    mean_m = sum(pm .* mgrid); sd_m = sqrt(sum(pm .* (mgrid .- mean_m) .^ 2))
    mean_l = sum(pl .* lgrid); sd_l = sqrt(sum(pl .* (lgrid .- mean_l) .^ 2))
    K = length(groups)
    mean_z = zeros(K); second_z = zeros(K); mean_b = zeros(K); second_b = zeros(K)
    for (j, l) in enumerate(lgrid), (i, m) in enumerate(mgrid)
        tau = exp(l); tau2 = tau^2; wij = w[i, j]
        for k in 1:K
            prec = 1 / tau2 + ng[k] / SIGMA^2
            bmean = (m / tau2 + ng[k] * ybar[k] / SIGMA^2) / prec
            bvar = 1 / prec
            ez = (bmean - m) / tau; vz = bvar / tau2
            mean_z[k] += wij * ez; second_z[k] += wij * (vz + ez^2)
            mean_b[k] += wij * bmean; second_b[k] += wij * (bvar + bmean^2)
        end
    end
    (; groups, ybar, ng, mean_m, sd_m, mean_l, sd_l,
        mean_z, sd_z=sqrt.(second_z .- mean_z .^ 2),
        mean_b, sd_b=sqrt.(second_b .- mean_b .^ 2))
end

wmean(X, w, i) = sum(w .* view(X, i, :))
wsd(X, w, i) = (m = wmean(X, w, i); sqrt(sum(w .* (view(X, i, :) .- m) .^ 2)))

function coordinate_indices(rk)
    names = String.(coordinate_names(rk.model.layout))
    il = findfirst(n -> occursin(".tau.", n), names)
    iz = findall(n -> occursin(".z.", n), names)
    pop = findall(n -> occursin("Intercept", n) || startswith(n, "pop_"), names)
    length(pop) == 1 || error("expected one population intercept coordinate among $names, found $(names[pop])")
    im = only(pop)
    il === nothing && error("no scale coordinate among $names")
    length(iz) == 2 || error("expected two z coordinates among $names")
    (; names, im, il, iz)
end

function run_bridge(rk, idx, options; seed, particles)
    fit = brm_smc_fit(rk; rng=Xoshiro(seed), particles, options, stages=0)
    rows = NamedTuple[]
    let X = fit.state.positions, w = fit.state.weights
        push!(rows, (; stage=0, temperature=0.0, ess_before_resampling=NaN, unique_parents=0,
            surviving_ancestors=length(w), divergences=0, mean_acceptance=nothing, mean_squared_movement=NaN,
            evaluations=0, tuning_index=0, sd_m=wsd(X, w, idx.im), sd_l=wsd(X, w, idx.il),
            mean_l=wmean(X, w, idx.il), lineages=length(Set(fit.state.lineage))))
    end
    while fit.state.temperature < 1
        fit = brm_smc_fit(fit; stages=1)
        st = last(fit.state.history)
        X, w = fit.state.positions, fit.state.weights
        push!(rows, (; st.stage, st.temperature, st.ess_before_resampling, st.unique_parents,
            st.surviving_ancestors, st.divergences, st.mean_acceptance, st.mean_squared_movement,
            st.evaluations, st.tuning_index, sd_m=wsd(X, w, idx.im), sd_l=wsd(X, w, idx.il),
            mean_l=wmean(X, w, idx.il), lineages=length(Set(fit.state.lineage))))
    end
    fit, rows
end

function final_summary(fit, idx, orc)
    X, w = fit.state.positions, fit.state.weights
    (; mean_m=wmean(X, w, idx.im), sd_m=wsd(X, w, idx.im),
        mean_l=wmean(X, w, idx.il), sd_l=wsd(X, w, idx.il),
        mean_z=[wmean(X, w, i) for i in idx.iz], sd_z=[wsd(X, w, i) for i in idx.iz],
        lineages=length(Set(fit.state.lineage)), stages=length(fit.state.history),
        evaluations=sum(s.evaluations for s in fit.state.history),
        divergences=sum(s.divergences for s in fit.state.history),
        log_normalizer=fit.state.log_normalizer)
end



const SPREAD_LO, SPREAD_HI, MEAN_TOL = 0.8, 1.25, 0.35

function tuned_schedule(rk, d; pilot, reference, pilot_seed, reference_inflation=nothing)
    kw = (; rng=Xoshiro(pilot_seed), init=zeros(d), pilot_steps=64,
        options=SMCOptions(; moves=2, max_tree_depth=7))
    if pilot == :adaptive
        kw = (; kw..., pilot=:adaptive, max_reflections=16, recording_target=128, pilot_evaluations=1000)
        reference === nothing || (kw = (; kw..., reference))
        reference_inflation === nothing || (kw = (; kw..., reference_inflation))
    else
        kw = (; kw..., pilot=:continuous)
    end
    brm_smc_tune(rk; kw...)
end

function check_spread(s, orc; group=true)
    @test SPREAD_LO <= s.sd_m / orc.sd_m <= SPREAD_HI
    @test SPREAD_LO <= s.sd_l / orc.sd_l <= SPREAD_HI
    @test abs(s.mean_m - orc.mean_m) <= MEAN_TOL * orc.sd_m
    @test abs(s.mean_l - orc.mean_l) <= MEAN_TOL * orc.sd_l
    if group
        for (a, b) in zip(s.sd_z, orc.sd_z)
            @test SPREAD_LO <= a / b <= SPREAD_HI
        end
    end
end

d = synthetic()
orc = oracle(d)
rk = RKBRMI(hierarchy(d))
idx = coordinate_indices(rk)
dim = length(idx.names)

@testset "two-group hierarchy: tuned SMC reproduces the oracle's population spread" begin
    for (label, pilot, reference) in (("continuous pilot", :continuous, nothing),
                                      ("adaptive pilot, standard-normal reference", :adaptive, nothing))
        schedule = tuned_schedule(rk, dim; pilot, reference, pilot_seed=11)
        options = SMCOptions(; tuning=schedule, history_limit=1000)
        fit, rows = run_bridge(rk, idx, options; seed=101, particles=256)
        s = final_summary(fit, idx, orc)
        @testset "$label" begin
            @test fit.state.temperature == 1
            @test first(rows).sd_m / orc.sd_m > 2      # the standard-normal reference is wider than the posterior
            check_spread(s, orc)
        end
    end
end

@testset "learned pilot moments: repaired reference with residual population under-dispersion" begin
    for (pilot_seed, seed) in ((11, 101), (12, 102))
        schedule = tuned_schedule(rk, dim; pilot=:adaptive, reference=:pilot, pilot_seed)
        @test schedule.pilot.reference_work.reference_inflation == 2.0
        options = SMCOptions(; tuning=schedule, history_limit=1000)
        fit, rows = run_bridge(rk, idx, options; seed, particles=256)
        s = final_summary(fit, idx, orc)
        @test fit.state.temperature == 1
        # Keep the original known-gap assertion. This is NOT the controls'
        # accuracy gate: the current seed-101 intercept SD is 0.786 of the
        # oracle, below SPREAD_LO. Reference repair does not fix finite-move
        # exploration; the strict failed accuracy run remains in the evidence.
        @test s.sd_m / orc.sd_m < 0.95
        @test SPREAD_LO <= s.sd_l / orc.sd_l <= SPREAD_HI
        @test abs(s.mean_m - orc.mean_m) <= MEAN_TOL * orc.sd_m
        @test abs(s.mean_l - orc.mean_l) <= MEAN_TOL * orc.sd_l
        for (a, b) in zip(s.sd_z, orc.sd_z)
            @test SPREAD_LO <= a / b <= SPREAD_HI
        end
        stage = last(fit.state.history)
        # The reference is wider than the oracle in both population coordinates.
        @test stage.reference_sd[idx.im] > orc.sd_m
        @test stage.reference_sd[idx.il] > orc.sd_l
        @test length(stage.coordinate_sd) == dim
        @test all(isfinite, stage.coordinate_sd)
        @test stage.coordinate_sd[idx.im] ≈ s.sd_m rtol=0.01
        @test stage.coordinate_sd[idx.il] ≈ s.sd_l rtol=0.01
        @printf("LEARNED_HIERARCHY seed=%d sd_ratio_m=%.4f sd_ratio_l=%.4f divergences=%d\n",
            seed, s.sd_m / orc.sd_m, s.sd_l / orc.sd_l, s.divergences)

        if pilot_seed == 11
            uninflated = tuned_schedule(rk, dim; pilot=:adaptive, reference=:pilot,
                pilot_seed, reference_inflation=1.0)
            @test uninflated.pilot.reference_work.reference_inflation == 1.0
            @test uninflated.pilot.reference_work.draw_sd == schedule.pilot.reference_work.draw_sd
            ref = smc_reference(schedule)
            @test ref.location == smc_reference(uninflated).location
            original = brm_smc_fit(rk; rng=Xoshiro(seed), particles=256, stages=0,
                options=SMCOptions(; tuning=uninflated))
            inflated = brm_smc_fit(rk; rng=Xoshiro(seed), particles=256, stages=0, options)
            @test inflated.state.positions .- ref.location ≈
                2 .* (original.state.positions .- ref.location)
        end
    end
end
