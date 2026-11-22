# test/adaptive_centering_projected_location.jl — a random-effect cell centred
# on everything its level's rows pin (snag adaptive-centrin-647119ba, decision
# `BayesianRegressionModels:snag.adaptive-centrin-647119ba/decisions/2026-11-22T11-56-21-218-10krhre`).
#
# A trip-like grouping sits in a factor level, carries a covariate constant
# within it and one that varies partly, has an intercept and a slope, and lies in
# one step of an `rw(...)` walk. The oracles do not read the wrapper's
# projection code:
#
# - at `c = 0` the wrapper is the compiled density and gradient exactly;
# - at `c = 1` the source coordinates of each level's cells are the
#   least-squares fit, on the level's own design `(1, pos)`, of the linear
#   predictor BridgeStan computes from the mapped point (the level's total as
#   the data see it), and a walk cell's source is the walk's step `sigma * z`;
# - under mixed controls the inverse map returns the source point and the
#   gradient equals central finite differences, on the per-pair, joint (with an
#   HSGP) and learned-order wrappers;
# - walk cells come first, as `:rw` cells, the reported locations name what
#   each cell reads, and `centeredness = :midpoint` leaves the walk cells at
#   their compiled start.
#
# Both backends' public cell rows for this model, their agreement, and a model
# with walk cells only are in test/adaptive_centering_cells.jl ("projected
# covariates and rw walk cells").
#
# Run: julia --project=test test/adaptive_centering_projected_location.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LinearAlgebra
using LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)
problem(model, p; kwargs...) = adaptive_centering_problem(model, p, AutoEnzyme(); kwargs...)

fd_gradient(f, q; h = 1e-5) = [begin
    p, m = copy(q), copy(q)
    p[i] += h; m[i] -= h
    (f(p) - f(m)) / (2h)
end for i in eachindex(q)]

# Trips nested in years (one year each), in one of two day types, with a
# covariate constant within the trip (`near`) and one that varies around it
# (`lprev`), `NR` stops each.
function trip_data(; seed = 11, NT = 24, NR = 5, years = 3)
    rng = Xoshiro(seed)
    trip = repeat(["t$i" for i in 1:NT], inner = NR)
    year = repeat([2021 + mod(i, years) for i in 1:NT], inner = NR)
    dtype = repeat([isodd(i) ? "a" : "b" for i in 1:NT], inner = NR)
    near = repeat(Float64.(rand(rng, 0:1, NT)), inner = NR)
    pos = repeat(collect(1.0:NR), NT)
    lprev = near .+ 0.1 .* randn(rng, NT * NR)
    x2 = randn(rng, NT * NR)
    y = randn(rng, NT * NR)
    (; trip, year, dtype, near, pos, lprev, x2, y)
end

const D = trip_data()

const M_A = SBBRMI(@brm(begin
    sigma ~ Exponential(1)
    mu ~ 0 + factor(dtype) + near + lprev + (1 + pos || trip) + rw(year)
    y ~ Normal(mu, sigma)
end)(D); mod = @__MODULE__)
const P_A = instantiate(M_A)
const U_A = BS.param_unc_names(P_A.model)

# Correlated levels, an intercept, a second-order walk and an HSGP (the joint
# wrapper).
const M_B = SBBRMI(@brm(begin
    sigma ~ Exponential(1)
    mu ~ 1 + near + lprev + (1 + pos | trip) + rw(year; order = 2) + hsgp(x2; k = 4)
    y ~ Normal(mu, sigma)
end)(merge(D, (; year = repeat([2021 + mod(i, 4) for i in 1:24], inner = 5)))); mod = @__MODULE__)
const P_B = instantiate(M_B)
const U_B = BS.param_unc_names(P_B.model)

# A sum-to-zero walk: its level is read, its path gets no cells.
const M_C = SBBRMI(@brm(begin
    sigma ~ Exponential(1)
    mu ~ 1 + near + (1 | trip) + rw(year; sum_to_zero = true)
    y ~ Normal(mu, sigma)
end)(D); mod = @__MODULE__)
const P_C = instantiate(M_C)
const U_C = BS.param_unc_names(P_C.model)

n_cells(model, p; kwargs...) =
    length(WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(problem(model, p; kwargs...))))

# The linear predictor `mu` BridgeStan computes from a model-frame point, less
# the row values of a term the locations do not read (`exclude`, an HSGP).
function linear_predictor(p, model_point; exclude = nothing)
    names = BS.param_names(p.model; include_tp = true)
    v = BS.param_constrain(p.model, model_point; include_tp = true)
    mu = v[findall(n -> occursin(r"^mu\.\d+$", n), names)]
    isnothing(exclude) && return mu
    mu .- v[findall(n -> occursin(Regex("^" * exclude * "\\.\\d+\$"), n), names)]
end

ranef_rows(rp, model) = [r for r in adaptive_centering_controls(rp, model) if r.family === :ranef]

# At `c = 1`, the sources of level `lev`'s cells, in term order `terms`, equal
# the least-squares coefficients of `mu` on the level's rows `[1 pos]`.
function check_level_totals(model, p, U, terms; seed = 3, exclude = nothing, kwargs...)
    n = n_cells(model, p; kwargs...)
    rp = problem(model, p; centeredness = ones(n), kwargs...)
    rows = adaptive_centering_controls(rp, model)
    q = 0.3 .* randn(Xoshiro(seed), LogDensityProblems.dimension(rp))
    _, point = WarmupHMC.reparametrizer(rp)(q)
    mu = linear_predictor(p, point; exclude)
    @test length(mu) == length(D.trip)
    worst = 0.0
    for lev in unique(D.trip)
        I = findall(==(lev), D.trip)
        Z = hcat((t === :Intercept ? ones(length(I)) : D.pos[I] for t in terms)...)
        beta = Z \ mu[I]
        for (k, t) in enumerate(terms)
            r = only(r for r in rows if r.family === :ranef && string(r.level) == lev &&
                     r.margin.coefficient === t)
            worst = max(worst, abs(q[r.index] - beta[k]))
        end
    end
    worst, rows, q, point
end

@testset "walk innovations are :rw cells, mapped before the trips that read them" begin
    rp = problem(M_A, P_A)
    rows = adaptive_centering_controls(rp, M_A)
    walk = [r for r in rows if r.family === :rw]
    @test [r.name for r in walk] == ["rw_mu_year_z.1", "rw_mu_year_z.2"]
    @test all(r -> r.location === nothing && r.binding === :rw_mu_year, walk)
    @test maximum(r.cell for r in walk) < minimum(r.cell for r in rows if r.family === :ranef)
    # A trip in the third year (`t11`: year 2023) reads both innovations, its day
    # type's level, `near` (0 on it) not at all, and `lprev`'s within-trip
    # intercept; its slope only `lprev`'s within-trip slope.
    i = findfirst(==("t11"), D.trip)
    I = findall(==("t11"), D.trip)
    b = hcat(ones(length(I)), D.pos[I]) \ D.lprev[I]
    ri = only(r for r in ranef_rows(rp, M_A) if r.level == "t11" && r.margin.coefficient === :Intercept)
    rs = only(r for r in ranef_rows(rp, M_A) if r.level == "t11" && r.margin.coefficient === :pos)
    dt = D.dtype[i] == "a" ? "cat_mu_dtype_beta.1" : "cat_mu_dtype_beta.2"
    @test D.near[i] == 0 && D.year[i] == 2023
    @test ri.location == dt * " + " * string(round(b[1]; sigdigits = 6)) *
                         "*pop_mu_beta_pop.2 + rw_mu_year_z.1 + rw_mu_year_z.2"
    @test rs.location == string(round(b[2]; sigdigits = 6)) * "*pop_mu_beta_pop.2"
    # a trip with `near = 1` in the first year reads `near` whole and no walk cell
    j = findfirst(t -> D.near[findfirst(==(t), D.trip)] == 1 &&
                       D.year[findfirst(==(t), D.trip)] == 2021, unique(D.trip))
    lev = unique(D.trip)[j]
    r1 = only(r for r in ranef_rows(rp, M_A) if r.level == lev && r.margin.coefficient === :Intercept)
    @test occursin(" + pop_mu_beta_pop.1 + ", r1.location)
    @test !occursin("rw_mu_year", r1.location)
end

@testset "c = 0 is the compiled frame, exactly" begin
    for (model, p) in ((M_A, P_A), (M_B, P_B), (M_C, P_C))
        rp = problem(model, p)
        q = 0.3 .* randn(Xoshiro(7), LogDensityProblems.dimension(rp))
        l0, g0 = LogDensityProblems.logdensity_and_gradient(p, q)
        l1, g1 = LogDensityProblems.logdensity_and_gradient(rp, q)
        @test l1 == l0
        @test g1 == g0
    end
end

@testset "c = 1: a level's sources are its linear predictor's fit on its design" begin
    worst, rows, q, point = check_level_totals(M_A, P_A, U_A, (:Intercept, :pos))
    @test worst < 1e-10
    sigma = exp(q[findfirst(==("rw_mu_year_sigma"), U_A)])
    for r in rows
        r.family === :rw && @test q[r.index] ≈ sigma * point[r.index] rtol = 1e-14
    end
    # The HSGP varies within a trip and is no term the locations read.
    @test first(check_level_totals(M_B, P_B, U_B, (:Intercept, :pos);
                                   exclude = "hsgp_x2")) < 1e-10
    @test first(check_level_totals(M_B, P_B, U_B, (:Intercept, :pos);
                                   exclude = "hsgp_x2", ranef_order = :learned)) < 1e-10
    @test first(check_level_totals(M_C, P_C, U_C, (:Intercept,))) < 1e-10
end

@testset "a sum-to-zero walk is read through its SD, and has no cells" begin
    rows = adaptive_centering_controls(problem(M_C, P_C), M_C)
    @test !any(r -> r.family === :rw, rows)
    lev = D.trip[findfirst(==(2023), D.year)]
    r = only(r for r in rows if r.family === :ranef && r.level == lev)
    @test occursin("exp(rw_mu_year_sigma)*", r.location)
    # Its plan reads no other cell, so it takes the hand-written K <= 2 reverse
    # pass, whose pullback through the walk's SD equals WarmupHMC's generic pass.
    enzyme_ext = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCEnzymeExt)
    wrapped = problem(M_C, P_C; centeredness = rand(Xoshiro(5), n_cells(M_C, P_C)))
    state = first(WarmupHMC.reparametrizer(wrapped).pairs).second.args[1].state
    @test any(!=(0), state.pair_factor_scales) && !enzyme_ext._small_plan_reads_cells(state)
    @test which(LogDensityProblems.logdensity_and_gradient,
                (typeof(wrapped), Vector{Float64})).module === enzyme_ext
    q = 0.3 .* randn(Xoshiro(6), LogDensityProblems.dimension(wrapped))
    lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, q)
    lp_generic, gradient_generic = invoke(LogDensityProblems.logdensity_and_gradient,
        Tuple{WarmupHMC.ReparametrizedProblem,AbstractVector}, wrapped, q)
    @test lp == lp_generic
    @test gradient ≈ gradient_generic rtol = 1e-11 atol = 1e-11
    # the trip model's cells read walk cells: that plan takes the generic pass
    state_a = first(WarmupHMC.reparametrizer(problem(M_A, P_A)).pairs).second.args[1].state
    @test enzyme_ext._small_plan_reads_cells(state_a)
end

@testset "centeredness = :midpoint moves the trip cells, not the walk cells" begin
    rows = adaptive_centering_controls(problem(M_A, P_A; centeredness = :midpoint), M_A)
    @test any(r -> r.family === :rw, rows)
    @test all(r -> r.c == (r.family === :rw ? 0.0 : 0.5), rows)
end

@testset "mixed controls: inverse, log-Jacobian and gradient" begin
    for (model, p, kwargs) in ((M_A, P_A, (;)), (M_B, P_B, (;)),
                               (M_B, P_B, (; ranef_order = :learned)), (M_C, P_C, (;)))
        n = n_cells(model, p; kwargs...)
        rng = Xoshiro(23)
        for _ in 1:2
            controls = rand(rng, [0.0, 0.3, 0.7, 1.0], n)
            rp = problem(model, p; centeredness = controls, kwargs...)
            q = 0.3 .* randn(rng, LogDensityProblems.dimension(rp))
            lp, grad = LogDensityProblems.logdensity_and_gradient(rp, q)
            @test lp ≈ LogDensityProblems.logdensity(rp, q)
            fd = fd_gradient(v -> LogDensityProblems.logdensity(rp, v), q)
            @test maximum(abs.(grad .- fd) ./ (1 .+ abs.(fd))) < 1e-5
            ir = WarmupHMC.reparametrizer(rp)
            ljac, mapped = ir(q)
            @test lp ≈ LogDensityProblems.logdensity(p, mapped) + ljac
            _, back = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
            @test back ≈ q atol = 1e-11
        end
    end
end

@testset "the public cells describe the walk cells and the weighted locations" begin
    plan = adaptive_centering_cells(M_A, U_A)
    walk = [c for c in plan.cells if c.family === :rw]
    @test length(walk) == 2 && all(c -> c.cell <= 2, walk)
    @test any(c -> !isempty(c.location.walks), plan.cells)
    @test all(c -> all(w -> w.cell < c.cell && plan.cells[w.cell].family === :rw,
                       c.location.walks), plan.cells)
    @test all(c -> all(t -> t.coordinate != 0 && t.log_scale == 0, c.location.terms),
              plan.cells)      # a zero-start walk is read through its cells, unscaled
    # a sum-to-zero walk's path values are terms scaled by the walk's log SD
    sigma_c = findfirst(==("rw_mu_year_sigma"), U_C)
    @test any(c -> any(t -> t.log_scale == sigma_c, c.location.terms),
              adaptive_centering_cells(M_C, U_C).cells)
    @test all(b -> b.family === :rw, plan.blocks[1:1])
end
