# test/shared_id_row_sets.jl — one shared `|ID|` block over predictors on
# different row sets.
#
# Run: julia --project=test test/shared_id_row_sets.jl
#
# `a ~ … + (1 + x | k | kreis)` beside `b ~ … + (1 | k | kreis_b)`, where `b`
# reads fewer rows than `a`, is ONE correlated block: its levels are the joint
# labels of `kreis` and `kreis_b`, and each predictor gathers its own rows by
# its own column (user decision 05d6dw7, snag a-correlated-kre-f027abd1). Before,
# SBBRMI silently built two uncorrelated blocks and RKBRMI refused the model.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using WarmupHMC
using LinearAlgebra, Random, Statistics

module SharedIdRowSets
using BayesianRegressionModels, Distributions, LinearAlgebra

# `a` on 8 rows over Kreise A-D; `b` observed on rows 1, 3, 4, 7, 8 of them.
const OBS = [1, 3, 4, 7, 8]
const KREIS = ["A", "A", "B", "B", "C", "C", "D", "D"]
const X = [-1.0, 0.5, -0.5, 1.0, 0.0, 1.5, -1.5, 0.25]
const YA = [0.3, 0.9, -0.2, 0.4, 0.1, 1.2, -0.8, 0.0]
const YB = [1.1, 0.0, 0.7, 0.2, 0.0, 0.0, 0.5, -0.3]
const JOINT = (; kreis=KREIS, x=X, ya=YA,
    kreis_b=KREIS[OBS], xb=X[OBS], yb=YB[OBS])
# The workaround the join replaces: `b` on every row of `a`, its unobserved
# rows weighted 0. Same parameters, same likelihood.
const ALIGNED = (; kreis=KREIS, x=X, ya=YA, xb=X, yb=YB,
    w=Float64.(in(OBS).(eachindex(KREIS))))

joint(d) = @brm d begin
    sa ~ Exponential(1)
    sb_ ~ Exponential(1)
    a ~ 1 + x + (1 + x | k | kreis)
    b ~ 1 + xb + (1 | k | kreis_b)
    sd(:, k) ~ Exponential(1)
    cor(:, k) ~ LKJCholesky(3, 2.0)
    ya ~ Normal(a, sa)
    yb ~ Normal(b, sb_)
end
aligned(d) = @brm d begin
    sa ~ Exponential(1)
    sb_ ~ Exponential(1)
    a ~ 1 + x + (1 + x | k | kreis)
    b ~ 1 + xb + (1 | k | kreis)
    sd(:, k) ~ Exponential(1)
    cor(:, k) ~ LKJCholesky(3, 2.0)
    ya ~ Normal(a, sa)
    yb ~ weighted(Normal(b, sb_), weights(w))
end

# `b`'s rows also name a Kreis `a` never reads (E), and `b` is observed through
# Kreis-year totals in a plate, as atlas-16 observes ETR/URS (RK's first slice
# admits one flat observation axis; plate cells are their own).
const CELLS = (; kreis=KREIS, x=X, ya=YA,
    kreis_b=["A", "B", "B", "D", "E"], xb=[-1.0, -0.5, 1.0, -1.5, 0.7],
    cell_b=["A1", "B1", "B1", "D1", "E1"], cells=["A1", "B1", "D1", "E1"],
    tb=[1.1, 0.9, 0.2, -0.3])
cells(d) = @brm d begin
    sa ~ Exponential(1)
    st ~ Exponential(1)
    a ~ 1 + x + (1 + x | k | kreis)
    b ~ 1 + xb + (1 | k | kreis_b)
    sd(:, k) ~ Exponential(1)
    cor(:, k) ~ LKJCholesky(3, 2.0)
    ya ~ Normal(a, sa)
    rows = groups(cell_b => cells)
    @plate for j in eachindex(cells)
        tb[j] ~ Normal(sum(b[rows[j]]), st)
    end
end

# atlas-16 as a minimal fit (snag forecast-replay-a44fea87): part 1 (`a`) on
# every unit-year, its response `ya` with a suppressed cell (`mi`), and a
# stratified unit block; part 2 (`b`) only on the unit-years of the observed
# district-years, sharing `|k|` with part 1 by label, reading part 1's
# completed response through a model-scope gather `c[pos_b]`, and observed as
# district-year totals. 4 units x 2 years in districts A, B.
const AT_KREIS = ["A", "A", "B", "B", "A", "A", "B", "B"]
const AT_KY = ["A-1", "A-1", "B-1", "B-1", "A-2", "A-2", "B-2", "B-2"]
const AT_PART1 = (; kreis=AT_KREIS, unit=[1, 2, 3, 4, 1, 2, 3, 4],
    size=[1, 1, 2, 2, 1, 1, 2, 2], x=[-1.0, 0.5, -0.5, 1.0, 0.0, 1.5, -1.5, 0.25],
    ya=Union{Missing,Float64}[1.2, 2.5, missing, 0.8, 1.4, 2.2, 3.1, 0.9], c0=0.3)
# The fit: part 2 on year 1, whose district-year totals are observed.
const AT_FIT = (; AT_PART1..., kreis_b=AT_KREIS[1:4], xb=[-0.9, 0.6, -0.4, 1.1],
    pos_b=[1, 2, 3, 4], ky_b=AT_KY[1:4], cells=["A-1", "B-1"], tb=[3.9, 2.6])
# The forecast: part 2 on year 2, district-years the fit never saw, no totals.
const AT_FUTURE = (; AT_PART1..., kreis_b=AT_KREIS[5:8], xb=[0.1, 1.6, -1.4, 0.35],
    pos_b=[5, 6, 7, 8], ky_b=AT_KY[5:8], cells=["A-2", "B-2"])
atlas(d) = @brm d begin
    sa ~ Exponential(1)
    st ~ Exponential(1)
    a ~ 1 + x + (1 | k | kreis) + (1 | gr(unit, by=size))
    mi(ya) ~ LogNormal(a, sa)
    c = log(ya) - c0
    c_b = c[pos_b]
    b ~ 1 + xb + c_b + (1 | k | kreis_b) + (1 | ko | ky_b)
    sd(:, k) ~ Exponential(1)
    sd(:, ko) ~ Exponential(1)
    rows = groups(ky_b => cells)
    @plate for j in eachindex(tb)
        tb[j] ~ Normal(sum(exp.(b[rows[j]])), st)
    end
end

# Snag reprocess-resamp-dab66282: `b`'s rows on Kreis C, which `a` observes and
# `b` does not (`kreis_b` names A, B, D), and on E, which neither observes; no
# `yb`, so `b` is predicted given the fit (a forecast).
const REDRAW = (; kreis=KREIS, x=X, ya=YA, kreis_b=["C", "E", "C"], xb=[0.4, -0.6, 1.2])
# The same block with Student-t levels.
student(d) = @brm d begin
    sa ~ Exponential(1)
    sb_ ~ Exponential(1)
    a ~ 1 + x + (1 + x | k | kreis)
    b ~ 1 + xb + (1 | k | kreis_b)
    sd(:, k) ~ Exponential(1)
    cor(:, k) ~ LKJCholesky(3, 2.0)
    ranef(:, k) ~ TDist(4)
    ya ~ Normal(a, sa)
    yb ~ Normal(b, sb_)
end
# The atlas's shape: `b` also walks per Kreis, `cdar(...; by = kreis_b)`, the
# column it reads its margin of the joined block through.
const WALK_FIT = merge(JOINT, (; year_b=[1, 1, 2, 1, 2], C=Matrix(1.0I, 3, 3)))
const WALK_REPLAY = (; kreis=KREIS, x=X, ya=YA, kreis_b=["C", "C"], year_b=[1, 2],
                     xb=[0.4, -0.6], C=Matrix(1.0I, 4, 4))
walk(d) = @brm d begin
    sa ~ Exponential(1)
    sb_ ~ Exponential(1)
    a ~ 1 + x + (1 + x | k | kreis)
    b ~ 1 + xb + (1 | k | kreis_b) + cdar(year_b; by = kreis_b, cor = C)
    sd(:, k) ~ Exponential(1)
    cor(:, k) ~ LKJCholesky(3, 2.0)
    sd(b, cdar(year_b)) ~ LogNormal(-1, 0.5)
    ya ~ Normal(a, sa)
    yb ~ Normal(b, sb_)
end
end

const SR = SharedIdRowSets

stan_block(code, name) = (k = findfirst("\n" * name * " {", code);
    code[k[1]+1:findnext("\n}", code, k[1] + 1)[end]])

function stan_problem(sb, name)
    path = joinpath(tempdir(), "brm-shared-id-row-sets", name * ".stan")
    mkpath(dirname(path))
    BRM.stan_instantiate(sb; path)
end

@stestset "SB: one block over the joint levels, equal to the weight-0 twin" begin
    sb = SBBRMI(SR.joint(SR.JOINT); mod=SR, total_groups=())
    code = stan_code(sb)
    # One block: one correlation factor over all three margins.
    @test count("lkj_corr_cholesky", code) == 1
    @test occursin("b_k_kreis[kreis_b_idx, 3]", code)
    @test !occursin("b_k_kreis_b", code)
    @test sb.data[:kreis_idx] == [1, 1, 2, 2, 3, 3, 4, 4]
    @test sb.data[:kreis_b_idx] == [1, 2, 2, 4, 4]
    @test sb.data[:n_kreis] == 4
    @test ranefcoefnames(SR.joint(SR.JOINT), :k) ==
          [(; predictor=:a, coefficient=:Intercept), (; predictor=:a, coefficient=:x),
           (; predictor=:b, coefficient=:Intercept)]
    block = only(ranef_blocks(sb))
    @test (block.group, block.id, block.n_terms, block.n_groups) == (:kreis, :k, 3, 4)
    @test block.levels == ["A", "B", "C", "D"]

    twin = SBBRMI(SR.aligned(SR.ALIGNED); mod=SR, total_groups=())
    joint_stan = stan_problem(sb, "joint")
    twin_stan = stan_problem(twin, "aligned")
    names = BridgeStan.param_unc_names(joint_stan.model)
    @test names == BridgeStan.param_unc_names(twin_stan.model)
    for u in (zeros(length(names)), collect(range(-0.4, 0.4; length=length(names))),
              [0.3cos(i) for i in eachindex(names)])
        @test BridgeStan.log_density(joint_stan.model, u; propto=false) ≈
              BridgeStan.log_density(twin_stan.model, u; propto=false) atol=1e-12
    end
end

@stestset "SB: a label only one column names is a level of the block" begin
    sb = SBBRMI(SR.cells(SR.CELLS); mod=SR, total_groups=())
    @test sb.data[:kreis_idx] == [1, 1, 2, 2, 3, 3, 4, 4]
    @test sb.data[:kreis_b_idx] == [1, 2, 2, 4, 5]
    @test sb.data[:n_kreis] == 5
    @test only(ranef_blocks(sb)).levels == ["A", "B", "C", "D", "E"]
    # One column alone keeps its own levels: the aligned twin's block reads
    # only `kreis`.
    twin = SBBRMI(SR.aligned(SR.ALIGNED); mod=SR, total_groups=())
    @test only(ranef_blocks(twin)).levels == ["A", "B", "C", "D"]
end

@stestset "RK and Stan agree on the joined block" begin
    brmi = SR.cells(SR.CELLS)
    rk, problem = consumer_problem(brmi)
    @test count(n -> startswith(string(n), "b_k_kreis.z."), coordinate_names(rk.model.layout)) ==
          3 * 5
    for centered_groups in ((), [:kreis_b])
        sb = SBBRMI(brmi; mod=SR, total_groups=(), centered_groups)
        stan = stan_problem(sb, isempty(centered_groups) ? "cells" : "cells_centered")
        names = BridgeStan.param_unc_names(stan.model)
        # `centered_groups` naming either column centres the whole block.
        @test any(startswith(isempty(centered_groups) ? "b_k_kreis_z_flat." :
                             "b_k_kreis_b_cols_bc."), names)
        transport = brm_coordinate_transport(rk, sb, names)
        @test length(transport) == length(names) == length(coordinate_names(rk.model.layout))
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))),
                  [0.37sin(i) for i in 1:length(transport)])
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

@stestset "Student-t and known-value joined blocks agree on RK and Stan" begin
    student_t = @brm SR.CELLS begin
        sa ~ Exponential(1)
        st ~ Exponential(1)
        a ~ 1 + x + (1 + x | k | kreis)
        b ~ 1 + xb + (1 | k | kreis_b)
        ranef(:, k) ~ TDist(4)
        ya ~ Normal(a, sa)
        rows = groups(cell_b => cells)
        @plate for j in eachindex(cells)
            tb[j] ~ Normal(sum(b[rows[j]]), st)
        end
    end
    known = @brm SR.CELLS begin
        sa ~ Exponential(1)
        st ~ Exponential(1)
        a ~ 1 + x + (1 + x | k | kreis)
        b ~ 1 + xb + (1 | k | kreis_b)
        sd(b, k) = 0.4
        cor(:, k) = [1.0 0.3 0.2; 0.3 1.0 -0.1; 0.2 -0.1 1.0]
        ya ~ Normal(a, sa)
        rows = groups(cell_b => cells)
        @plate for j in eachindex(cells)
            tb[j] ~ Normal(sum(b[rows[j]]), st)
        end
    end
    for (label, brmi) in (("student_t", student_t), ("known", known))
        rk, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=SR, total_groups=())
        @test only(ranef_blocks(sb)).n_groups == 5
        stan = stan_problem(sb, label)
        names = BridgeStan.param_unc_names(stan.model)
        transport = brm_coordinate_transport(rk, sb, names)
        @test length(transport) == length(names) == length(coordinate_names(rk.model.layout))
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))))
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

@stestset "a forecast replay codes new rows into the fitted levels" begin
    fit = SBBRMI(SR.cells(SR.CELLS); mod=SR, total_groups=())
    fit_names = BridgeStan.param_unc_names(stan_problem(fit, "forecast_fit").model)
    # The forecast year: `b` on rows in Kreise C (which `b` never read in the
    # fit) and E, its totals unobserved.
    future = (; kreis=["A", "C", "E"], x=[0.2, -0.1, 0.4], ya=[0.5, 0.0, 0.3],
        kreis_b=["C", "E", "E"], xb=[0.3, -0.2, 0.9],
        cell_b=["C2", "E2", "E2"], cells=["C2", "E2"])
    forecast = reprocess(fit, future)
    @test forecast.data[:kreis_idx] == [1, 3, 5]
    @test forecast.data[:kreis_b_idx] == [3, 5, 5]
    @test forecast.data[:n_kreis] == 5
    @test BridgeStan.param_unc_names(stan_problem(forecast, "forecast").model) == fit_names
    unseen = merge(future, (; kreis_b=["C", "F", "E"]))
    @test_throws "unseen level `F`" reprocess(fit, unseen)

    # RK replays an in-cell response only onto rows that carry it: the future
    # totals as observed.
    rk = RKBRMI(SR.cells(SR.CELLS))
    rk_replay = reprocess(rk, merge(future, (; tb=[0.4, 0.1])))
    @test coordinate_names(rk_replay.model.layout) == coordinate_names(rk.model.layout)
end

@stestset "a minimal fit forecasts district-years it never saw" begin
    # snag forecast-replay-a44fea87: part 2 fits only the observed
    # district-years; the forecast replays it onto new district-years, whose
    # `ko` effects are drawn from the fitted `sd(:, ko)`, while the joined `|k|`
    # block, the stratified unit block and part 1's `mi()` completions keep
    # their fitted coordinates.
    fit = SBBRMI(SR.atlas(SR.AT_FIT); mod=SR, total_groups=())
    fitted = stan_problem(fit, "atlas_fit")
    forecast = reprocess(fit, SR.AT_FUTURE; resample_groups=[:ky_b])
    code = stan_code(forecast)
    @test forecast.held_out == Set([:tb])
    @test !occursin(r"\btb\b", stan_block(code, "data"))
    @test !occursin("tb_likelihood", code)
    # the program of scoring placeholder totals, minus the placeholders
    placeholders = reprocess(fit, merge(SR.AT_FUTURE, (; tb=[0.0, 0.0]));
        resample_groups=[:ky_b], held_out=:tb)
    for block in ("parameters", "model")
        @test stan_block(code, block) == stan_block(stan_code(placeholders), block)
    end
    data = BRM.stan_data(forecast)
    @test data[:kreis_idx] == [1, 1, 2, 2, 1, 1, 2, 2]
    @test data[:kreis_b_idx] == [1, 1, 2, 2]
    @test data[:ky_b_idx] == [1, 1, 2, 2]
    @test data[:pos_b] == [5, 6, 7, 8]
    @test data[:unit__by__size_idx] == [1, 2, 3, 4, 1, 2, 3, 4]
    problem = stan_problem(forecast, "atlas_forecast")
    names = BridgeStan.param_unc_names(problem.model)
    # every fitted coordinate but the fitted district-years' standardized effects
    @test names == [n for n in BridgeStan.param_unc_names(fitted.model) if n in names]
    @test length(names) == length(BridgeStan.param_unc_names(fitted.model)) - 2
    @test any(startswith("ya_y_mis"), names)
    # so the fit's draws feed it through transport_draws, which selects the
    # forecast's coordinates by name; each keeps the fit draw's parameter values
    # (snag reprocess-resamp-cedeba79)
    unc_fit = BridgeStan.param_unc_names(fitted.model)
    draws = [0.37sin(i + 5k) for k in 1:3, i in eachindex(unc_fit)]
    moved = transport_draws(fit, forecast, draws, unc_fit, names)
    @test moved == draws[:, [findfirst(==(n), unc_fit) for n in names]]
    fit_params = BridgeStan.param_names(fitted.model)
    kept = [findfirst(==(n), fit_params) for n in BridgeStan.param_names(problem.model)]
    @test kept isa Vector{Int}
    for k in axes(draws, 1)
        @test BridgeStan.param_constrain(problem.model, moved[k, :]) ≈
              BridgeStan.param_constrain(fitted.model, draws[k, :])[kept]
    end
    q = collect(range(-0.3, 0.4; length=length(names)))
    all_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(problem.model, 1))
    @test all(isfinite, values)
    @test count(startswith("tb_gen."), all_names) == 2
    @test count(startswith("b."), all_names) == 4
    d = brm_descriptor(forecast)
    @test :predict in [o.name for o in d.operations]
    @test :fit in [o.name for o in d.operations]       # `ya` is still observed
    # a district the fit never saw is still refused for the joined block
    @test_throws "unseen level `C`" reprocess(fit, merge(SR.AT_FUTURE,
        (; kreis_b=["A", "C", "B", "B"])); resample_groups=[:ky_b])
end

# Constrained values (parameters, transformed parameters, generated quantities)
# of one unconstrained draw, and the values of one name among them.
function constrained(problem, q; seed=1)
    sm = problem.model
    names = String.(BridgeStan.param_names(sm; include_tp=true, include_gq=true))
    names, BridgeStan.param_constrain(sm, q; include_tp=true, include_gq=true,
                                      rng=BridgeStan.StanRNG(sm, seed))
end
pick(names, values, name) =
    [v for (n, v) in zip(names, values) if n == name || startswith(n, name * ".")]

# The redraw law written out (`joint_level_redraw`): in margin `order`, with `P`
# the Cholesky factor of the block's covariance, a redrawn label that is fitted
# joint level `pos` keeps its kept margins and draws the others given them, `P
# [P_KK \ b_K; e_R]`; a new label draws all, `P e`. A level weight scales both.
function redrawn_rows(b, tau, L, eta, pos, order, n_kept;
                      w_fit=ones(size(b, 1)), w_new=ones(length(pos)))
    T = length(tau)
    S = Diagonal(tau) * L * L' * Diagonal(tau)
    P = cholesky(Symmetric(S[order, order])).L
    E = reshape(eta, T, length(pos))
    out = zeros(length(pos), T)
    for j in eachindex(pos)
        z, w = E[:, j], w_new[j]
        if pos[j] > 0
            w = w_fit[pos[j]]
            K = order[1:n_kept]
            z[1:n_kept] = LowerTriangular(P[1:n_kept, 1:n_kept]) \ (b[pos[j], K] ./ sqrt(w))
        end
        out[j, order] = sqrt(w) .* (P * z)
    end
    out
end

@stestset "resample_groups redraws a joined column's margins given the kept ones" begin
    # snag reprocess-resamp-dab66282: `b` never observed Kreis C, `a` did, so C
    # is a fitted level of the joined block; E is new to both. Redrawing
    # `kreis_b` draws `b`'s margin of C given its fitted `a` margins, and all
    # of E's margins, while the fitted block stays a parameter `a` reads.
    fit = SBBRMI(SR.joint(SR.JOINT); mod=SR, total_groups=())
    @test_throws "unseen level `E`" reprocess(fit, SR.REDRAW)
    replay = reprocess(fit, SR.REDRAW; resample_groups=[:kreis_b])
    code = stan_code(replay)
    @test occursin("b_k_kreis_z_flat", stan_block(code, "parameters"))
    @test occursin("b_k_kreis_new = joint_level_redraw", stan_block(code, "generated quantities"))
    @test occursin("ya ~ normal(a, sa)", stan_block(code, "model"))
    data = BRM.stan_data(replay)
    @test data[:kreis_idx] == [1, 1, 2, 2, 3, 3, 4, 4]   # `a` reads the fitted levels A-D
    @test data[:n_kreis] == 4
    @test data[:kreis_b_idx] == [1, 2, 1]                # among the redrawn labels C, E
    @test data[:b_k_kreis_new_pos] == [3, 0]             # C is fitted level 3; E is new
    @test data[:b_k_kreis_new_order] == [1, 2, 3]        # `a`'s two margins kept first
    @test data[:b_k_kreis_new_n_kept] == 2

    fitted = stan_problem(fit, "redraw_fit")
    problem = stan_problem(replay, "redraw")
    unc = BridgeStan.param_unc_names(fitted.model)
    @test BridgeStan.param_unc_names(problem.model) == unc
    draws = [0.37sin(i + 5k) for k in 1:3, i in eachindex(unc)]
    @test transport_draws(fit, replay, draws, unc, unc) == draws

    q = 0.4 .* randn(Xoshiro(7), length(unc))
    kept_C, new_C, new_E = Float64[], Float64[], Vector{Float64}[]
    local S, b
    for seed in 1:2000
        cn, cv = constrained(problem, q; seed)
        tau = pick(cn, cv, "b_k_kreis_tau")
        L = reshape(pick(cn, cv, "b_k_kreis_L"), 3, 3)
        b = reshape(pick(cn, cv, "b_k_kreis"), 4, 3)
        new = reshape(pick(cn, cv, "b_k_kreis_new"), 2, 3)
        eta = pick(cn, cv, "b_k_kreis_new_eta")
        S = Diagonal(tau) * L * L' * Diagonal(tau)
        # exactly the law, given the generated draws
        @test new ≈ redrawn_rows(b, tau, L, eta, [3, 0], [1, 2, 3], 2) atol=1e-10
        @test new[1, 1:2] ≈ b[3, 1:2] atol=1e-10     # C keeps its fitted `a` margins
        @test pick(cn, cv, "r_b_k_kreis") ≈ new[[1, 2, 1], 3] atol=1e-10
        push!(new_C, new[1, 3])
        push!(new_E, new[2, :])
    end
    # C's `b` margin is the conditional normal given C's fitted `a` margins
    K, R = [1, 2], [3]
    m = only(S[R, K] / S[K, K] * b[3, K])
    v = only(S[R, R] - S[R, K] / S[K, K] * S[K, R])
    @test abs(mean(new_C) - m) < 5sqrt(v / 2000)
    @test var(new_C) ≈ v rtol=0.15
    # E's margins are a fresh level of the block
    H = reduce(hcat, new_E)
    @test maximum(abs.(mean(H; dims=2)) ./ sqrt.(diag(S))) < 5 / sqrt(2000)
    @test cov(H; dims=2) ≈ S rtol=0.2
    d = brm_descriptor(replay)
    out = brm_execute(d, :predict; problem, draws=permutedims(draws), seed=1234)
    @test haskey(out, :yb_gen) || haskey(out, :yb)
end

@stestset "a redrawn first column, every column, Student-t levels, and what stays refused" begin
    fit = SBBRMI(SR.joint(SR.JOINT); mod=SR, total_groups=())
    # `kreis` is the block's first column: the fitted block is addressed
    # through `kreis_b`, and `a`'s two margins follow `b`'s in the redraw order.
    rows = (; kreis=["E", "C"], x=[0.1, 0.2], kreis_b=["A", "D"], xb=[0.3, 0.4])
    first_col = reprocess(fit, rows; resample_groups=[:kreis])
    data = BRM.stan_data(first_col)
    @test data[:kreis_idx] == [2, 1]
    @test data[:kreis_b_idx] == [1, 4]
    @test data[:b_k_kreis_new_pos] == [3, 0]
    @test data[:b_k_kreis_new_order] == [3, 1, 2]
    @test data[:b_k_kreis_new_n_kept] == 1
    problem = stan_problem(first_col, "redraw_first")
    unc = BridgeStan.param_unc_names(problem.model)
    @test unc == BridgeStan.param_unc_names(stan_problem(fit, "redraw_fit").model)
    q = 0.4 .* randn(Xoshiro(8), length(unc))
    for seed in 1:20
        cn, cv = constrained(problem, q; seed)
        tau = pick(cn, cv, "b_k_kreis_tau")
        L = reshape(pick(cn, cv, "b_k_kreis_L"), 3, 3)
        b = reshape(pick(cn, cv, "b_k_kreis"), 4, 3)
        new = reshape(pick(cn, cv, "b_k_kreis_new"), 2, 3)
        @test new ≈ redrawn_rows(b, tau, L, pick(cn, cv, "b_k_kreis_new_eta"),
                                 [3, 0], [3, 1, 2], 1) atol=1e-10
        @test new[1, 3] ≈ b[3, 3] atol=1e-10
    end

    # Every column: the whole block is drawn anew over the labels they name.
    all_cols = reprocess(fit, SR.REDRAW; resample_groups=[:kreis, :kreis_b])
    code = stan_code(all_cols)
    @test occursin("b_k_kreis_n_g = max(max(kreis_idx), max(kreis_b_idx))", code)
    @test occursin("b_k_kreis_z_flat", stan_block(code, "generated quantities"))
    @test !occursin("b_k_kreis_new", code)
    @test all_cols.data[:n_kreis] == 5                     # A-E
    @test BRM.stan_data(all_cols)[:kreis_b_idx] == [3, 5, 3]
    names = BridgeStan.param_unc_names(stan_problem(all_cols, "redraw_all").model)
    @test names == [n for n in unc if !startswith(n, "b_k_kreis_z_flat")]
    # the block's labels: the fitted ones while a kept column reads it, the
    # replay's when every column is redrawn
    partial = only(ranef_blocks(first_col))
    @test (partial.levels, partial.generated) == (["A", "B", "C", "D"], false)
    whole = only(ranef_blocks(all_cols))
    @test (whole.levels, whole.generated) == (["A", "B", "C", "D", "E"], true)

    # Student-t levels: a fitted level keeps its fitted weight, a new one draws one.
    tfit = SBBRMI(SR.student(SR.JOINT); mod=SR, total_groups=())
    treplay = reprocess(tfit, SR.REDRAW; resample_groups=[:kreis_b])
    tproblem = stan_problem(treplay, "redraw_student")
    tunc = BridgeStan.param_unc_names(tproblem.model)
    @test tunc == BridgeStan.param_unc_names(stan_problem(tfit, "redraw_student_fit").model)
    q = 0.4 .* randn(Xoshiro(9), length(tunc))
    for seed in 1:20
        cn, cv = constrained(tproblem, q; seed)
        tau = pick(cn, cv, "b_k_kreis_tau")
        L = reshape(pick(cn, cv, "b_k_kreis_L"), 3, 3)
        b = reshape(pick(cn, cv, "b_k_kreis"), 4, 3)
        new = reshape(pick(cn, cv, "b_k_kreis_new"), 2, 3)
        @test new ≈ redrawn_rows(b, tau, L, pick(cn, cv, "b_k_kreis_new_eta"), [3, 0],
            [1, 2, 3], 2; w_fit=pick(cn, cv, "b_k_kreis_w"),
            w_new=pick(cn, cv, "b_k_kreis_new_w")) atol=1e-10
        @test new[1, 1:2] ≈ b[3, 1:2] atol=1e-10
    end

    # refused: the redraw is given the fitted block, which a fresh template
    # rebuilds on the replay rows (its coordinates are not the fit's)
    @test_throws "use the default `freeze_constants=true`" reprocess(fit,
        merge(SR.REDRAW, (; yb=[0.1, 0.2, 0.3])); resample_groups=[:kreis_b],
        freeze_constants=false)
    # not built on RKBRMI yet
    @test_throws "drawing new levels of a joined block is not implemented" reprocess(
        RKBRMI(SR.cells(SR.CELLS)), SR.CELLS; resample_groups=[:kreis_b])
end

@stestset "the atlas's ETR hold-out: a joined block and a cdar walk on one column" begin
    # snag reprocess-resamp-dab66282 as the atlas has it: `b` walks per Kreis by
    # the column it reads its margin of `|k|` through; Kreis C has no `b` rows
    # in the fit. Its replay redraws C's walk from the fitted `sigma`, `rho` and
    # C's `b` margin given its fitted `a` margins.
    fit = SBBRMI(SR.walk(SR.WALK_FIT); mod=SR, total_groups=())
    replay = reprocess(fit, SR.WALK_REPLAY; resample_groups=[:kreis_b])
    data = BRM.stan_data(replay)
    @test data[:b_k_kreis_new_pos] == [3]
    @test StanBlocks.getvalue(replay.data[:cdar_b_year_b_n_new_groups]) == 1
    fitted = stan_problem(fit, "walk_fit")
    problem = stan_problem(replay, "walk")
    unc = BridgeStan.param_unc_names(fitted.model)
    @test BridgeStan.param_unc_names(problem.model) == unc
    q = 0.4 .* randn(Xoshiro(10), length(unc))
    for seed in 1:20
        cn, cv = constrained(problem, q; seed)
        b = reshape(pick(cn, cv, "b_k_kreis"), 4, 3)
        new = reshape(pick(cn, cv, "b_k_kreis_new"), 1, 3)
        @test new[1, 1:2] ≈ b[3, 1:2] atol=1e-10
        @test pick(cn, cv, "b") ≈ pick(cn, cv, "pop_b") .+ new[1, 3] .+
                                  pick(cn, cv, "cdar_b_year_b") atol=1e-10
    end
    d = brm_descriptor(replay)
    out = brm_execute(d, :predict; problem, seed=1234,
                      draws=permutedims([0.3sin(i + 2k) for k in 1:3, i in eachindex(unc)]))
    @test haskey(out, :yb_gen) || haskey(out, :yb)
end

@stestset "descriptor, draws and adaptive centering read the joined block" begin
    sb = SBBRMI(SR.cells(SR.CELLS); mod=SR, total_groups=())
    stan = stan_problem(sb, "descriptor")
    unc = BridgeStan.param_unc_names(stan.model)
    d = brm_descriptor(sb)
    q = collect(range(-0.3, 0.4; length=length(unc)))
    draws = permutedims(BridgeStan.param_constrain(stan.model, q; include_tp=true))
    names = BridgeStan.param_names(stan.model; include_tp=true)
    entry = only(brm_ranef_draws(d, draws, names; id=:k))
    @test entry.levels == ["A", "B", "C", "D", "E"]
    @test entry.margins == [(; predictor=:a, coefficient=:Intercept),
                            (; predictor=:a, coefficient=:x),
                            (; predictor=:b, coefficient=:Intercept)]
    @test size(entry.effects) == (1, 5, 3)
    ac = only(adaptive_centering_blocks(sb, unc))
    @test (ac.ranef.n_terms, ac.ranef.n_groups) == (3, 5)
    wrapped = adaptive_centering_problem(sb, stan, AutoEnzyme())
    lp, g = LogDensityProblems.logdensity_and_gradient(stan, q)
    lpw, gw = LogDensityProblems.logdensity_and_gradient(wrapped, q)
    @test lpw ≈ lp atol=1e-10
    @test gw ≈ g atol=1e-10
end

@stestset "an omitted response keeps the whole joined block on SB" begin
    # `b`'s totals omitted: SB samples the full block (its prior on `b`'s
    # margin is exact), with no marginal split. RK's narrowed planning of a
    # joined block is refused until it matches (snag a-correlated-kre-f027abd1).
    omitted = (; (k => v for (k, v) in pairs(SR.CELLS) if k !== :tb)...)
    full = SBBRMI(SR.cells(SR.CELLS); mod=SR, total_groups=())
    sb = SBBRMI(SR.cells(omitted); mod=SR, total_groups=())
    @test !occursin("_active", stan_code(sb))
    @test count(n -> startswith(n, "b_k_kreis_z_flat."),
                BridgeStan.param_unc_names(stan_problem(sb, "omitted").model)) ==
          count(n -> startswith(n, "b_k_kreis_z_flat."),
                BridgeStan.param_unc_names(stan_problem(full, "omitted_full").model))
    @test_throws "narrowed to its read predictors is not implemented on RK" RKBRMI(
        SR.cells(omitted))
end

@stestset "refusals" begin
    # Columns sharing no label cannot name one factor.
    disjoint = merge(SR.JOINT, (; kreis_b=["P", "Q", "Q", "R", "R"]))
    @test_throws "shares no label" SBBRMI(SR.joint(disjoint); mod=SR, total_groups=())
    @test_throws "shares no label" RKBRMI(SR.cells(merge(SR.CELLS,
        (; kreis_b=["P", "Q", "Q", "R", "S"]))))
    # Not implemented yet for a joined block (todo `089dnl4`): level correlation.
    levelcor = @brm SR.JOINT begin
        Omega ~ LKJCholesky(4, 2.0)
        a ~ 1 + (1 | k | gr(kreis; cor = Omega))
        b ~ 1 + (1 | k | gr(kreis_b; cor = Omega))
        ya ~ Normal(a, 1)
        yb ~ Normal(b, 1)
    end
    @test_throws "level correlation over joined columns is not implemented" SBBRMI(
        levelcor; mod=SR, total_groups=())
    # refused: one predictor reads its margins of a block through one grouping
    # column, so a row gathers one level (decision 05d6dw7 joins predictors).
    two_columns = @brm SR.ALIGNED begin
        a ~ 1 + x + (1 | k | kreis) + (0 + x | k | w)
        ya ~ Normal(a, 1)
    end
    @test_throws "groups its `|k|` terms by both" SBBRMI(two_columns; mod=SR, total_groups=())
    # refused: `kreis_b` (no Kreis C) is coded into the 5 joint levels for
    # `|k|`, so a block over `kreis_b` alone would read another coding under
    # the same index name `kreis_b_idx`.
    alone = @brm SR.CELLS begin
        a ~ 1 + x + (1 + x | k | kreis)
        b ~ 1 + xb + (1 | k | kreis_b) + (0 + xb | kreis_b)
        ya ~ Normal(a, 1)
        rows = groups(cell_b => cells)
        @plate for j in eachindex(cells)
            tb[j] ~ Normal(sum(b[rows[j]]), 1)
        end
    end
    @test_throws "Another random-effect block groups by `kreis_b` alone" SBBRMI(
        alone; mod=SR, total_groups=())
    # refused: one `|ID|` names one block, and a stratified block is a different
    # grouping factor (mirrors RK, test/rk_emitter.jl).
    stratified = @brm merge(SR.ALIGNED, (; arm=[1, 1, 1, 1, 2, 2, 2, 2])) begin
        a ~ 1 + x + (1 | k | kreis) + (0 + x | k | gr(kreis, by=arm))
        ya ~ Normal(a, 1)
    end
    @test_throws "conflicting grouping factors" SBBRMI(stratified; mod=SR, total_groups=())
end
