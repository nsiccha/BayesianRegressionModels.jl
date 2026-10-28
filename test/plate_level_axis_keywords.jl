# test/plate_level_axis_keywords.jl — a predictor read through a `@plate`
# grouping whose term carries data on another axis than its rows: the levels'
# coordinates in `hsgp(t; by=g, cor=multigroup((x, y)))`, a known level
# correlation `cor=C` (on `hsgp`, on `gr` and on `cdar`), and an areal term's
# `car(area; edges=(from, to))`. Those fields have one entry per level or per
# edge, so the plate's check that every entry of the predictor is labelled by
# the grouping column must not count them as rows (`_brm_row_keyword`,
# src/preparation_kernel.jl; snags `rkbrmi-on-a-mode-0bd31c04`,
# `cdar-cor-matrix-c8278be6`).
#
# Each plate model must equal its plate-free twin, on RKBRMI and on SBBRMI
# (BridgeStan), and the reported spelling (the term as a predictor of its own,
# beside a random-effect predictor) must equal the term written inside that
# predictor.
#
#     julia --project=test test/plate_level_axis_keywords.jl
#
# Set `BRM_PLATE_LEVEL_AXIS_RUNTIME=0` to skip the compiled-model sets.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LinearAlgebra
using Random
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan

const BRM = BayesianRegressionModels
const BS = BridgeStan
const RUNTIME = get(ENV, "BRM_PLATE_LEVEL_AXIS_RUNTIME", "1") != "0"
const CACHE = joinpath(tempdir(), "brm-plate-level-axis-keywords")
const G = 6            # levels
const T = 5            # rows per level

function level_data(; seed=1)
    rng = Random.MersenneTwister(seed)
    A = randn(rng, G, G)
    S = A * A' + G * I
    D = Diagonal(1 ./ sqrt.(diag(S)))
    # Rows interleaved across levels, so a cell's rows are not contiguous.
    (; g=repeat(1:G, T), t=repeat(collect(1.0:T), inner=G), y=randn(rng, G * T),
       sx=100 .* rand(rng, G), sy=100 .* rand(rng, G), C=Matrix(Symmetric(D * S * D)),
       from=collect(1:G-1), to=collect(2:G), xg=randn(rng, G),
       x=randn(rng, G * T), steps=collect(1.0:T), tot=G .+ rand(rng, T))
end
# The same `C`, computed from a row column by `constant(...)`: the form that
# built before `cdar`'s `cor` was skipped, since its one input lies on the rows.
level_cor(g) = (length(unique(g)) == G || error("expected $G levels"); level_data().C)

# Each term once as a predictor read through the plate's grouping and once in a
# plate-free twin; both keep the predictor name `st`, so their coordinates
# carry the same names.
const CASES = [
    "multigroup coordinates" => (
        quote
            st ~ 0 + hsgp(t; k=4, by=g, cor=multigroup((sx, sy); k=4))
            @plate for p in groups(g; order=t)
                y[p] ~ Normal(st[p], 0.7)
            end
        end,
        quote
            st ~ 0 + hsgp(t; k=4, by=g, cor=multigroup((sx, sy); k=4))
            y ~ Normal(st, 0.7)
        end, (;)),
    "hsgp with a known cor = C" => (
        quote
            st ~ 0 + hsgp(t; k=4, by=g, cor=C)
            @plate for p in groups(g; order=t)
                y[p] ~ Normal(st[p], 0.7)
            end
        end,
        quote
            st ~ 0 + hsgp(t; k=4, by=g, cor=C)
            y ~ Normal(st, 0.7)
        end, (;)),
    "gr(g; cor = C)" => (
        quote
            st ~ 0 + (1 | gr(g; cor=C))
            @plate for p in groups(g; order=t)
                y[p] ~ Normal(st[p], 0.7)
            end
        end,
        quote
            st ~ 0 + (1 | gr(g; cor=C))
            y ~ Normal(st, 0.7)
        end, (;)),
    "car edges" => (
        quote
            st ~ 0 + car(g; edges=(from, to), type=:icar)
            @plate for p in groups(g; order=t)
                y[p] ~ Normal(st[p], 0.7)
            end
        end,
        quote
            st ~ 0 + car(g; edges=(from, to), type=:icar)
            y ~ Normal(st, 0.7)
        end, (;)),
    # The reported spelling against the one that already built: the term as a
    # predictor of its own beside a random-effect predictor, or inside it. On
    # SBBRMI with `total_groups=:auto` the first keeps `mu`'s random intercept
    # as an exact total block (`total_mu`) and the second does not, so the two
    # programs share their coordinates only with `total_groups=()` (the
    # default, stated here so the comparison does not depend on it); the
    # exact-total program is checked separately below.
    "reported: a predictor of its own beside mu" => (
        quote
            sigma ~ Exponential(1.0)
            mu ~ 1 + (1 | lv | g)
            st ~ 0 + hsgp(t; k=4, by=g, cor=multigroup((sx, sy); k=4))
            @plate for p in groups(g; order=t)
                m = mu[p] .+ st[p]
                y[p] ~ Normal(m, sigma)
            end
        end,
        quote
            sigma ~ Exponential(1.0)
            mu ~ 1 + (1 | lv | g) + hsgp(t; k=4, by=g, cor=multigroup((sx, sy); k=4))
            @plate for p in groups(g; order=t)
                y[p] ~ Normal(mu[p], sigma)
            end
        end, (; total_groups=())),
]

# `cdar(step; by=g, cor=C)`, whose `C` has one row and column per level of `g`
# (on RKBRMI since BRM 0.2.33, snag `rkbrmi-cdar-lowe-d39fc042`). Kept apart
# from `CASES`, whose last entry the exact-total set below reads.
const CDAR_CASES = [
    "cdar with a known cor = C" => (
        quote
            st ~ 0 + cdar(t; by=g, cor=C)
            @plate for p in groups(g; order=t)
                y[p] ~ Normal(st[p], 0.7)
            end
        end,
        quote
            st ~ 0 + cdar(t; by=g, cor=C)
            y ~ Normal(st, 0.7)
        end, (;)),
    # The reported spelling: `cdar` beside other terms, its predictor summed per
    # step through a `groups(t => steps)` binding (each cell crosses the
    # levels), against the same `C` computed by `constant(...)`.
    "reported: cdar summed through a groups(t => steps) binding" => (
        quote
            sigma ~ Exponential(1.0)
            st ~ 1 + x + cdar(t; by=g, cor=C)
            rows = groups(t => steps)
            @plate for k in eachindex(steps)
                tot[k] ~ Normal(sum(exp.(st[rows[k]])), sigma)
            end
        end,
        quote
            sigma ~ Exponential(1.0)
            Cc = constant(level_cor(g))
            st ~ 1 + x + cdar(t; by=g, cor=Cc)
            rows = groups(t => steps)
            @plate for k in eachindex(steps)
                tot[k] ~ Normal(sum(exp.(st[rows[k]])), sigma)
            end
        end, (;)),
]

build(body, df) = BRM.brm(body; mod=@__MODULE__)(df)
rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])
# `perm[i]` is the twin coordinate of the plate model's coordinate `i`.
function name_permutation(names, twin_names)
    perm = [only(findall(==(n), twin_names)) for n in names]
    @test sort(perm) == collect(eachindex(twin_names))
    perm
end

@testset "the row axis skips level and edge keywords" begin
    @test !BRM._brm_row_keyword(BRM.hsgp, Val(:cor))
    @test !BRM._brm_row_keyword(BRM.gr, Val(:cor))
    @test !BRM._brm_row_keyword(BRM.car, Val(:edges))
    @test !BRM._brm_row_keyword(BRM.cdar, Val(:cor))
    @test BRM._brm_row_keyword(BRM.cdar, Val(:by))
    @test BRM._brm_row_keyword(BRM.hsgp, Val(:by))
    @test BRM._brm_row_keyword(BRM.hsgp, Val(:rows))
    @test BRM._brm_row_keyword(BRM.car, Val(:type))
end

@testset "RKBRMI: each plate model equals its plate-free twin" begin
    df = level_data()
    for (label, (plate, twin, _)) in vcat(CASES, CDAR_CASES)
        @testset "$label" begin
            rk, rk_twin = RKBRMI(build(plate, df)), RKBRMI(build(twin, df))
            names = String.(coordinate_names(rk.model.layout))
            perm = name_permutation(names, String.(coordinate_names(rk_twin.model.layout)))
            problem, twin_problem = rk_problem(rk), rk_problem(rk_twin)
            for u in points(length(names))
                v = similar(u); v[perm] .= u
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                twin_value, twin_gradient = LogDensityProblems.logdensity_and_gradient(twin_problem, v)
                @test isfinite(value)
                @test value ≈ twin_value atol=1e-10 rtol=1e-12
                @test gradient ≈ twin_gradient[perm] atol=1e-10 rtol=1e-10
            end
        end
    end
    # A replay rebuilds the plate's row partition in the graph from the bound
    # columns; the level fields are not among its row axes there either.
    m = build(first(last(first(CASES))), df)
    rk = RKBRMI(m)
    replayed = reprocess(rk, df)
    @test replayed isa RKBRMI
    u = last(points(length(coordinate_names(rk.model.layout))))
    @test LogDensityProblems.logdensity(rk_problem(replayed), u) ≈
        LogDensityProblems.logdensity(rk_problem(rk), u) atol=1e-10 rtol=1e-12
end

refusal(f) = try
    f(); ""
catch err
    sprint(showerror, err)
end

function compiled(sb, tag)
    mkpath(CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(CACHE, tag * "_" * string(hash(code)) * ".stan"))
end

@testset "SBBRMI: each plate model equals its plate-free twin" begin
    df = level_data()
    if RUNTIME
        for (i, (label, (plate, twin, sb_keywords))) in enumerate(vcat(CASES, CDAR_CASES))
            @testset "$label" begin
                stan = compiled(SBBRMI(build(plate, df); mod=@__MODULE__, sb_keywords...), "plate$i")
                stan_twin = compiled(SBBRMI(build(twin, df); mod=@__MODULE__, sb_keywords...), "twin$i")
                names = BS.param_unc_names(stan.model)
                perm = name_permutation(names, BS.param_unc_names(stan_twin.model))
                for u in points(length(names))
                    v = similar(u); v[perm] .= u
                    g, g_twin = similar(u), similar(v)
                    value, _ = BS.log_density_gradient!(stan.model, u, g; propto=false, jacobian=true)
                    twin_value, _ = BS.log_density_gradient!(stan_twin.model, v, g_twin;
                                                            propto=false, jacobian=true)
                    @test isfinite(value)
                    @test value ≈ twin_value atol=1e-10 rtol=1e-12
                    @test g ≈ g_twin[perm] atol=1e-10 rtol=1e-10
                end
            end
        end
        @testset "reported spelling, exact-total program" begin
            sb = SBBRMI(build(first(last(last(CASES))), df); mod=@__MODULE__,
                        total_groups=:auto)
            @test !isempty(total_effect_blocks(brm_descriptor(sb).plan))
            stan = compiled(sb, "reported_totals")
            n = length(BS.param_unc_names(stan.model))
            for u in points(n)
                value, _ = BS.log_density_gradient!(stan.model, u, similar(u);
                                                    propto=false, jacobian=true)
                @test isfinite(value)
            end
        end
    else
        for (label, (plate, _, _)) in vcat(CASES, CDAR_CASES)
            @test occursin("model", BRM.stan_code(SBBRMI(build(plate, df); mod=@__MODULE__)))
        end
        @info "Skipping the compiled SB sets (BRM_PLATE_LEVEL_AXIS_RUNTIME=0)"
    end
end

@testset "a column on another row axis is still refused" begin
    # `xg` has one value per level, so as a covariate of the predictor's rows it
    # leaves most rows unlabelled: still the plate's refusal, on both backends.
    df = level_data()
    m = build(quote
        st ~ 0 + xg
        @plate for p in groups(g; order=t)
            y[p] ~ Normal(st[p], 0.7)
        end
    end, df)
    for text in (refusal(() -> RKBRMI(m)), refusal(() -> BRM.stan_code(SBBRMI(m; mod=@__MODULE__))))
        @test occursin("`st` read through `g`: `st` has $G entries (data vector `xg`)", text)
    end
end
