# test/adaptive_hsgp_level_centering.jl — adaptive centring of grouped HSGPs
# whose levels are correlated (`hsgp(x...; by=g, cor=Omega)`, or a known
# correlation matrix) or scaled per level (a hyper-predictor on the term's `sd`
# or `length_scale`). Basis column `b`'s level weights `diag(s_b) * L * z_b`
# are mapped like one group of a correlated random-effect block
# (`BRM._HSGPLevelAdaptiveCenteringBlock`; snag adaptive-centeri-11062238).
#
#     julia --project=test test/adaptive_hsgp_level_centering.jl
#
# Generic synthetic models only.

using Test
using BayesianRegressionModels
using StanBlocks
using LinearAlgebra
using Random
using LogDensityProblems
using WarmupHMC
using Enzyme
using DifferentiationInterface: AutoEnzyme
import BridgeStan

const BRM = BayesianRegressionModels
const BS = BridgeStan
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)
const CACHE = joinpath(tempdir(), "brm-adaptive-hsgp-levels")
const NG = 3

function level_df(; per_level=16, seed=5)
    rng = MersenneTwister(seed)
    t = repeat(collect(range(0.0, 10.0; length=per_level)), NG)
    u = repeat(collect(range(-2.0, 2.0; length=per_level)) .^ 2, NG)
    g = repeat(["c", "a", "b"]; inner=per_level)
    subject = repeat(1:4, NG * per_level ÷ 4)
    y = sin.(t ./ 2) .+ 0.2 .* u .+ 0.3 .* randn(rng, NG * per_level)
    C = [1.0 0.5 0.2; 0.5 1.0 0.3; 0.2 0.3 1.0]
    (; t, u, g, subject, y, C)
end

# The reported shape: a two-axis field per level, the levels correlated, each
# with its own pooled amplitude, projected off each level's intercept.
const icm_sd = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(t, u; k=4, by=g, cor=Omega, orthogonal_to=:intercept)
    log(sd(hsgp(t, u))) ~ 1 + (1 | g)
    y ~ Normal(mu, 0.5)
end
const icm = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(t; k=5, by=g, cor=Omega)
    y ~ Normal(mu, 0.5)
end
const icm_both = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(t; k=5, by=g, cor=Omega)
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    log(length_scale(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 0.5)
end
const icm_aniso = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(t, u; k=(3, 4), iso=false, by=g, cor=Omega)
    y ~ Normal(mu, 0.5)
end
const known_sd = @brm begin
    mu ~ 1 + hsgp(t; k=5, by=g, cor=C)
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 0.5)
end
# Independent levels with their own amplitudes: no `cor=`, still per level.
const hyper_only = @brm begin
    mu ~ 1 + hsgp(t; k=5, by=g)
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 0.5)
end
# Beside a correlated random-effect block, an ungrouped HSGP and a known
# length scale.
const joint = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + u + (1 + u | subject) + hsgp(t; k=5, by=g, cor=Omega) + hsgp(u; k=4)
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    length_scale(mu, hsgp(u)) = 1.5
    y ~ Normal(mu, 0.5)
end
const multigroup_tau = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(Omega, 0.7))
    y ~ Normal(mu, 0.5)
end

const MODELS = (; icm_sd, icm, icm_both, icm_aniso, known_sd, hyper_only, joint)

function compiled(sb, tag)
    mkpath(CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(CACHE, tag * "_" * string(hash(code)) * ".stan"))
end

const BUILT = Dict{Symbol,Any}()
function built(tag::Symbol)
    get!(BUILT, tag) do
        sb = SBBRMI(getfield(MODELS, tag)(level_df()); mod=@__MODULE__)
        problem = compiled(sb, string(tag))
        names = BS.param_unc_names(problem.model)
        (; sb, problem, names)
    end
end

points(n, k; seed=7) = (rng = MersenneTwister(seed); [0.6 .* randn(rng, n) for _ in 1:k])

# The compiled model's level weights `w[g, b]` (`beta_scaled` of the term's
# submodel) at the unconstrained point `x`.
function compiled_level_weights(problem, block, x)
    names = BS.param_names(problem.model; include_tp=true)
    values = BS.param_constrain(problem.model, x; include_tp=true)
    G, B = size(block.effects)
    prefix = string(block.term, "_beta_scaled.")
    idx = findall(n -> startswith(n, prefix), names)
    length(idx) == G * B || error("expected $(G * B) `$prefix` entries, found $(length(idx))")
    reshape(values[idx], G, B)
end

# Independent reference of the level map: per basis column, the factor
# `C_b = diag(s_b) * L` materialized by the metadata reader and the triangular
# map written out with it (cf. `manual_adaptive_map` in
# test/adaptive_centering_warmuphmc.jl).
function manual_level_map(x, block, controls)
    y = copy(x)
    ljac = zero(eltype(x))
    G, B = size(block.effects)
    p = 1
    for b in 1:B
        C = BRM._adaptive_hsgp_level_factor(x, block, b)
        z = zeros(eltype(x), G)
        for k in 1:G
            c = controls[p]
            m = sum(C[k, l] * z[l] for l in 1:k-1; init=zero(eltype(x)))
            s = C[k, k]
            z[k] = (x[block.effects[k, b]] - c * m) / s^c
            y[block.effects[k, b]] = z[k]
            ljac -= c * log(s)
            p += 1
        end
    end
    ljac, y
end

# Every pair's control set to `controls` (in pair order) on a fresh wrapper.
function with_controls!(adaptive, controls)
    ir = adaptive.reparametrizer
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(value.target, WarmupHMC.PartiallyCentered(c),
                                           value.args...)
    end
    AC_EXT._sync_sources!(AC_EXT._centering_state(ir), ir)
    adaptive
end

function finite_jacobian(f, y; h=1e-6)
    n = length(y)
    J = zeros(n, n)
    for j in 1:n
        e = zeros(n)
        e[j] = h
        J[:, j] = (f(y .+ e) .- f(y .- e)) ./ (2h)
    end
    J
end

@testset "metadata: one level block per correlated or per-level-scaled term" begin
    for tag in keys(MODELS)
        (; sb, names) = built(tag)
        blocks = BRM._adaptive_hsgp_level_centering_blocks(sb, names)
        @test length(blocks) == 1
        block = only(blocks)
        G, B = size(block.effects)
        @test G == NG
        @test block.group === :g
        # the grouping's level order (sorted labels), which the map follows
        @test block.levels == ["a", "b", "c"]
        brmi = brm_descriptor(sb).plan.parent
        flat = string(BRM._adaptive_grouped_hsgp_weights(brmi, :mu,
            only(e for e in BRM._brm_term_coordinate_entries(brmi, :mu)
                 if e.term === block.term))[2])
        @test [names[block.effects[g, b]] for g in 1:G, b in 1:B] ==
              ["$flat.$((g - 1) * B + b)" for g in 1:G, b in 1:B]
        # Every other HSGP resolves through the per-basis reader, never this term.
        others = BRM._adaptive_hsgp_centering_blocks(sb, names)
        @test all(o -> o.term !== block.term, others)
        @test isdisjoint(vec(block.effects),
                         reduce(vcat, (o.effects for o in others); init=Int[]))
    end
    let block = only(BRM._adaptive_hsgp_level_centering_blocks(built(:icm).sb, built(:icm).names))
        @test block.correlation === :parameter
        @test built(:icm).names[block.cholesky_free] == ["Omega.1", "Omega.2", "Omega.3"]
        @test !block.sigma.predicted && !block.rho.predicted
    end
    let block = only(BRM._adaptive_hsgp_level_centering_blocks(built(:known_sd).sb,
                                                               built(:known_sd).names))
        @test block.correlation === :data
        @test isempty(block.cholesky_free)
        @test BRM._adaptive_hsgp_level_correlation_factor(zeros(3), block) ≈
              cholesky(level_df().C).L
        @test block.sigma.predicted && length(block.sigma.z) == NG
    end
    let block = only(BRM._adaptive_hsgp_level_centering_blocks(built(:hyper_only).sb,
                                                               built(:hyper_only).names))
        @test block.correlation === :none
        @test block.sigma.predicted
    end
    let block = only(BRM._adaptive_hsgp_level_centering_blocks(built(:icm_both).sb,
                                                               built(:icm_both).names))
        @test block.sigma.predicted && block.rho.predicted
        @test block.rho_floor > 0
    end
    @test size(only(BRM._adaptive_hsgp_level_centering_blocks(built(:icm_aniso).sb,
        built(:icm_aniso).names)).omega2) == (12, 2)

    # A multigroup coherence correlates the levels per frequency: refused.
    sb = SBBRMI(multigroup_tau(level_df()); mod=@__MODULE__)
    names = vcat(["Omega.$i" for i in 1:3], ["zflat_hsgpw_t_g.$i" for i in 1:15])
    @test_throws "cor = multigroup(...)" BRM._adaptive_hsgp_level_centering_blocks(sb, names)
    @test_throws "cor = multigroup(...)" BRM._adaptive_hsgp_centering_blocks(sb, names)
end

@testset "the reference factor reproduces the compiled level weights" begin
    for tag in keys(MODELS)
        (; sb, problem, names) = built(tag)
        block = only(BRM._adaptive_hsgp_level_centering_blocks(sb, names))
        for x in points(length(names), 3)
            W = compiled_level_weights(problem, block, x)
            for b in axes(block.effects, 2)
                C = BRM._adaptive_hsgp_level_factor(x, block, b)
                @test C * x[block.effects[:, b]] ≈ W[:, b] rtol=1e-10 atol=1e-12
            end
        end
    end
end

@testset "the wrapper's map is the triangular level map, exact and invertible" begin
    for tag in keys(MODELS)
        (; sb, problem, names) = built(tag)
        block = only(BRM._adaptive_hsgp_level_centering_blocks(sb, names))
        adaptive = adaptive_centering_problem(sb, problem, AutoEnzyme())
        ir = adaptive.reparametrizer
        state = AC_EXT._centering_state(ir)
        level = state.level
        @test state.n_level == length(block.effects)
        offset = state.n_ranef + state.n_r2d2 + state.n_hsgp
        @test [first(ir.pairs[offset + p]) for p in 1:state.n_level] == level.pair_index
        n = length(ir.pairs)
        x = first(points(length(names), 1; seed=11))
        # At the compiled controls the wrapper is the plain problem.
        @test LogDensityProblems.logdensity(adaptive, x) ≈
              LogDensityProblems.logdensity(problem, x)
        rng = MersenneTwister(19)
        for controls in (rand(rng, n), ones(n))
            with_controls!(adaptive, controls)
            ljac, mapped = ir(x)
            level_controls = controls[offset .+ (1:state.n_level)]
            ref_ljac, ref = manual_level_map(x, block, level_controls)
            @test mapped[vec(block.effects)] ≈ ref[vec(block.effects)] rtol=1e-12
            # The level cells' share of the log-Jacobian is the reference's.
            others = copy(controls)
            others[offset .+ (1:state.n_level)] .= 0
            with_controls!(adaptive, others)
            other_ljac, _ = ir(x)
            @test ljac - other_ljac ≈ ref_ljac rtol=1e-12 atol=1e-12
            with_controls!(adaptive, controls)
            # log|det J| of the whole map, by central differences.
            J = finite_jacobian(v -> last(ir(v)), x)
            @test logabsdet(J)[1] ≈ ljac atol=1e-6
            # The inverse rebuilds the source point.
            src = last(WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped))
            @test src ≈ x rtol=1e-10 atol=1e-12
            # Density identity and the Enzyme gradient.
            lp, grad = LogDensityProblems.logdensity_and_gradient(adaptive, x)
            @test lp ≈ LogDensityProblems.logdensity(problem, mapped) + ljac rtol=1e-12
            fd = map(eachindex(x)) do j
                e = zeros(length(x))
                e[j] = 1e-6
                (LogDensityProblems.logdensity(adaptive, x .+ e) -
                 LogDensityProblems.logdensity(adaptive, x .- e)) / 2e-6
            end
            @test grad ≈ fd rtol=1e-5 atol=1e-5
        end
        # At c = 1 every level cell's source coordinate is its compiled level
        # weight `w[g, b]`: the levels are fully centred.
        with_controls!(adaptive, ones(n))
        compiled_point = last(ir(x))
        W = compiled_level_weights(problem, block, compiled_point)
        @test x[block.effects] ≈ W rtol=1e-10 atol=1e-12
    end
end

@testset "learned order keeps the level cells' triangular map" begin
    (; sb, problem, names) = built(:joint)
    adaptive = adaptive_centering_problem(sb, problem, AutoEnzyme(); ranef_order=:learned)
    ir = adaptive.reparametrizer
    @test ir isa AC_EXT.BRMOrderedCenteringReparametrization
    n = length(WarmupHMC.reparam_controls(ir))
    WarmupHMC.restore_reparam_controls!(ir, rand(MersenneTwister(4), n))
    x = first(points(length(names), 1; seed=13))
    ljac, mapped = ir(x)
    J = finite_jacobian(v -> last(ir(v)), x)
    @test logabsdet(J)[1] ≈ ljac atol=1e-6
    src = last(WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped))
    @test src ≈ x rtol=1e-10 atol=1e-12
    # The level cells map as in the per-pair wrapper.
    perpair = adaptive_centering_problem(sb, problem, AutoEnzyme())
    with_controls!(perpair, WarmupHMC.reparam_controls(ir))
    block = only(BRM._adaptive_hsgp_level_centering_blocks(sb, names))
    @test mapped[vec(block.effects)] ≈ last(perpair.reparametrizer(x))[vec(block.effects)] rtol=1e-12
end

@testset "controls report one :hsgp row per level cell" begin
    (; sb, problem, names) = built(:joint)
    adaptive = adaptive_centering_problem(sb, problem, AutoEnzyme())
    rows = adaptive_centering_controls(adaptive, sb)
    state = AC_EXT._centering_state(adaptive.reparametrizer)
    block = only(state.level.blocks)
    G, B = size(block.effects)
    level_rows = [r for r in rows if r.family === :hsgp && r.group === :g]
    @test length(level_rows) == G * B
    @test [(r.term, r.rank) for r in level_rows] == [(b, g) for b in 1:B for g in 1:G]
    @test [r.level for r in level_rows] == repeat(block.levels, B)
    @test all(r -> r.margin == (; predictor=:mu, coefficient=block.term), level_rows)
    @test [r.index for r in level_rows] == [block.effects[g, b] for b in 1:B for g in 1:G]
end

@testset "the reported shape warms up through adaptive_warmup_mcmc" begin
    (; sb, problem, names) = built(:icm_sd)
    adaptive = adaptive_centering_problem(sb, problem, AutoEnzyme())
    result = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(0x20261028), adaptive;
        n_draws=40, n_evaluations=400, stepsize_adaptation_limit=40,
        max_tree_depth=7, progress=nothing, monitor_ess=false)
    @test size(result.posterior_position) == (length(names), 40)
    @test all(isfinite, result.posterior_position)
    state = AC_EXT._centering_state(adaptive.reparametrizer)
    offset = state.n_ranef + state.n_r2d2 + state.n_hsgp
    adapted = [value.c for (_, value) in WarmupHMC.reparam_sources(adaptive)]
    @test all(c -> c in 0.0:0.1:1.0, adapted)
    # Some level cell left the compiled frame.
    @test any(!=(0.0), adapted[offset .+ (1:state.n_level)])
end

println("adaptive_hsgp_level_centering.jl: all testsets passed")
