# test/total_effects_mean_location.jl — mean-located exact totals under
# WarmupHMC adaptive centering (user pick `1rncfjy`, snag
# exact-totals-fun-f116d6b2).
#
# Each total column's cells are centred on the column's least-squares fit on
# its location design (the mean of the totals for a plain block), a function of
# the totals, instead of a constant. Covers the column table, the exact map
# (round trip, log-Jacobian against finite differences and its closed form,
# aliasing, wrapped density and gradient), the frames at c = 1 and c = 0, the
# window rule's exact per-cell candidates, the controls round trip and report,
# and an online fit.
#
# Run: julia --project=test test/total_effects_mean_location.jl
using Test, Random, LinearAlgebra, Statistics
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
const EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)

const J = 8
const GRP = repeat(1:J, inner=4)
const WG = [0., 1., 0., 1., 1., 0., 1., 0.]
const ZG = [0.4, -1.2, 0.9, 0.1, -0.6, 1.5, -0.3, 0.7]
const SG = [0.5, -0.8, 1.1, 0.3, -1.4, 0.9, 0.2, -0.4]

const DATA = let rng = Xoshiro(41)
    x = randn(rng, length(GRP))
    y = 0.5 .+ 0.8 .* WG[GRP] .+ 0.3 .* ZG[GRP] .+ 0.5 .* x .+ randn(rng, J)[GRP] .+
        0.3 .* randn(rng, length(GRP))
    (; y, w=WG[GRP], z=ZG[GRP], x, xw=x .* SG[GRP], g=GRP)
end

# A plain intercept block (location design: ones) beside a fixed slope.
const PLAIN = @brm begin
    mu ~ 1 + x + (1 | a | g)
    effect(mu, :) ~ Normal(0, 2)
    sd(:, a) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

# Group-level predictors: the intercept column's design is [1 w z], the slope
# column's [1 SG] (`xw` is the slope column times a per-level value).
const GROUP_LEVEL = @brm begin
    mu ~ 1 + w + z + x + xw + (1 | a | g) + (0 + x | b | g)
    effect(mu, :) ~ Normal(0, 2)
    sd(:, a) ~ Normal(0, 1)
    sd(:, b) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

function setup(builder, name)
    sb = SBBRMI(builder(DATA); mod=@__MODULE__, total_groups=:auto)
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(mktempdir(), "$name.stan"))
    names = BridgeStan.param_unc_names(problem.model)
    block = only(total_effect_blocks(sb))
    coords = BRM._total_coordinates(sb, block, names)
    physical = 0.5 .* randn(Xoshiro(7), length(names))
    physical[coords.scales] .= log.(0.4 .+ 0.3 .* eachindex(coords.scales))
    (; sb, problem, names, block, coords, physical)
end

function fd_jacobian(f, x; h=1e-6)
    out = zeros(length(x), length(x))
    for i in eachindex(x)
        plus, minus = copy(x), copy(x)
        plus[i] += h
        minus[i] -= h
        out[:, i] = (last(f(plus)) - last(f(minus))) / (2h)
    end
    out
end

fd_gradient(rp, x; h=1e-6) = map(eachindex(x)) do i
    e = zeros(length(x)); e[i] = h
    (LogDensityProblems.logdensity(rp, x + e) - LogDensityProblems.logdensity(rp, x - e)) / (2h)
end

# Column `k`'s design, derived from the block independently of the table.
function design(block, k)
    W = zeros(size(block.W, 1), 0)
    for p in axes(block.A, 2)
        iszero(block.A[k, p]) && continue
        trial = hcat(W, block.W[:, p])
        rank(trial) == size(trial, 2) && (W = trial)
    end
    W
end

projection(W, T) = size(W, 2) == 0 ? zero(T) : W * ((W' * W) \ (W' * T))

const CASES = [(:plain, setup(PLAIN, "plain"), [1]), (:group_level, setup(GROUP_LEVEL, "group_level"), [3, 2])]

@testset "column table: $name" for (name, m, widths) in CASES
    cols = BRM._total_centering_columns(m.sb, m.names)
    @test cols.widths == widths
    @test cols.indices == vec(m.coords.totals)
    @test cols.scales == m.coords.scales
    @test cols.starts == [1 + J * (k - 1) for k in 1:length(widths) + 1]
    for k in eachindex(widths)
        stored = reshape(cols.design[cols.offsets[k] .+ (1:J * widths[k])], J, widths[k])
        @test stored == design(m.block, k)
    end
    name === :plain && @test only(cols.widths) == 1 && all(==(1.0), cols.design)
end

@testset "exact map: $name" for (name, m, _) in CASES
    n = length(m.coords.totals)
    for c in (0.0, 0.37, 1.0, collect(range(0.05, 0.95; length=n)))
        rp = adaptive_centering_problem(m.sb, m.problem, AutoEnzyme(); centeredness=c)
        ir = WarmupHMC.reparametrizer(rp)
        @test ir isa EXT.BRMMeanTotalsCenteringReparametrization
        _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, m.physical)
        jac, mapped = ir(source)
        @test mapped ≈ m.physical atol = 1e-10
        @test first(logabsdet(fd_jacobian(ir, source))) ≈ jac atol = 1e-6
        aliased = copy(source)
        ajac, _ = WarmupHMC.with_logabsdet_jacobian!(aliased, ir, aliased)
        @test ajac ≈ jac && aliased ≈ mapped
        lp, g = LogDensityProblems.logdensity_and_gradient(rp, source)
        @test lp ≈ LogDensityProblems.logdensity(m.problem, m.physical) + jac atol = 1e-8
        @test g ≈ fd_gradient(rp, source) atol = 1e-5 rtol = 1e-5
        # Per column: the location is the least-squares fit of the totals in
        # every frame, each cell moves by its own a_j, and the log-Jacobian is
        # sum(log a) + logdet(W'W) - logdet(W'AW).
        controls = WarmupHMC.reparam_controls(ir)
        closed = 0.0
        for k in axes(m.coords.totals, 2)
            W = design(m.block, k)
            idx = m.coords.totals[:, k]
            T = m.physical[idx]
            ell = m.physical[m.coords.scales[k]]
            a = exp.((1 .- controls[(k - 1) * J .+ (1:J)]) .* ell)
            loc = projection(W, T)
            @test source[idx] ≈ loc .+ (T .- loc) ./ a atol = 1e-10
            closed += sum(log, a) + logdet(W' * W) - logdet(W' * Diagonal(a) * W)
        end
        @test jac ≈ closed atol = 1e-10
        if c == 1.0
            @test source ≈ m.physical atol = 1e-12
            @test abs(jac) < 1e-12
        end
        if c == 0.0 && name === :plain
            # c = 0: the standard non-centred deviations around the totals' mean.
            T = m.physical[vec(m.coords.totals)]
            tau = exp(m.physical[only(m.coords.scales)])
            @test source[vec(m.coords.totals)] ≈ mean(T) .+ (T .- mean(T)) ./ tau atol = 1e-10
        end
    end
end

@testset "window candidates are exact per cell: $name" for (name, m, _) in CASES
    n = length(m.coords.totals)
    c = collect(range(0.9, 0.1; length=n))
    rp = adaptive_centering_problem(m.sb, m.problem, AutoEnzyme(); centeredness=c)
    ir = WarmupHMC.reparametrizer(rp)
    _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, m.physical)
    _, g = LogDensityProblems.logdensity_and_gradient(rp, source)
    Q = zeros(length(EXT._CENTERING_CANDIDATES), n)
    G = zeros(size(Q))
    EXT._mean_totals_candidates!(Q, G, ir, source, g)
    # The candidate frame of cell `p` at `t`, built directly: every other cell
    # keeps its control, and the same physical point is mapped into it.
    moved_rp = adaptive_centering_problem(m.sb, m.problem, AutoEnzyme(); centeredness=c)
    moved_ir = WarmupHMC.reparametrizer(moved_rp)
    for p in (1, 2, J, n), i in (1, 4, 11)
        moved = copy(c)
        moved[p] = EXT._CENTERING_CANDIDATES[i]
        WarmupHMC.restore_reparam_controls!(moved_ir, moved)
        _, s2 = WarmupHMC._inverse_with_logabsdet_jacobian(moved_ir, m.physical)
        _, g2 = LogDensityProblems.logdensity_and_gradient(moved_rp, s2)
        index = ir.state.indices[p]
        @test Q[i, p] ≈ s2[index] atol = 1e-10
        @test G[i, p] ≈ g2[index] atol = 1e-8 rtol = 1e-8
    end
    # The window rule picks candidates and reports whether it moved any.
    positions = hcat((source .+ 0.05 .* randn(Xoshiro(s), length(source)) for s in 1:6)...)
    gradients = hcat((last(LogDensityProblems.logdensity_and_gradient(rp, x)) for x in eachcol(positions))...)
    moved_any = EXT._mean_totals_select!(ir, positions, gradients)
    @test moved_any isa Bool
    @test all(c -> c in EXT._CENTERING_CANDIDATES, WarmupHMC.reparam_controls(ir))
end

@testset "the pilot selector scores the wrapper's frame: $name" for (name, m, _) in CASES
    n = length(m.coords.totals)
    draws = permutedims(hcat((m.physical .+ 0.1 .* randn(Xoshiro(s), length(m.physical)) for s in 1:9)...))
    gradients = permutedims(hcat((last(LogDensityProblems.logdensity_and_gradient(m.problem, collect(r)))
                                  for r in eachrow(draws))...))
    selected = select_total_centeredness(m.sb, draws, m.names; criterion=:gradient, gradients)
    @test selected.indices == vec(m.coords.totals)
    # From the compiled frame (every control 1), the window rule's candidates
    # are the selector's: the same correlation, cell by cell.
    rp = adaptive_centering_problem(m.sb, m.problem, AutoEnzyme(); centeredness=1.0)
    ir = WarmupHMC.reparametrizer(rp)
    Qs = [zeros(length(EXT._CENTERING_CANDIDATES), n) for _ in axes(draws, 1)]
    Gs = [zeros(size(first(Qs))) for _ in axes(draws, 1)]
    for r in axes(draws, 1)
        EXT._mean_totals_candidates!(Qs[r], Gs[r], ir, draws[r, :], gradients[r, :])
    end
    for p in 1:n, i in eachindex(EXT._CENTERING_CANDIDATES)
        @test selected.losses[p, i] ≈ cor([q[i, p] for q in Qs], [g[i, p] for g in Gs]) atol = 1e-10
    end
    @test all(c -> 0 <= c <= 1, select_total_centeredness(m.sb, draws, m.names).centeredness)
end

@testset "controls round trip and report" begin
    m = CASES[2][2]
    n = length(m.coords.totals)
    rp = adaptive_centering_problem(m.sb, m.problem, AutoEnzyme(); centeredness=0.3)
    ir = WarmupHMC.reparametrizer(rp)
    @test WarmupHMC.reparam_controls(ir) == fill(0.3, n)
    snapshot = WarmupHMC.snapshot_reparametrization(ir)
    WarmupHMC.restore_reparam_controls!(ir, fill(0.6, n))
    @test WarmupHMC.reparam_controls(snapshot) == fill(0.3, n)
    @test_throws ArgumentError WarmupHMC.restore_reparam_controls!(ir, fill(0.6, n - 1))
    @test_throws ArgumentError WarmupHMC.restore_reparam_controls!(ir, fill(1.5, n))
    rows = adaptive_centering_controls(rp, m.sb)
    @test [r.index for r in rows] == vec(m.coords.totals)
    @test all(r -> r.c == 0.6 && r.target == 1.0, rows)
    # `:midpoint` moves random-effect cells only: totals keep the compiled start.
    midpoint = WarmupHMC.reparametrizer(adaptive_centering_problem(m.sb, m.problem,
        AutoEnzyme(); centeredness=:midpoint))
    @test WarmupHMC.reparam_controls(midpoint) == midpoint.targets
end

@testset "online fit" begin
    m = CASES[1][2]
    rp = adaptive_centering_problem(m.sb, m.problem, AutoEnzyme(); centeredness=0.0)
    _, init = WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(rp), m.physical)
    fit = adaptive_warmup_mcmc(Xoshiro(27), rp; init, n_draws=1000, nonlinear_adapt=true,
        monitor_ess=false)
    @test size(fit.posterior_position, 2) >= 1000
    @test all(isfinite, fit.posterior_position)
    @test fit.n_divergent_samples <= 5
    controls = WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(rp))
    @test all(c -> 0 <= c <= 1, controls)
    # Draws come back in the compiled frame: the totals' mean tracks the data.
    totals = fit.posterior_position[vec(m.coords.totals), :]
    @test abs(mean(totals) - mean(DATA.y .- 0.5 .* DATA.x)) < 0.5
    println("MEAN_LOCATION_ONLINE controls=", round.(controls; digits=1),
        " divergences=", fit.n_divergent_samples); flush(stdout)
end

# Totals beside an ordinary correlated block: the ordinary cells keep the
# wrapper they would have without totals (per-pair for the declared order,
# learned-order otherwise), centred on their population coefficients, and the
# mean-located totals join beside it.
const MIXED = @brm begin
    mu ~ 1 + x + (1 + x | h)
    log(sigma) ~ 1 + (1 | g)
    effect(mu, :) ~ Normal(0, 2)
    effect(sigma, Intercept) ~ Normal(0, 1)
    y ~ Normal(mu, sigma)
end

@testset "totals beside ordinary cells: $order" for order in (:declared, :learned)
    sb = SBBRMI(MIXED(merge(DATA, (; h=mod1.(eachindex(DATA.y), 3)))); mod=@__MODULE__,
                total_groups=[:g])
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(mktempdir(), "mixed_$order.stan"))
    names = BridgeStan.param_unc_names(problem.model)
    ordinary = BRM.AdaptiveCenteringBlock[b for b in adaptive_centering_blocks(sb, names)
                                          if b isa BRM.AdaptiveCenteringBlock]
    _, joint = EXT._adaptive_joint_centering_reparametrizer(ordinary, BRM._HSGPAdaptiveCenteringBlock[])
    totals = BRM._total_centering_columns(sb, names).indices
    n = length(joint.pairs) + length(totals)
    c = collect(range(0.15, 0.85; length=n))
    rp = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=c, ranef_order=order)
    ir = WarmupHMC.reparametrizer(rp)
    @test ir isa EXT.BRMTotalsBesideCenteringReparametrization
    @test ir.primary isa (order === :declared ? WarmupHMC.IndexedReparametrization :
                          EXT.BRMOrderedCenteringReparametrization)
    @test WarmupHMC.reparam_controls(ir) ≈ c
    rows = adaptive_centering_controls(rp, sb)
    @test [r.index for r in rows] == vcat(first.(joint.pairs), totals)
    # The ordinary cells keep their hierarchical centring on `mu`'s population
    # coefficients beside the totals; the totals have no such location.
    nranef = length(joint.pairs)
    @test all(r -> !isnothing(r.location), rows[1:nranef])
    @test all(r -> isnothing(r.location), rows[nranef+1:end])
    midpoint = WarmupHMC.reparametrizer(adaptive_centering_problem(sb, problem, AutoEnzyme();
        centeredness=:midpoint, ranef_order=order))
    @test WarmupHMC.reparam_controls(midpoint)[1:nranef] == fill(0.5, nranef)
    @test WarmupHMC.reparam_controls(midpoint)[nranef+1:end] == midpoint.totals.targets
    physical = 0.4 .* randn(Xoshiro(5), length(names))
    _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, physical)
    jac, mapped = ir(source)
    @test mapped ≈ physical atol = 1e-10
    @test first(logabsdet(fd_jacobian(ir, source))) ≈ jac atol = 1e-6
    lp, g = LogDensityProblems.logdensity_and_gradient(rp, source)
    @test lp ≈ LogDensityProblems.logdensity(problem, physical) + jac atol = 1e-8
    @test g ≈ fd_gradient(rp, source) atol = 1e-5 rtol = 1e-5
    # The default snapshot is independent of later control changes.
    snapshot = WarmupHMC.snapshot_reparametrization(ir)
    WarmupHMC.restore_reparam_controls!(ir, reverse(c))
    @test WarmupHMC.reparam_controls(ir) ≈ reverse(c)
    @test WarmupHMC.reparam_controls(snapshot) ≈ c
    @test first(snapshot(source)) ≈ jac
    @test_throws ArgumentError WarmupHMC.restore_reparam_controls!(ir, c[1:end-1])
    positions = hcat((source .+ 0.05 .* randn(Xoshiro(s), length(source)) for s in 1:6)...)
    gradients = hcat((last(LogDensityProblems.logdensity_and_gradient(rp, x)) for x in eachcol(positions))...)
    @test EXT._totals_beside_select!(ir, positions, gradients) isa Bool
    @test all(c -> c in EXT._CENTERING_CANDIDATES, WarmupHMC.reparam_controls(ir))
    if order === :declared
        fresh = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.5)
        _, init = WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(fresh), physical)
        fit = adaptive_warmup_mcmc(Xoshiro(31), fresh; init, n_draws=500, nonlinear_adapt=true,
            monitor_ess=false)
        @test all(isfinite, fit.posterior_position)
        # A correlated block on three groups diverges under every
        # parametrization. Measured per 500 draws, seeds 31-35: 9-98 with
        # totals beside it, 12-56 without totals (the per-pair online plan),
        # 7-44 with the beside controls held at 0.5. So this bound only
        # catches a broken sampler.
        @test fit.n_divergent_samples < 125
        println("MEAN_LOCATION_BESIDE divergences=", fit.n_divergent_samples); flush(stdout)
    end
end
