# test/adaptive_centering_window_transport.jl — the window-end transport keeps
# every cell's model point (snag read-brm-s-ordin-8860f258).
#
# A location accessor re-applies earlier cells' maps with their controls: a
# correlated block's earlier margins, a nested cell's containing grouping. At a
# restarting window WarmupHMC snapshots the old map as
# `IndexedReparametrization(copy(ir.pairs))`, commits the new controls, runs the
# scoring plan's `synchronize!`, and only then transports the retained pool and
# the active point through the snapshot. The snapshot must therefore still be
# the old map after the synchronization. Before the fix its accessors read the
# plan's state, so a cell whose OWN control did not change moved to another
# model point whenever an earlier cell's control did (0.31 on a K = 3 block,
# 0.35 on a nested grouping).
#
# The oracles do not use the wrapper's accessors after the change: the old
# map's values are taken BEFORE the controls move, and the end-to-end check
# compares against a `deepcopy` of the wrapper taken before WarmupHMC's own
# `find_reparametrization!`.
#
# Run: julia --project=test test/adaptive_centering_window_transport.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, LKJCholesky, Normal, TDist

const BRM = BayesianRegressionModels
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)
const BS = StanBlocks.BridgeStan

with_controls(pairs, controls) = map(pairs, controls) do (idx, value), c
    idx => WarmupHMC.Reparametrization(value.target, WarmupHMC.PartiallyCentered(c),
                                       value.args...)
end
controls_of(ir) = [value.source.c for (_, value) in ir.pairs]

# One window boundary in WarmupHMC's order, with the controls `new` committed
# where `optimize!` would commit them. `stay[p]` marks the cells whose own
# control does not change.
function check_window_boundary(rp, x, new, stay)
    ir = WarmupHMC.reparametrizer(rp)
    old_ljac, old_y = ir(x)
    snapshot = WarmupHMC.IndexedReparametrization(copy(ir.pairs))
    ir.pairs .= with_controls(ir.pairs, new)
    WarmupHMC._synchronize_scoring!(rp)
    # The snapshot is still the old map, bit for bit.
    @test snapshot(x) == (old_ljac, old_y)
    x_new, model = similar(x), similar(x)
    WarmupHMC._transport_position!(x_new, model, ir, snapshot, x)
    idx = first.(ir.pairs)
    @test maximum(abs, last(ir(x_new))[idx] .- old_y[idx]) < 1e-12
    # A cell whose own control did not change keeps its coordinate: what it is
    # centred on is a function of the model point.
    @test maximum(abs, x_new[idx[stay]] .- x[idx[stay]]) < 1e-12
    # The scoring frames read the committed controls.
    frame = WarmupHMC.candidate_scoring_plan(rp).prepare(ir, x, zero(x))
    @test frame.source == new
    # A rollback restores the old map.
    WarmupHMC._restore_reparametrization!(ir, snapshot)
    WarmupHMC._synchronize_scoring!(rp)
    @test ir(x) == (old_ljac, old_y)
    # A synchronization that changes no control keeps the accessors.
    bound = AC_EXT._centering_state(ir)
    WarmupHMC._synchronize_scoring!(rp)
    @test AC_EXT._centering_state(ir) === bound
end

struct FlatTarget
    d::Int
end
LogDensityProblems.dimension(p::FlatTarget) = p.d
LogDensityProblems.capabilities(::Type{FlatTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::FlatTarget, y) = -sum(abs2, y) / 2

@testset "correlated block (1 + x + t | subject): later margins keep their model point" begin
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x + t | subject)
        y ~ Normal(mu, sigma)
    end
    df = (; subject = repeat([11, 12], inner=3), x = collect(range(-1.0, 1.0, length=6)),
          t = repeat([0.0, 1.0, 2.0], 2), y = zeros(6))
    names = ["sigma", "pop_mu_beta_pop.1", "pop_mu_beta_pop.2",
             "r_mu_subject_L.1", "r_mu_subject_L.2", "r_mu_subject_L.3",
             "r_mu_subject_tau.1", "r_mu_subject_tau.2", "r_mu_subject_tau.3",
             ["r_mu_subject_z_flat.$i" for i in 1:6]...]
    sb = SBBRMI(builder(df); total_groups=(), mod=@__MODULE__)
    block = only(adaptive_centering_blocks(sb, names))
    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    ir.pairs .= with_controls(ir.pairs, fill(0.7, length(ir.pairs)))
    AC_EXT._sync_sources!(state, ir)
    rp = WarmupHMC.ReparametrizedProblem(ir, FlatTarget(length(names)), AutoEnzyme();
                                         scoring_plan=AC_EXT._candidate_scoring_plan(state))
    # The map's accessors own their controls: no synchronization writes them.
    @test AC_EXT._centering_state(ir) !== state
    x = randn(Xoshiro(20261114), length(names))
    x[block.log_scales] .= log.([0.8, 1.3, 0.6])
    terms = [state.pair_terms[p] for p in eachindex(ir.pairs)]
    check_window_boundary(rp, x, [k == 1 ? 0.2 : 0.7 for k in terms], terms .>= 2)
    check_window_boundary(rp, x, [k == 2 ? 0.0 : 0.7 for k in terms], terms .!= 2)
end

rng = Xoshiro(7)
G = 12
g = repeat(["g$j" for j in 1:G]; inner = 6)
band = ["b$(cld(parse(Int, s[2:end]), 4))" for s in g]   # 3 bands, each holding 4 g-levels
x, y = randn(rng, 72), randn(rng, 72)
data = (; g, band, x, y)
instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)
ranef_groups(rp, model) =
    [r.group for r in adaptive_centering_controls(rp, model) if r.family === :ranef]

const NESTED = SBBRMI(merge(@brm(begin
    mu ~ 1 + (1 | band) + (1 | g)
    y ~ Normal(mu, sigma)
end), @brm(begin
    sigma ~ Exponential(1)
end))(data); mod = @__MODULE__, total_groups = ())
const P_NESTED = instantiate(NESTED)

@testset "nested (1 | band) + (1 | g): nested cells keep their model point" begin
    rp = adaptive_centering_problem(NESTED, P_NESTED, AutoEnzyme(); centeredness = 0.7)
    groups = ranef_groups(rp, NESTED)
    @test length(groups) == length(WarmupHMC.reparametrizer(rp).pairs)
    q = 0.8 .* randn(Xoshiro(3), LogDensityProblems.dimension(rp))
    check_window_boundary(rp, q, [gr === :band ? 0.2 : 0.7 for gr in groups],
                          groups .=== :g)
end

@testset "joint wrapper, Student-t (1 + x | p | g): later margins keep their model point" begin
    tblock = SBBRMI(merge(@brm(begin
        mu ~ 1 + x + (1 + x | p | g)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
        ranef(:, p) ~ TDist(4)
    end))(data); mod = @__MODULE__, total_groups = ())
    rp = adaptive_centering_problem(tblock, instantiate(tblock), AutoEnzyme();
                                    centeredness = 0.7)
    @test AC_EXT._centering_state(WarmupHMC.reparametrizer(rp)) isa
          AC_EXT.BRMJointAdaptiveCenteringState
    rows = [r for r in adaptive_centering_controls(rp, tblock) if r.family === :ranef]
    first_margin = [r.margin.coefficient === :Intercept for r in rows]
    q = 0.5 .* randn(Xoshiro(5), LogDensityProblems.dimension(rp))
    check_window_boundary(rp, q, [m ? 0.2 : 0.7 for m in first_margin], .!first_margin)
end

@testset "grouped HSGP hsgp(t; by=g, cor=Omega): later levels keep their model point" begin
    icm = @brm begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=Omega)
        y ~ Normal(mu, 0.5)
    end
    t = repeat(collect(range(0.0, 10.0; length=16)), 3)
    levels = repeat(["c", "a", "b"]; inner=16)
    sb = SBBRMI(icm((; t, g = levels, y = sin.(t ./ 2) .+ 0.3 .* randn(Xoshiro(5), 48)));
                mod=@__MODULE__)
    rp = adaptive_centering_problem(sb, instantiate(sb), AutoEnzyme(); centeredness = 0.7)
    ir = WarmupHMC.reparametrizer(rp)
    state = AC_EXT._centering_state(ir)
    @test state isa AC_EXT.BRMJointAdaptiveCenteringState
    @test state.n_level > 0
    offset = state.n_ranef + state.n_r2d2 + state.n_hsgp
    first_level = falses(length(ir.pairs))
    for q in 1:state.n_level
        first_level[offset + q] = state.level.pair_level[q] == 1
    end
    @test any(first_level) && count(!, first_level) > 0
    q = 0.5 .* randn(Xoshiro(9), LogDensityProblems.dimension(rp))
    check_window_boundary(rp, q, [f ? 0.2 : 0.7 for f in first_level], .!first_level)
end

@testset "WarmupHMC's find_reparametrization! keeps the pool's and the active point's model points" begin
    rp = adaptive_centering_problem(NESTED, P_NESTED, AutoEnzyme(); centeredness = 1.0)
    ir = WarmupHMC.reparametrizer(rp)
    groups = ranef_groups(rp, NESTED)
    d = LogDensityProblems.dimension(rp)
    local_rng = Xoshiro(11)
    n = 60
    position = 0.6 .* randn(local_rng, d, n)
    gradient = similar(position)
    for j in 1:n
        gradient[:, j] = last(LogDensityProblems.logdensity_and_gradient(rp, position[:, j]))
    end
    q = 0.6 .* randn(local_rng, d)
    point = WarmupHMC.DynamicHMC.evaluate_ℓ(rp, q; strict=false)
    oracle = deepcopy(ir)                    # the old map, with its own controls
    before = controls_of(ir)
    old_position = copy(position)
    new_point = WarmupHMC.find_reparametrization!(rp, position, gradient, point)
    after = controls_of(ir)
    # Not vacuous: the selection moved some containing (band) cell.
    @test any(after[groups .=== :band] .!= before[groups .=== :band])
    idx = first.(ir.pairs)
    errors = [maximum(abs, last(ir(position[:, j]))[idx] .- last(oracle(old_position[:, j]))[idx])
              for j in 1:n]
    @test maximum(errors) < 1e-10
    @test maximum(abs, last(ir(new_point.q))[idx] .- last(oracle(q))[idx]) < 1e-10
end

println("adaptive_centering_window_transport.jl: all testsets passed")
