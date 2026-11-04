# A cell's Hessian block compiles at a bounded Enzyme batch width: a cell with
# more integrated coordinates takes several batches of the same program. The
# batches must give the same block and target as one batch over every direction
# (compiled here only where that is still cheap), and a much wider cell must
# not compile a wider program at all.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Treebars
const LDP = LogDensityProblems

# One nonlinear ordinary cell per entry of `rows`, each reading its rows' effects.
function hvp_width_fixture(n; short=n)
    data = (; a=["a$i" for i in 1:n], rows=[collect(1:n), collect(1:short)],
        weights=[0.5 + 0.5 * cos(i) for i in 1:n], y=[0.2, 0.3])
    RKBRMI(@brm data begin
        eta ~ 1 + (1 | a)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(exp(0.2 * dot(eta[rows[k]], weights[rows[k]])), 0.7)
        end
    end)
end

hvp_width_problem(rk; kwargs...) = brm_laplace_problem(rk; ad_backend=NLME_AD,
    integrate=(:a,), max_cell_width=256, kwargs...)

# The ordinary cell group and a packed point with distinct values.
function hvp_width_group(p)
    group = only(g for g in p.cells if hasproperty(g, :dirs))
    u = 0.1 .* cos.(1:length(p.u))
    group, u
end

function cell_block(ad, group, c, u)
    BRM._brm_spl_cell_directions!(group.dirs, c.lpos, 1.0)
    try
        BRM._rk_spl_cell_newton(ad, u) do batch
            g, columns = BRM._rk_spl_cell_block(ad, group.dirs, u, c.sel, c.i;
                context=batch, used=length(c.lz))
            copy(g), map(copy, columns)
        end
    finally
        BRM._brm_spl_cell_directions!(group.dirs, c.lpos, 0.0)
    end
end

@stestset "batched cell blocks equal one batch over every direction" begin
    width = BRM._BRM_SPL_CELL_HVP_WIDTH
    # Past one batch, with a partial last one, and a shorter second cell.
    n = 2width + 3
    rk = hvp_width_fixture(n; short=width + 1)
    p = hvp_width_problem(rk)
    group, u = hvp_width_group(p)
    K = length(group.dirs)
    @test K == n
    @test group.ad.width === Val(width)
    wide = BRM._rk_spl_cell_ad(rk, (group.name,), copy(p.u), first(group.cells).sel, NLME_AD, K;
        width=K, partitioned=BRM._brm_spl_layout(rk, (:a,)).partitioned)
    @test wide.width === Val(K)
    @test length(group.cells) == 2
    @test sort(length.(getfield.(group.cells, :lz))) == [width + 1, n]
    for c in group.cells
        g, columns = cell_block(group.ad, group, c, u)
        g_wide, columns_wide = cell_block(wide, group, c, u)
        @test g ≈ g_wide rtol=1e-12
        @test length(columns) == length(c.lz)
        @test all(k -> columns[k] ≈ columns_wide[k], eachindex(columns))
        # A later batch must not overwrite the columns of an earlier one.
        @test allunique(objectid.(columns))
        # The block is the gradient's own derivative, independently of Enzyme's batches.
        h = 1e-5
        for k in (1, width, width + 1, length(c.lz))
            e = zeros(length(u)); e[c.sel[c.lpos[k]]] = h
            plus, _ = cell_block(group.ad, group, c, u .+ e)
            minus, _ = cell_block(group.ad, group, c, u .- e)
            @test (plus .- minus) ./ 2h ≈ columns[k] rtol=1e-6 atol=1e-7
        end
    end
    @test all(iszero, Iterators.flatten(group.dirs))

    # The whole target, value and gradient, through either preparation.
    control = hvp_width_problem(rk)
    position = findfirst(g -> hasproperty(g, :dirs), control.cells)
    control.cells[position] = merge(control.cells[position], (; ad=wide))
    φ = 0.05 .* sin.(1:length(p.population))
    v, d = LDP.logdensity_and_gradient(p, φ)
    v_control, d_control = LDP.logdensity_and_gradient(control, φ)
    @test isfinite(v) && all(isfinite, d)
    @test v ≈ v_control rtol=1e-10
    @test d ≈ d_control rtol=1e-8
end

@stestset "a much wider cell compiles the same batch width" begin
    width = BRM._BRM_SPL_CELL_HVP_WIDTH
    n = 129
    rk = hvp_width_fixture(n)
    p = hvp_width_problem(rk)
    group, u = hvp_width_group(p)
    @test length(group.dirs) == n
    @test group.ad.width === Val(width)
    c = first(group.cells)
    g, columns = cell_block(group.ad, group, c, u)
    @test length(columns) == n
    @test all(isfinite, g) && all(col -> all(isfinite, col), columns)
    # The block is symmetric across batches.
    H = [columns[j][c.lpos[i]] for i in 1:n, j in 1:n]
    @test H ≈ H' rtol=1e-10
    φ = zeros(length(p.population))
    v, d = LDP.logdensity_and_gradient(p, φ)
    @test isfinite(v) && all(isfinite, d)
end

struct HVPWidthRaising{E}
    err::E
end
ReactiveKernels.ad_gradient_and_hvp(h::HVPWidthRaising, _directions, _base, _active,
    _indices, _i) = throw(h.err)

# A stub, because which cells Enzyme can take changes: this holds whatever it takes.
# The group keeps its width through the move, and its batches the same columns.
@stestset "a batched group moves to reverse over forward at its width" begin
    width = BRM._BRM_SPL_CELL_HVP_WIDTH
    n = 2width + 3
    p = hvp_width_problem(hvp_width_fixture(n; short=width + 1))
    group, u = hvp_width_group(p)
    ad = group.ad
    @test length(group.dirs) == n
    blocks = [cell_block(ad, group, c, u) for c in group.cells]
    @test !ad.reverse_over_forward
    ad.hvp = HVPWidthRaising(Enzyme.Compiler.EnzymeRuntimeActivityError("stub", nothing, nothing))
    for (c, (g, columns)) in zip(group.cells, blocks)
        g_moved, columns_moved = cell_block(ad, group, c, u)
        @test ad.reverse_over_forward
        @test ad.width === Val(width)
        @test g_moved ≈ g rtol=1e-12
        @test length(columns_moved) == length(c.lz)
        @test all(k -> columns_moved[k] ≈ columns[k], eachindex(columns))
    end
    @test all(iszero, Iterators.flatten(group.dirs))
end

# Forwards its first calls and then raises, so a move happens partway through a sweep.
mutable struct HVPWidthRaisingLater{H,E}
    hvp::H
    err::E
    calls::Int
end
function ReactiveKernels.ad_gradient_and_hvp(h::HVPWidthRaisingLater, directions, base, active,
        indices, i)
    h.calls == 0 && throw(h.err)
    h.calls -= 1
    ReactiveKernels.ad_gradient_and_hvp(h.hvp, directions, base, active, indices, i)
end

@stestset "a move partway through a sweep counts every cell once" begin
    data = (; n=4, grid=collect(range(0.1, 0.9; length=4)), y=ones(Int, 4))
    rk = RKBRMI(@brm data begin
        beta ~ Normal(0, 1)
        z[1:n] .~ Normal.(0, 1)
        @plate for i in eachindex(grid)
            y[i] ~ Poisson(exp(z[i] + beta * grid[i]))
        end
    end)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:z,))
    group = only(g for g in p.cells if hasproperty(g, :dirs))
    @test length(group.cells) == 4
    φ = [0.2]
    v, d = LDP.logdensity_and_gradient(p, φ)
    activity = Enzyme.Compiler.EnzymeRuntimeActivityError("stub", nothing, nothing)
    group.ad.hvp = HVPWidthRaisingLater(group.ad.hvp, activity, 2)
    v_moved, d_moved = LDP.logdensity_and_gradient(p, φ)
    @test group.ad.reverse_over_forward
    @test v_moved ≈ v rtol=1e-10
    @test d_moved ≈ d rtol=1e-8
    @test all(iszero, Iterators.flatten(group.dirs))
end

@stestset "plan and problem report their preparation through Treebars" begin
    rk = hvp_width_fixture(5)
    report = Ref("")
    plan = Treebars.with_progress(:state; description="plan") do node
        out = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=(:a,), progress=node)
        report[] = Treebars.render_text(node)
        out
    end
    @test only(t for t in plan.terms if t.grouping === :a).admitted
    @test occursin("Laplace plan", report[])
    @test occursin("Cell width screening", report[])
    @test occursin("Observation `y`, 2 cells", report[])
    @test occursin(r"Compiling a Hessian block: \d+ coordinates, 5 integrated", report[])
    target = Treebars.with_progress(:state; description="problem") do node
        out = hvp_width_problem(rk; progress=node)
        report[] = Treebars.render_text(node)
        out
    end
    @test occursin("Compiling a Hessian block", report[])
    @test !occursin("Laplace plan", report[])
    # Without `integrate` the structured path has no such reporting: refused, not ignored.
    @test_throws "needs `integrate`" brm_laplace_problem(rk; ad_backend=NLME_AD, progress=:state)
    # No `progress` compiles at first use and gives the same target.
    quiet = hvp_width_problem(rk)
    φ = 0.05 .* sin.(1:length(quiet.population))
    @test LDP.logdensity(quiet, φ) ≈ LDP.logdensity(target, φ) rtol=1e-12
end
