# A sparse Laplace cell with no sums has two routes: compact operators, or its
# factor outer when a compact cell would compute a value whole. They must be the
# same target, including a cell wider than the HVP batch, a retained plan and a copy.
#
# RUN: julia --project=test test/laplace_cell_routes.jl
#
# A model that takes the factor route on its own needs a real disagreement between
# RK's strict preparations, so this forces the route by overriding the locality
# report for one testset and restoring it afterwards.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems
const LDP = LogDensityProblems

route_field(x, beta) = [exp(beta * t) for t in x]
route_opaque(z) = (v = copy(z); v .+= 0.0; v)

route_fd(f, x; h=1e-5) = map(eachindex(x)) do i
    e = zeros(length(x)); e[i] = h
    (f(x + e) - f(x - e)) / (2h)
end

function route_narrow()
    data = (; n=4, grid=collect(range(0.1, 0.9; length=4)), y=ones(Int, 4))
    RKBRMI(@brm data begin
        beta ~ Normal(0, 1)
        z[1:n] .~ Normal.(0, 1)
        f = route_field(grid, beta)
        @plate for i in eachindex(grid)
            y[i] ~ Poisson(exp(z[i]) * f[i])
        end
    end), (:z,)
end

function route_wide()
    n = BRM._BRM_SPL_CELL_HVP_WIDTH + 2
    data = (; a=["a$i" for i in 1:n], rows=[collect(1:n), collect(1:3)],
        weights=[0.5 + 0.5 * cos(i) for i in 1:n], y=[0.2, 0.3])
    RKBRMI(@brm data begin
        eta ~ 1 + (1 | a)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(exp(0.2 * dot(eta[rows[k]], weights[rows[k]])), 0.7)
        end
    end), (:a,)
end

# A stiff observation whose mean is nonlinear in its integrated effects, with a
# population intercept aliased with an integrated one. The log determinant is very
# sensitive to the stiff coordinate, and its explicit and implicit derivatives cancel on
# the intercept, so only an accurate solve of h v = s leaves that cancellation.
function route_stiff()
    data = (; n=4, m=2, rows=[[1, 2], [3, 4]], y=[0.2, 0.3])
    RKBRMI(@brm data begin
        off ~ Normal(0, 1)
        cal[1:m] .~ Normal.(0, 1)
        z[1:n] .~ Normal.(0, 1)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(off + cal[k] + 0.5 * dot(z[rows[k]], z[rows[k]]), 3e-4)
        end
    end), (:cal, :z)
end

is_factor(group) = group isa BRM._BRMSumCellGroup

@stestset "both routes of a no-sums cell are the same target" begin
    fixtures = map((narrow=route_narrow, wide=route_wide)) do build
        rk, integrate = build()
        p = brm_laplace_problem(rk; integrate, ad_backend=NLME_AD)
        @test !any(is_factor, p.cells)
        points = [0.1 .* sin.(eachindex(p.population)), -0.2 .* cos.(eachindex(p.population))]
        (; rk, integrate, points, compact=[LDP.logdensity_and_gradient(p, phi) for phi in points])
    end
    @eval BRM._rk_spl_cell_whole(::ReactiveKernelsPPL.CellQuery) = true
    try
        for (shape, (; rk, integrate, points, compact)) in pairs(fixtures)
            @testset "$shape" begin
                p = Base.invokelatest(brm_laplace_problem, rk; integrate, ad_backend=NLME_AD)
                @test all(is_factor, p.cells)
                # The outer compiles a bounded batch however many directions it takes.
                @test all(length(first(g.cells).outer.dirs) <= BRM._BRM_SPL_CELL_HVP_WIDTH for g in p.cells)
                for (phi, (value, gradient)) in zip(points, compact)
                    v, g = Base.invokelatest(LDP.logdensity_and_gradient, p, phi)
                    @test v ≈ value rtol=1e-9
                    @test g ≈ gradient rtol=1e-6 atol=1e-6
                    @test g ≈ route_fd(x -> Base.invokelatest(LDP.logdensity, p, x), phi) rtol=2e-5 atol=2e-5
                end
                plan = Base.invokelatest(brm_laplace_plan, rk; ad_backend=NLME_AD, integrate, draws=12)
                reused = Base.invokelatest(brm_laplace_problem, rk; ad_backend=NLME_AD, integrate=plan)
                v, g = Base.invokelatest(LDP.logdensity_and_gradient, reused, points[1])
                @test v ≈ compact[1][1] rtol=1e-9
                @test g ≈ compact[1][2] rtol=1e-6 atol=1e-6
                copied = Base.invokelatest(LDP.logdensity_and_gradient, deepcopy(reused), points[2])
                @test all(isapprox.(copied, compact[2]; rtol=1e-6, atol=1e-6))
            end
        end
    finally
        @eval BRM._rk_spl_cell_whole(query::ReactiveKernelsPPL.CellQuery) =
            any(entry -> entry.status === :whole, ReactiveKernels.partition_locality(query))
    end
end

# The compact route's Cholesky of h never sees the factor route's augmented system,
# so it is an independent reference. Reported on a bounded ODL target, where a
# residual of 0.8% in that system moved a gradient of 395 to 19214 (snag
# sparse-laplace-c-b7cc5e58); on this fixture it gave -50401 for -2.16.
@stestset "a stiff cell's factor route keeps the marginal gradient" begin
    rk, integrate = route_stiff()
    p = brm_laplace_problem(rk; integrate, ad_backend=NLME_AD)
    @test !any(is_factor, p.cells)
    points = [0.1 .* sin.(eachindex(p.population)), -0.2 .* cos.(eachindex(p.population))]
    compact = [LDP.logdensity_and_gradient(p, phi) for phi in points]
    @eval BRM._rk_spl_cell_whole(::ReactiveKernelsPPL.CellQuery) = true
    try
        q = Base.invokelatest(brm_laplace_problem, rk; integrate, ad_backend=NLME_AD)
        @test all(is_factor, q.cells)
        for (phi, (value, gradient)) in zip(points, compact)
            v, g = Base.invokelatest(LDP.logdensity_and_gradient, q, phi)
            @test v ≈ value rtol=1e-8
            @test g ≈ gradient rtol=1e-6
            @test g ≈ route_fd(x -> Base.invokelatest(LDP.logdensity, q, x), phi) rtol=1e-4
        end
    finally
        @eval BRM._rk_spl_cell_whole(query::ReactiveKernelsPPL.CellQuery) =
            any(entry -> entry.status === :whole, ReactiveKernels.partition_locality(query))
    end
end

@stestset "a value computed whole in a compact cell is reported, and refused by name" begin
    data = (; n=2, y=[[0.5, 0.4], [0.3]])
    rk = RKBRMI(@brm data begin
        z[1:n] .~ Normal.(0, 1)
        field = route_opaque(z)
        @plate for k in eachindex(y)
            y[k] ~ normal(field[k], 1.0)
        end
    end)
    layout = BRM._brm_spl_layout(rk, (:z,))
    u0 = zeros(length(layout.names))
    query = BRM._rk_nlme_cell_query(rk, (:y,); layout.partitioned, u0, locality=:ignore)
    @test BRM._rk_spl_cell_whole(query)
    # The helper hides its rows from both routes, so no route evaluates it whole.
    @test_throws ReactiveKernels.PartitionLocalityError brm_laplace_problem(rk; integrate=(:z,),
        ad_backend=NLME_AD)
end
