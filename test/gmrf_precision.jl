# Independent dense oracles are deliberately tiny. The implementation stores
# spatial sparse geometry and applies the full precision without forming it.
using Test, LinearAlgebra, SparseArrays, Random

module GMRFPrecisionFixture
using LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "src", "gmrf_precision.jl"))
include(joinpath(@__DIR__, "..", "src", "gmrf_interpolation.jl"))
end
const GM = GMRFPrecisionFixture

vertices = [0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5]
cells = [1 2 5; 2 3 5; 3 4 5; 4 1 5]
mesh = GM._brm_gmrf_mesh(vertices, cells)

@testset "GMRF semantic state axes without Cartesian label storage" begin
    channels, times = ["north", "south"], [0.0, 0.3, 1.7]
    layout = GM._BRMGMRFStateLayout(5, channels, times)
    channels[1], times[1] = "changed", -1
    @test layout.ncoordinates == 30
    # The stored axis identities survive mutation of the caller's inputs.
    @test GM._brm_gmrf_state_coordinate(layout, 1) ==
        (; vertex=1, channel="north", time=0.0)
    @test GM._brm_gmrf_state_coordinate(layout, 30) ==
        (; vertex=5, channel="south", time=1.7)
    vertex_reads = [5 1 3; 4 2 1; 3 5 2]
    row_times, row_channels = [1.7, 0.0, 0.3], ["south", "north", "south"]
    indices = GM._brm_gmrf_state_indices(vertex_reads, row_times, row_channels, layout)
    @test all(GM._brm_gmrf_state_coordinate(layout, indices[r, c]) ==
        (; vertex=vertex_reads[r, c], channel=row_channels[r], time=row_times[r])
        for r in 1:3, c in 1:3)
    @test_throws BoundsError GM._brm_gmrf_state_coordinate(layout, 31)
    @test_throws ArgumentError GM._brm_gmrf_state_indices(zeros(Int, 3, 3), row_times, row_channels, layout)
    @test_throws ArgumentError GM._BRMGMRFStateLayout(5, [1, 1], [0.0])
    @test_throws ArgumentError GM._BRMGMRFStateLayout(5, [1], [0.3, 0.0])
    @test_throws OverflowError GM._BRMGMRFStateLayout(typemax(Int), [1, 2], [0.0])
    # This proves bounded metadata storage only, not a viable field allocation.
    huge = GM._BRMGMRFStateLayout(100_000, collect(1:38), collect(0.0:8199.0))
    @test huge.ncoordinates == 31_160_000_000
    @test Base.summarysize(huge) < 100_000
    @test GM._brm_gmrf_state_coordinate(huge, huge.ncoordinates) ==
        (; vertex=100_000, channel=38, time=8199.0)
end

@testset "Local mesh interpolation and frozen state identities" begin
    x = [0.1, 0.2, 0.9, 0.7, 0.0, 1.0, 0.5, 0.5]
    y = [0.2, 0.1, 0.6, 0.8, 0.0, 1.0, 0.5, 0.0]
    (; indices, weights) = GM._brm_gmrf_interpolation(mesh, x, y)
    @test size(indices) == (8, 3)
    @test sum(weights; dims=2) ≈ ones(8, 1)
    @test all(>=(0), weights)
    for (a, b, c) in ((1.0, 0.0, 0.0), (0.2, 3.0, -1.5))
        field = a .+ b .* vertices[:, 1] .+ c .* vertices[:, 2]
        @test vec(sum(weights .* field[indices]; dims=2)) ≈ a .+ b .* x .+ c .* y
    end
    reverse_mesh = GM._brm_gmrf_mesh(vertices, cells[:, [3, 2, 1]])
    reversed = GM._brm_gmrf_interpolation(reverse_mesh, x, y)
    field = [0.2, -0.8, 1.5, -0.1, 0.5]
    @test sum(weights .* field[indices]; dims=2) ≈
        sum(reversed.weights .* field[reversed.indices]; dims=2)
    @test_throws ArgumentError GM._brm_gmrf_interpolation(mesh, [-0.01], [0.2])
    @test_throws ArgumentError GM._brm_gmrf_interpolation(mesh, [NaN], [0.2])
    @test_throws DimensionMismatch GM._brm_gmrf_interpolation(mesh, x, y[1:2])
    row_times = [2, 2, 2, 2, 1, 1, 1, 1]
    row_channels = ["b", "a", "a", "b", "b", "a", "b", "a"]
    mapped = GM._brm_gmrf_state_indices(indices, row_times, row_channels, [1, 2], ["a", "b"], 5)
    state = reshape(1:20, 5, 2, 2)
    @test all(mapped[i, a] == state[indices[i, a], row_channels[i] == "a" ? 1 : 2, row_times[i]]
        for i in 1:8, a in 1:3)
    @test_throws ArgumentError GM._brm_gmrf_state_indices(indices, row_times, row_channels, [1], ["a", "b"], 5)
    @test_throws ArgumentError GM._brm_gmrf_state_indices(indices, row_times, row_channels, [1, 2], ["a"], 5)
    # A larger triangulation exercises tree branches and many shared edges.
    side = 17
    xy = hcat(repeat(Float64.(0:(side-1)); outer=side),
        repeat(Float64.(0:(side-1)); inner=side))
    triangles = reduce(vcat, ([i + side*j i + 1 + side*j i + side*(j+1);
        i + 1 + side*j i + 1 + side*(j+1) i + side*(j+1)]
        for j in 0:(side-2) for i in 1:(side-1)))
    large = GM._brm_gmrf_mesh(xy, triangles)
    locator = GM._brm_gmrf_locator(large)
    @test length(locator.bounds) > 1
    xx, yy = repeat(collect(0.0:0.5:16); inner=33), repeat(collect(0.0:0.5:16); outer=33)
    local_rows = GM._brm_gmrf_interpolation(large, xx, yy, locator)
    affine = 2 .+ xy[:, 1] .- 3xy[:, 2]
    @test vec(sum(local_rows.weights .* affine[local_rows.indices]; dims=2)) ≈ 2 .+ xx .- 3yy
end

@testset "Matérn finite-element operator and fixed-pattern factor" begin
    @test sum(mesh.mass) ≈ 1
    @test mesh.stiffness * ones(5) ≈ zeros(5) atol=1e-14
    @test mesh.mass ≈ [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3]
    # Four unit right triangles, assembled independently from their gradients.
    @test Matrix(mesh.stiffness) ≈ [1 0 0 0 -1; 0 1 0 0 -1;
        0 0 1 0 -1; 0 0 0 1 -1; -1 -1 -1 -1 4]
    for ell in (0.03, 0.4, 2.0, 20.0)
        k2 = 2 / ell^2
        K = Matrix(mesh.stiffness) + k2 * Diagonal(mesh.mass)
        Q = K * Diagonal(1 ./ mesh.mass) * K / (4pi * k2)
        @test GM._brm_gmrf_spatial_logdet(mesh, ell) ≈ logdet(Symmetric(Q)) rtol=2e-9
        F = GM._brm_gmrf_ldl(k2, mesh.factor_mass, mesh.factor_stiffness, mesh.factor_pattern)
        p = mesh.factor_pattern.permutation
        L = Matrix(sparse(mesh.factor_pattern.rowval,
            reduce(vcat, (fill(j, mesh.factor_pattern.colptr[j + 1] - mesh.factor_pattern.colptr[j]) for j in 1:5)),
            F.lower, 5, 5))
        @test L * Diagonal(F.diagonal) * L' ≈ K[p, p] rtol=1e-12
        b = [1.0, -2, 3, 0.2, -0.8]
        x = similar(b)
        GM._brm_gmrf_ldl_solve!(x, b, F, mesh.factor_pattern)
        @test K[p, p] * x ≈ b rtol=1e-11
        actual = similar(b)
        GM._brm_gmrf_spatial_mul!(actual, b, mesh, ell, similar(b))
        @test actual ≈ Q * b rtol=1e-12
    end
    @test_throws ArgumentError GM._brm_gmrf_mesh(vertices, [1 2 2])
    @test_throws ArgumentError GM._brm_gmrf_mesh(vertices, vcat(cells, cells[1:1, :]))
    @test_throws ArgumentError GM._brm_gmrf_mesh([vertices; 2 2], cells)
    @test_throws DomainError GM._brm_gmrf_spatial_logdet(mesh, 0.0)
end

function dense_prior(mesh, times, ell_s, ell_t, omega)
    n = length(mesh.mass)
    k2 = 2 / ell_s^2
    K = Matrix(mesh.stiffness) + k2 * Diagonal(mesh.mass)
    Qs = K * Diagonal(1 ./ mesh.mass) * K / (4pi * k2)
    # Independent covariance definition of the stationary OU law.
    Ct = [exp(-abs(a - b) / ell_t) for a in times, b in times]
    C = kron(Ct, kron(omega, inv(Symmetric(Qs))))
    (; C, Q=inv(Symmetric(C)))
end

@testset "Normalized space-time-channel law and precision action" begin
    rng = Xoshiro(52026)
    omega = [1.0 0.55; 0.55 1.0]
    L = Matrix(cholesky(Symmetric(omega)).L)
    for times in ([0.0], [0.0, 0.001, 1.5, 80.0]), ell_s in (0.4, 2.0), ell_t in (0.1, 2.5)
        state = 0.2randn(rng, 5, 2, length(times))
        (; C, Q) = dense_prior(mesh, times, ell_s, ell_t, omega)
        u = vec(state)
        expected = -(length(u) * log(2pi) + logdet(Symmetric(C)) + dot(u, Q * u)) / 2
        @test GM._brm_gmrf_logpdf(state, mesh, times, ell_s, ell_t, L) ≈ expected rtol=2e-9
        out = similar(state)
        GM._brm_gmrf_precision_mul!(out, state, mesh, times, ell_s, ell_t, L)
        @test vec(out) ≈ Q * u rtol=2e-9
        # The negative state gradient is Q*u, independently differenced.
        h = 1e-6
        i = min(7, length(state))
        plus, minus = copy(state), copy(state)
        plus[i] += h; minus[i] -= h
        derivative = (GM._brm_gmrf_logpdf(plus, mesh, times, ell_s, ell_t, L) -
                      GM._brm_gmrf_logpdf(minus, mesh, times, ell_s, ell_t, L)) / (2h)
        @test derivative ≈ -out[i] rtol=2e-6 atol=2e-6
    end
    @test GM._brm_gmrf_times([5, 1, 5, 2]) == [1, 2, 5]
    @test_throws ArgumentError GM._brm_gmrf_times([1, Inf])
    @test_throws ArgumentError GM._brm_gmrf_logpdf(zeros(5, 2, 2), mesh, [0.0, 0.0], 1.0, 1.0, L)
end
