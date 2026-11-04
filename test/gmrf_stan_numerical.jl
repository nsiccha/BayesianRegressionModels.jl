include("gmrf_preparation.jl")
using StanBlocks, Random, Enzyme

function gmrf_shared_density_coordinates(u, factor_data, times)
    n = length(factor_data.mass) * 2 * length(times)
    rho = tanh(u[n + 3])
    L = [1.0 0.0; rho sqrt(1 - rho^2)]
    BRM.brm_gmrf_logdensity(view(u, 1:n), exp(u[n + 1]), exp(u[n + 2]),
        L, times, values(factor_data)...)
end

function gmrf_native_density_coordinates(u, mesh, times)
    n = length(mesh.mass) * 2 * length(times)
    rho = tanh(u[n + 3])
    L = [1.0 0.0; rho sqrt(1 - rho^2)]
    state = reshape(view(u, 1:n), length(mesh.mass), 2, length(times))
    BRM._brm_gmrf_logpdf(state, mesh, times, exp(u[n + 1]), exp(u[n + 2]), L)
end

@testset "Stan sparse field density matches numerical reference" begin
    mesh = BRM._brm_gmrf_mesh(data.xy, data.triangles)
    factor_data = BRM._brm_gmrf_factor_data(mesh)
    times = [0.0, 0.3, 1.7]
    L = Matrix(cholesky(Symmetric(data.Omega)).L)
    state = reshape(0.03 .* sin.(1:30), 5, 2, 3)
    for spatial_length in (0.2, 1.3), temporal_length in (0.1, 2.1)
        expected = BRM._brm_gmrf_logpdf(state, mesh, times, spatial_length, temporal_length, L)
        actual = BRM.brm_gmrf_logdensity(vec(state), spatial_length, temporal_length, L,
            times, values(factor_data)...)
        @test actual ≈ expected rtol=2e-12 atol=2e-12
    end
    k2 = 2 / 0.7^2
    factor = BRM.brm_gmrf_factor(k2, 1.0, factor_data.mass,
        factor_data.gptr, factor_data.grow, factor_data.gval, factor_data.lptr,
        factor_data.lrow, factor_data.previous_ptr, factor_data.previous_col, factor_data.previous_pos)
    b = [0.2, -0.8, 0.1, 0.5, 1.3]
    old = copy(b)
    solution = BRM.brm_gmrf_solve(b, factor, factor_data.lptr, factor_data.lrow)
    @test b == old
    @test (Matrix(mesh.factor_stiffness) + k2 * Diagonal(mesh.factor_mass)) * solution ≈ b
end

@testset "Shared Julia/Stan helper differentiates learned normalizers" begin
    mesh = BRM._brm_gmrf_mesh(data.xy, data.triangles)
    factor_data = BRM._brm_gmrf_factor_data(mesh)
    times = [0.0, 0.3, 1.7]
    u = [0.03 .* sin.(1:30); log(0.7); log(1.4); atanh(0.45)]
    gradient = zeros(length(u))
    Enzyme.autodiff(Enzyme.Reverse, gmrf_shared_density_coordinates, Enzyme.Active,
        Enzyme.Duplicated(u, gradient), Enzyme.Const(factor_data), Enzyme.Const(times))
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        h = 2e-6 * max(1, abs(u[j]))
        plus[j] += h
        minus[j] -= h
        expected = (gmrf_native_density_coordinates(plus, mesh, times) -
            gmrf_native_density_coordinates(minus, mesh, times)) / (2h)
        @test gradient[j] ≈ expected rtol=2e-6 atol=2e-6
    end
    u[1:30] .= 0
    fill!(gradient, 0)
    Enzyme.autodiff(Enzyme.Reverse, gmrf_shared_density_coordinates, Enzyme.Active,
        Enzyme.Duplicated(u, gradient), Enzyme.Const(factor_data), Enzyme.Const(times))
    @test all(iszero, gradient[1:30])
    @test all(g -> isfinite(g) && abs(g) > 1e-4, gradient[31:33])
end
