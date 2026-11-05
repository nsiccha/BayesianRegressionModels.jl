# Native reverse AD must include the learned spatial normalizer, irregular OU
# transition variance and channel correlation, not just a state quadratic.
include("gmrf_precision.jl")
using Enzyme

function gmrf_density_coordinates(u, geometry, times)
    n = length(geometry.mass) * 2 * length(times)
    state = reshape(view(u, 1:n), length(geometry.mass), 2, length(times))
    spatial_length, temporal_length = exp(u[n + 1]), exp(u[n + 2])
    rho = tanh(u[n + 3])
    L = [1.0 0.0; rho sqrt(1 - rho^2)]
    GM._brm_gmrf_logpdf(state, geometry, times, spatial_length, temporal_length, L)
end

@testset "Native Enzyme state, range and correlation derivatives" begin
    times = [0.0, 0.2, 3.0]
    rng = Xoshiro(725)
    u = [0.05randn(rng, 30); log(0.7); log(1.4); atanh(0.45)]
    gradient = zeros(length(u))
    Enzyme.autodiff(Enzyme.Reverse, gmrf_density_coordinates, Enzyme.Active,
        Enzyme.Duplicated(u, gradient), Enzyme.Const(mesh), Enzyme.Const(times))
    for i in eachindex(u)
        plus, minus = copy(u), copy(u)
        h = 2e-6 * max(1, abs(u[i]))
        plus[i] += h; minus[i] -= h
        expected = (gmrf_density_coordinates(plus, mesh, times) -
            gmrf_density_coordinates(minus, mesh, times)) / (2h)
        @test gradient[i] ≈ expected rtol=2e-6 atol=2e-6
    end
    # A zero field still has nonzero range derivatives from its normalizer.
    u[1:30] .= 0
    fill!(gradient, 0)
    Enzyme.autodiff(Enzyme.Reverse, gmrf_density_coordinates, Enzyme.Active,
        Enzyme.Duplicated(u, gradient), Enzyme.Const(mesh), Enzyme.Const(times))
    @test all(iszero, gradient[1:30])
    @test all(x -> isfinite(x) && abs(x) > 1e-4, gradient[31:33])
end
