# Continuum checks use selected covariance entries through sparse solves;
# neither the full spatial nor the space-time covariance is materialized.
include("gmrf_precision.jl")
using SpecialFunctions: besselk

function regular_mesh(side, halfwidth)
    axis = collect(range(-halfwidth, halfwidth; length=side))
    xy = hcat(repeat(axis; outer=side), repeat(axis; inner=side))
    triangles = reduce(vcat, ([i + side*j i + 1 + side*j i + side*(j+1);
        i + 1 + side*j i + 1 + side*(j+1) i + side*(j+1)]
        for j in 0:(side-2) for i in 1:(side-1)))
    GM._brm_gmrf_mesh(xy, triangles)
end

function selected_covariance(mesh, points, ell)
    k2 = 2 / ell^2
    F = GM._brm_gmrf_ldl(k2, mesh.factor_mass, mesh.factor_stiffness, mesh.factor_pattern)
    rows = GM._brm_gmrf_interpolation(mesh, points[:, 1], points[:, 2])
    n, m = length(mesh.mass), size(points, 1)
    solved = zeros(n, m)
    for r in 1:m
        weights = zeros(n)
        for j in 1:3
            weights[rows.indices[r, j]] += rows.weights[r, j]
        end
        GM._brm_gmrf_ldl_solve!(view(solved, :, r),
            weights[mesh.factor_pattern.permutation], F, mesh.factor_pattern)
    end
    4pi * k2 * (solved' * (mesh.factor_mass .* solved))
end

@testset "Matérn continuum limit and finite boundary variance" begin
    # Practical range ρ=1, with a domain reaching four ranges in every direction.
    points = [0.0 0.0; 0.5 0; 1 0; 2 0; 0.37 0.23; 4 0]
    errors = Float64[]
    nodal_sd_errors = Float64[]
    for side in (41, 81)
        refined = regular_mesh(side, 4.0)
        covariance = selected_covariance(refined, points, 0.5)
        sd = sqrt.(diag(covariance))
        push!(nodal_sd_errors, abs(sd[1] - 1))
        target = [sqrt(8) * d * besselk(1, sqrt(8) * d) for d in (0.5, 1.0, 2.0)]
        actual = covariance[1, 2:4] ./ (sd[1] .* sd[2:4])
        push!(errors, maximum(abs.(actual - target)))
        @test all(abs.(sd[1:5] .- 1) .< 0.10)
        @test errors[end] < 0.03
        # Natural Neumann boundaries do not preserve unit marginal variance.
        # This guards against accidentally presenting σ as exact finite-mesh SD.
        @test sd[6] / sd[1] > 1.3
        println((; vertices=side^2, edge=8/(side-1), interior_sd=sd[1],
            off_vertex_sd=sd[5], max_correlation_error=errors[end],
            boundary_sd_ratio=sd[6]/sd[1], factor_entries=length(refined.factor_pattern.rowval)))
    end
    # The origin is a vertex on both meshes. The correlation queries mix
    # vertices and interpolated points on the coarse mesh, so their errors
    # need not decrease monotonically when their support changes on refinement.
    @test nodal_sd_errors[2] < nodal_sd_errors[1]
end
