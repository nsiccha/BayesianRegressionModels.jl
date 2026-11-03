# This measures the native prior operator, not a Laplace fit or a log-determinant
# approximation. Dense correctness oracles stay in the small precision fixture.
include("gmrf_precision.jl")

function gmrf_operator_mesh(side)
    axis = collect(range(-4.0, 4.0; length=side))
    xy = hcat(repeat(axis; outer=side), repeat(axis; inner=side))
    triangles = reduce(vcat, ([i + side*j i + 1 + side*j i + side*(j+1);
        i + 1 + side*j i + 1 + side*(j+1) i + side*(j+1)]
        for j in 0:(side-2) for i in 1:(side-1)))
    GM._brm_gmrf_mesh(xy, triangles)
end

@testset "GMRF operator has bounded spatial work buffers and per-step overhead" begin
    L = Matrix(cholesky(Symmetric([1.0 0.4; 0.4 1.0])).L)
    rng = MersenneTwister(8391)
    for side in (41, 81)
        geometry = @timed gmrf_operator_mesh(side)
        mesh = geometry.value
        bytes = Int[]
        for nt in (24, 72, 168)
            # Unequal gaps exercise OU transitions without a regular-time shortcut.
            times = cumsum([0.75 + 0.25 * isodd(i) for i in 1:nt])
            x = randn(rng, side^2, 2, nt)
            y = randn(rng, size(x))
            qx, qy = similar(x), similar(y)
            apply!(out, state) = GM._brm_gmrf_precision_mul!(out, state, mesh,
                times, 0.5, 12.0, L)
            apply!(qx, x) # warm the exact signature before measuring
            measured = @timed apply!(qx, x)
            apply!(qy, y)
            @test all(isfinite, qx)
            @test dot(x, qx) > 0
            @test dot(x, qy) ≈ dot(qx, y) rtol=1e-10 atol=1e-7
            push!(bytes, measured.bytes)
            # Cumulative allocation includes small temporaries per channel/time,
            # even though the three numerical buffers are spatial-sized. Bound
            # both terms explicitly; a field-sized temporary fails this check.
            @test measured.bytes < 64 * side^2 * 2 + 256 * 2 * nt + 4096
            println((; vertices=side^2, channels=2, times=nt,
                coordinates=length(x), operator_seconds=measured.time,
                operator_allocated_bytes=measured.bytes,
                one_state_bytes=sizeof(x), mesh_bytes=Base.summarysize(mesh),
                mesh_seconds=geometry.time,
                spatial_factor_entries=length(mesh.factor_pattern.rowval)))
        end
        # Increasing T may allocate scalar/view carriers, not N-vertex vectors.
        @test bytes[end] - bytes[1] <= 256 * 2 * (168 - 24) + 1024
    end
end
